import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=nn.ReLU(), dropout=0.5):
        super().__init__()
        layers = []
        drop = nn.Dropout(dropout)
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers += [nn.Linear(input_dim, hidden_dim), activation, drop]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), activation, drop]
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Custom3DCNN(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout3d(0.2)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=3)
        self.dropout2 = nn.Dropout3d(0.2)
        self.conv5 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv3d(128, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=4)
        self.dropout3 = nn.Dropout3d(0.2)
        self.fc = nn.Linear(hidden_dim * 3 * 3 * 4, hidden_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(self.pool1(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.dropout2(self.pool2(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.dropout3(self.pool3(x))
        return self.fc(x.view(x.size(0), -1))


class PatchEmbeddings(nn.Module):
    def __init__(self, feature_size, num_patches, embed_dim, dropout=0.25):
        super().__init__()
        patch_size = math.ceil(feature_size / num_patches)
        self.pad_size = num_patches * patch_size - feature_size
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        x = F.pad(x, (0, self.pad_size)).view(x.shape[0], self.num_patches, self.patch_size)
        return self.projection(x)


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def sample_observed_denoising_targets(observed, mask_probability):
    """Split naturally observed modalities into conditioning and denoising targets.

    A denoising target is always a modality that was genuinely observed before
    this function was called.  Samples with fewer than two observed modalities
    are left unchanged because they cannot provide both a target and context.
    For every other sample and every positive probability, the sampled mask is
    repaired so at least one observed modality is held out and at least one
    remains available for conditioning.
    """
    if observed.ndim != 2:
        raise ValueError("observed must have shape [batch, modalities]")
    if not 0.0 <= mask_probability <= 1.0:
        raise ValueError("mask_probability must be in [0, 1]")

    original_observed = observed.bool()
    target_mask = torch.zeros_like(original_observed)
    if mask_probability == 0.0:
        return original_observed.clone(), target_mask

    observed_counts = original_observed.sum(dim=1)
    eligible = observed_counts >= 2
    target_mask = (
        torch.rand_like(original_observed, dtype=torch.float32) < mask_probability
    ) & original_observed & eligible.unsqueeze(1)

    needs_target = eligible & ~target_mask.any(dim=1)
    if needs_target.any():
        rows = needs_target.nonzero(as_tuple=False).squeeze(1)
        scores = torch.rand(
            rows.numel(),
            original_observed.shape[1],
            device=original_observed.device,
        )
        scores = scores.masked_fill(~original_observed.index_select(0, rows), -1.0)
        chosen = scores.argmax(dim=1)
        target_mask[rows, chosen] = True

    needs_condition = eligible & (target_mask.sum(dim=1) == observed_counts)
    if needs_condition.any():
        rows = needs_condition.nonzero(as_tuple=False).squeeze(1)
        scores = torch.rand(
            rows.numel(),
            original_observed.shape[1],
            device=original_observed.device,
        )
        scores = scores.masked_fill(~target_mask.index_select(0, rows), -1.0)
        restored = scores.argmax(dim=1)
        target_mask[rows, restored] = False

    conditioning_mask = original_observed & ~target_mask
    return conditioning_mask, target_mask


class LatentVAE(nn.Module):
    """Token-space latent VAE mirroring ACADiff's latent compression stage."""
    def __init__(self, hidden_dim, latent_dim, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h).clamp(-8, 8)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def decode(self, z):
        return self.decoder(z)


class AdaptiveFusion(nn.Module):
    """Adaptive 1-to-1 / multi-to-1 conditioning from available modality latents."""
    def __init__(self, latent_dim, num_heads=4, dropout=0.3):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        self.attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, latent_dim), nn.SiLU())

    def forward(self, z, observed):
        q = self.query.expand(z.shape[0], -1, -1)
        key_padding_mask = ~observed
        all_missing = key_padding_mask.all(dim=1)
        if all_missing.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_missing] = False
        fused, _ = self.attn(q, z, z, key_padding_mask=key_padding_mask, need_weights=False)
        return self.proj(fused.squeeze(1))


class DenoiseBlock(nn.Module):
    """DDPM-style denoiser with timestep FiLM and adaptive conditioning."""
    def __init__(self, latent_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, latent_dim * 2))
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, noisy, cond, t_emb):
        gamma, beta = self.time(t_emb).chunk(2, dim=-1)
        h = noisy * (1 + gamma) + beta
        return self.net(torch.cat([h, cond], dim=-1))


class LatentDiffusion(nn.Module):
    """DDPM in token-latent space for ACADiff-style missing modality imputation."""
    def __init__(self, latent_dim, hidden_dim, timesteps=50, beta_start=1e-4, beta_end=2e-2, dropout=0.3):
        super().__init__()
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.denoiser = DenoiseBlock(latent_dim, hidden_dim, dropout)

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1e-20))

    def _extract(self, values, timesteps, x):
        gathered = values.gather(0, timesteps.reshape(-1)).reshape(timesteps.shape)
        return gathered.unsqueeze(-1).expand_as(x)

    def q_sample(self, x_start, timesteps, noise):
        return (
            self._extract(self.sqrt_alphas_cumprod, timesteps, x_start) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start) * noise
        )

    def predict_noise(self, x_t, cond, timesteps):
        t_emb = timestep_embedding(timesteps.reshape(-1), self.latent_dim).reshape(*timesteps.shape, self.latent_dim)
        return self.denoiser(x_t, cond, t_emb)

    def training_loss(self, x_start, cond, target_mask):
        """Denoise only explicitly held-out, naturally observed targets."""
        if target_mask.shape != x_start.shape[:2]:
            raise ValueError("target_mask must match the first two x_start dimensions")
        timesteps = torch.randint(0, self.timesteps, x_start.shape[:2], device=x_start.device)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, timesteps, noise)
        pred_noise = self.predict_noise(x_t, cond, timesteps)
        targets = target_mask.to(dtype=pred_noise.dtype)
        loss = ((pred_noise - noise).pow(2).mean(dim=-1) * targets).sum()
        return loss / targets.sum().clamp_min(1)

    @torch.no_grad()
    def sample_missing(self, observed_latents, cond, observed):
        x_t = torch.randn_like(observed_latents)
        x_t = torch.where(observed.unsqueeze(-1), observed_latents, x_t)

        for step in reversed(range(self.timesteps)):
            timesteps = torch.full(observed.shape, step, dtype=torch.long, device=observed_latents.device)
            pred_noise = self.predict_noise(x_t, cond, timesteps)
            beta_t = self._extract(self.betas, timesteps, x_t)
            sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_t)
            sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas, timesteps, x_t)
            mean = sqrt_recip_alpha * (x_t - beta_t * pred_noise / sqrt_one_minus.clamp_min(1e-8))

            if step > 0:
                var = self._extract(self.posterior_variance, timesteps, x_t)
                x_t = mean + torch.sqrt(var) * torch.randn_like(x_t)
            else:
                x_t = mean
            x_t = torch.where(observed.unsqueeze(-1), observed_latents, x_t)

        return x_t

    def ddim_impute(self, observed_latents, cond, observed, deterministic=False):
        # A fixed zero latent makes repeated validation/test passes comparable.
        # Training and legacy callers retain stochastic DDIM initialization.
        x_t = torch.zeros_like(observed_latents) if deterministic else torch.randn_like(observed_latents)
        x_t = torch.where(observed.unsqueeze(-1), observed_latents, x_t)

        for step in reversed(range(self.timesteps)):
            timesteps = torch.full(observed.shape, step, dtype=torch.long, device=observed_latents.device)
            pred_noise = self.predict_noise(x_t, cond, timesteps)
            sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, x_t)
            sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_t)
            x0 = (x_t - sqrt_one_minus * pred_noise) / sqrt_alpha.clamp_min(1e-8)
            if step > 0:
                prev = torch.full_like(timesteps, step - 1)
                x_t = (
                    self._extract(self.sqrt_alphas_cumprod, prev, x_t) * x0
                    + self._extract(self.sqrt_one_minus_alphas_cumprod, prev, x_t) * pred_noise
                )
            else:
                x_t = x0
            x_t = torch.where(observed.unsqueeze(-1), observed_latents, x_t)
        return x_t


class ACADiffModel(nn.Module):
    """
    ADNI adaptation of ACADiff:
    latent VAE compression, adaptive fusion of available modalities, and a DDPM-style
    denoiser used to impute missing modality latents before classification.
    """
    def __init__(self, num_modalities, hidden_dim, output_dim, num_heads=4, num_layers=1,
                 latent_dim=64, diffusion_steps=50, dropout=0.3, deterministic_eval=False,
                 artificial_mask_probability=0.25):
        super().__init__()
        if not 0.0 <= artificial_mask_probability <= 1.0:
            raise ValueError("artificial_mask_probability must be in [0, 1]")
        self.num_modalities = num_modalities
        self.latent_dim = latent_dim
        self.deterministic_eval = deterministic_eval
        self.artificial_mask_probability = float(artificial_mask_probability)
        self.modality_embed = nn.Parameter(torch.randn(1, num_modalities, hidden_dim) * 0.02)
        self.latent_modality_embed = nn.Parameter(torch.randn(1, num_modalities, latent_dim) * 0.02)
        self.vae = LatentVAE(hidden_dim, latent_dim, dropout)
        self.fusion = AdaptiveFusion(latent_dim, num_heads, dropout)
        self.diffusion = LatentDiffusion(latent_dim, hidden_dim, timesteps=diffusion_steps, dropout=dropout)
        self.post_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, *modality_tokens, observed_mask, return_aux=False):
        x = torch.stack([tokens.mean(dim=1) for tokens in modality_tokens], dim=1) + self.modality_embed
        original_observed = observed_mask.bool()
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar) + self.latent_modality_embed

        denoise_loss = z.new_zeros(())
        conditioning_mask = original_observed
        target_mask = torch.zeros_like(original_observed)
        if self.training and return_aux:
            conditioning_mask, target_mask = sample_observed_denoising_targets(
                original_observed, self.artificial_mask_probability
            )
            denoise_cond = self.fusion(z, conditioning_mask)
            denoise_cond_tokens = denoise_cond.unsqueeze(1).expand_as(z)
            # ``z`` is captured before synthetic masking.  The explicit target
            # mask is a subset of natural observations, so zero placeholders
            # for genuinely missing modalities can never become supervision.
            denoise_loss = self.diffusion.training_loss(
                z.detach(), denoise_cond_tokens, target_mask
            )

        # Synthetic masking is an auxiliary denoising protocol only.  Preserve
        # the historical classification/imputation path and its evaluation
        # behavior by conditioning it on the natural observed mask.
        cond = self.fusion(z, original_observed)
        cond_tokens = cond.unsqueeze(1).expand_as(z)
        imputed = self.diffusion.ddim_impute(
            z, cond_tokens, original_observed,
            deterministic=self.deterministic_eval and not self.training,
        )
        decoded = self.vae.decode(imputed)
        attn_out, _ = self.post_attn(self.norm(decoded), self.norm(decoded), self.norm(decoded), need_weights=False)
        decoded = decoded + attn_out
        weights = torch.softmax(decoded.mean(dim=-1), dim=1)
        fused = (decoded * weights.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(fused)

        if not return_aux:
            return {"logits": logits}
        observed_weights = original_observed.to(dtype=z.dtype)
        observed_count = observed_weights.sum().clamp_min(1)
        recon_per_modality = (self.vae.decode(z) - x.detach()).pow(2).mean(dim=-1)
        recon = (recon_per_modality * observed_weights).sum() / observed_count
        kl_per_modality = -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        ).mean(dim=-1)
        kl = (kl_per_modality * observed_weights).sum() / observed_count
        aux_loss = denoise_loss + 0.1 * recon + 0.001 * kl
        return {
            "logits": logits,
            "aux_loss": aux_loss,
            "denoise_supervision_mask": target_mask,
            "denoise_conditioning_mask": conditioning_mask,
        }
