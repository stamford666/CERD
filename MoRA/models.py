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


class LoRALinear(nn.Module):
    """Modality-aware low-rank adaptation, following MoRA's LoRA-guided adaptation idea."""
    def __init__(self, hidden_dim, rank=16, dropout=0.3):
        super().__init__()
        self.base = nn.Linear(hidden_dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = rank ** -0.5
        nn.init.normal_(self.up.weight, std=1e-4)

    def forward(self, x, gate):
        return self.base(x) + gate.unsqueeze(-1) * self.up(self.dropout(self.down(x))) * self.scale


class MissingAwarePromptBank(nn.Module):
    """MoRA-style prompt selection: complete prompt plus missing-modality prompts."""
    def __init__(self, num_modalities, prompt_length, hidden_dim):
        super().__init__()
        self.prompt_length = prompt_length
        self.complete_prompt = nn.Parameter(torch.randn(prompt_length, hidden_dim) * 0.02)
        self.missing_prompts = nn.Parameter(torch.randn(num_modalities, prompt_length, hidden_dim) * 0.02)
        self.prompt_mixer = nn.Sequential(
            nn.Linear(num_modalities, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, prompt_length),
        )

    def forward(self, observed_mask):
        missing = (~observed_mask.bool()).float()
        prompt = self.complete_prompt.unsqueeze(0).expand(observed_mask.shape[0], -1, -1)
        prompt = prompt + torch.einsum('bm,mph->bph', missing, self.missing_prompts)
        gate = torch.sigmoid(self.prompt_mixer(missing)).unsqueeze(-1)
        return prompt * (1.0 + gate)


class MoRABlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, rank=16, dropout=0.3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q = LoRALinear(hidden_dim, rank, dropout)
        self.k = LoRALinear(hidden_dim, rank, dropout)
        self.v = LoRALinear(hidden_dim, rank, dropout)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, observed_mask, lora_gate):
        z = self.norm1(x)
        q = self.q(z, lora_gate).view(z.shape[0], z.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(z, lora_gate).view(z.shape[0], z.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(z, lora_gate).view(z.shape[0], z.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(~observed_mask[:, None, None, :], -1e4)
        attn = torch.softmax(attn, dim=-1)
        msg = torch.matmul(attn, v).transpose(1, 2).reshape_as(z)
        x = x + self.dropout(self.out(msg))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class MoRAModel(nn.Module):
    """
    ADNI adaptation of MoRA:
    modality-aware low-rank adapters are gated by the available-modality pattern, and
    missing modality tokens are recovered from the observed context before fusion.
    """
    def __init__(self, num_modalities, hidden_dim, output_dim, num_heads=4, num_layers=2,
                 adapter_rank=16, prompt_length=4, dropout=0.3):
        super().__init__()
        self.num_modalities = num_modalities
        self.prompt_length = prompt_length
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.modality_embed = nn.Parameter(torch.randn(1, num_modalities, hidden_dim) * 0.02)
        self.missing_embed = nn.Parameter(torch.randn(1, num_modalities, hidden_dim) * 0.02)
        self.prompt_bank = MissingAwarePromptBank(num_modalities, prompt_length, hidden_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(num_modalities, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_modalities),
        )
        self.recover = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_modalities)
        ])
        self.blocks = nn.ModuleList([
            MoRABlock(hidden_dim, num_heads, adapter_rank, dropout) for _ in range(max(1, num_layers))
        ])
        self.pooler = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def _recover_modalities(self, x, observed, context, prompts):
        prompt_context = prompts.mean(dim=1)
        recovered = []
        for m, layer in enumerate(self.recover):
            rec_in = torch.cat([context, prompt_context + self.modality_embed[:, m, :]], dim=-1)
            recovered.append(layer(rec_in))
        recovered = torch.stack(recovered, dim=1) + self.modality_embed
        return torch.where(observed.unsqueeze(-1), x, recovered), recovered

    def forward(self, *modality_tokens, observed_mask, return_aux=False):
        x = torch.stack([tokens.mean(dim=1) for tokens in modality_tokens], dim=1)
        observed = observed_mask.bool()
        x = torch.where(observed.unsqueeze(-1), x, self.missing_embed.expand(x.shape[0], -1, -1))
        x = x + self.modality_embed

        obs_float = observed.float()
        context = (x * obs_float.unsqueeze(-1)).sum(dim=1) / obs_float.sum(dim=1, keepdim=True).clamp_min(1)
        prompts = self.prompt_bank(observed)
        x, recovered = self._recover_modalities(x, observed, context, prompts)

        modality_gate = torch.sigmoid(self.gate_net(obs_float))
        missing_strength = (~observed).float().mean(dim=1, keepdim=True)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        seq = torch.cat([cls, prompts, x], dim=1)
        prompt_gate = missing_strength.expand(-1, self.prompt_length + 1)
        lora_gate = torch.cat([prompt_gate, modality_gate], dim=1)
        full_mask = torch.ones(seq.shape[:2], dtype=torch.bool, device=seq.device)
        for block in self.blocks:
            seq = block(seq, full_mask, lora_gate)

        cls_out = seq[:, 0]
        mod_out = seq[:, 1 + self.prompt_length:]
        scores = self.pooler(mod_out).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        fused = (mod_out * weights.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(fused + cls_out)

        if not return_aux:
            return {"logits": logits}
        # Self-drop observed modalities and reconstruct them from the remaining context. This gives
        # the missing-aware prompts a supervised signal even when the raw sample is nearly complete.
        if observed.any():
            drop_scores = torch.rand_like(obs_float).masked_fill(~observed, -1.0)
            drop_idx = drop_scores.argmax(dim=1)
            drop_mask = F.one_hot(drop_idx, num_classes=self.num_modalities).bool() & observed
            kept = observed & ~drop_mask
            empty = ~kept.any(dim=1)
            kept[empty] = observed[empty]
            drop_mask[empty] = False
            kept_float = kept.float()
            sim_context = (x.detach() * kept_float.unsqueeze(-1)).sum(dim=1) / kept_float.sum(dim=1, keepdim=True).clamp_min(1)
            sim_prompts = self.prompt_bank(kept)
            _, sim_recovered = self._recover_modalities(x.detach(), kept, sim_context, sim_prompts)
            rec_loss = (
                (1.0 - F.cosine_similarity(sim_recovered, x.detach(), dim=-1))
                + 0.05 * (F.normalize(sim_recovered, dim=-1) - F.normalize(x.detach(), dim=-1)).pow(2).mean(dim=-1)
            )
            rec_loss = (rec_loss * drop_mask.float()).sum()
            rec_loss = rec_loss / drop_mask.float().sum().clamp_min(1)
        else:
            rec_loss = torch.zeros((), device=x.device)
        prompt_div = F.cosine_similarity(
            self.prompt_bank.missing_prompts.unsqueeze(0),
            self.prompt_bank.missing_prompts.unsqueeze(1),
            dim=-1,
        ).abs().mean()
        return {"logits": logits, "aux_loss": rec_loss + 0.01 * prompt_div}
