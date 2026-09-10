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


class ReFlowCoupling(nn.Module):
    """Affine flow module used to move observed latent distributions toward missing modalities."""
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

    def forward(self, context, base):
        shift, log_scale = self.net(torch.cat([context, base], dim=-1)).chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale)
        return base * torch.exp(log_scale) + shift, log_scale


class MaskedAdaptiveGraphAttention(nn.Module):
    """MAT: masked graph attention over observed and ReFlow-recovered modalities."""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.edge_bias = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, node_mask):
        z = self.norm(x)
        q = self.q(z).view(z.shape[0], z.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(z).view(z.shape[0], z.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(z).view(z.shape[0], z.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        src = z.unsqueeze(2).expand(-1, -1, z.shape[1], -1)
        dst = z.unsqueeze(1).expand(-1, z.shape[1], -1, -1)
        bias = self.edge_bias(torch.cat([src, dst], dim=-1)).permute(0, 3, 1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn + bias
        attn = attn.masked_fill(~node_mask[:, None, None, :], -1e4)
        attn = torch.softmax(attn, dim=-1)
        msg = torch.matmul(attn, v).transpose(1, 2).reshape_as(z)
        return x + self.dropout(self.out(msg)), attn


class AGDiCModel(nn.Module):
    """
    ADNI adaptation of AGDiC:
    ReFlow latent recovery, masked adaptive graph transformer, and anatomy/modality
    regularization for incomplete multimodal diagnosis.
    """
    def __init__(self, num_modalities, hidden_dim, output_dim, num_heads=4, num_layers=2,
                 dropout=0.3):
        super().__init__()
        self.num_modalities = num_modalities
        self.modality_embed = nn.Parameter(torch.randn(1, num_modalities, hidden_dim) * 0.02)
        self.base_missing = nn.Parameter(torch.randn(1, num_modalities, hidden_dim) * 0.02)
        self.flows = nn.ModuleList([ReFlowCoupling(hidden_dim, dropout) for _ in range(num_modalities)])
        self.graph_layers = nn.ModuleList([
            MaskedAdaptiveGraphAttention(hidden_dim, num_heads, dropout) for _ in range(max(1, num_layers))
        ])
        self.shared_proj = nn.Linear(hidden_dim, hidden_dim)
        self.private_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_modalities)])
        self.pooler = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def _context(self, x, observed):
        w = observed.float()
        return (x * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=1, keepdim=True).clamp_min(1)

    def forward(self, *modality_tokens, observed_mask, return_aux=False):
        x = torch.stack([tokens.mean(dim=1) for tokens in modality_tokens], dim=1)
        observed = observed_mask.bool()
        context = self._context(torch.where(observed.unsqueeze(-1), x, 0.0), observed)

        recovered, log_scales = [], []
        for m, flow in enumerate(self.flows):
            base = self.base_missing[:, m].expand(x.shape[0], -1)
            z_m, log_scale = flow(context, base)
            recovered.append(z_m)
            log_scales.append(log_scale)
        recovered = torch.stack(recovered, dim=1)
        log_scales = torch.stack(log_scales, dim=1)

        z = torch.where(observed.unsqueeze(-1), x, recovered) + self.modality_embed
        shared = F.normalize(self.shared_proj(z), dim=-1)
        private = torch.stack([proj(z[:, m]) for m, proj in enumerate(self.private_proj)], dim=1)
        graph_x = shared + private
        full_mask = torch.ones_like(observed, dtype=torch.bool)
        attn_maps = []
        for layer in self.graph_layers:
            graph_x, attn = layer(graph_x, full_mask)
            attn_maps.append(attn)

        scores = self.pooler(graph_x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        fused = (graph_x * weights.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(fused)

        if not return_aux:
            return {"logits": logits}

        flow_loss = ((recovered - context.detach().unsqueeze(1)).pow(2).mean(dim=-1) * (~observed).float()).sum()
        flow_loss = flow_loss / (~observed).float().sum().clamp_min(1)
        log_jacobian = log_scales.abs().mean()
        orth = (F.normalize(private, dim=-1) * shared).sum(dim=-1).abs().mean()
        graph_entropy = 0.0
        for attn in attn_maps:
            graph_entropy = graph_entropy + (-(attn * attn.clamp_min(1e-8).log()).sum(dim=-1).mean())
        aux_loss = flow_loss + 0.01 * log_jacobian + 0.1 * orth - 0.01 * graph_entropy
        return {"logits": logits, "aux_loss": aux_loss}
