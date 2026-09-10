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


class AnatomicalGraphLayer(nn.Module):
    """Dense ADNI modality graph analogue of AGMD's anatomical ROI graph."""
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, observed):
        src = x.unsqueeze(2).expand(-1, -1, x.shape[1], -1)
        dst = x.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
        logits = self.edge_mlp(torch.cat([src, dst], dim=-1)).squeeze(-1)
        mask = observed.unsqueeze(1) & observed.unsqueeze(2)
        logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=-1)
        msg = torch.matmul(weights, x)
        return self.update(torch.cat([x, msg], dim=-1)), weights


class GlobalTransformer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, num_layers=2, dropout=0.3):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, num_layers))

    def forward(self, x, observed):
        key_padding_mask = ~observed
        all_missing = key_padding_mask.all(dim=1)
        if all_missing.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_missing] = False
        return self.encoder(x, src_key_padding_mask=key_padding_mask)


class AGMDModel(nn.Module):
    """
    ADNI adaptation of AGMD:
    local branch (per-modality atrophy/detail), regional branch (anatomical graph),
    global semantic branch (transformer), and entropy-weighted multilevel distillation.
    """
    def __init__(self, num_modalities, hidden_dim, output_dim, num_heads=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.num_modalities = num_modalities
        self.modality_embed = nn.Parameter(torch.randn(1, num_modalities, hidden_dim) * 0.02)
        self.missing_embed = nn.Parameter(torch.randn(1, num_modalities, hidden_dim) * 0.02)
        self.local_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim))
            for _ in range(num_modalities)
        ])
        self.graph = AnatomicalGraphLayer(hidden_dim, dropout)
        self.regional_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim))
        self.global_encoder = GlobalTransformer(hidden_dim, num_heads, num_layers, dropout)
        self.global_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim))
        self.uncertainty_gate = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.final_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def _pool_observed(self, x, observed):
        w = observed.float()
        return (x * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=1, keepdim=True).clamp_min(1)

    def forward(self, *modality_tokens, observed_mask, return_aux=False):
        x = torch.stack([tokens.mean(dim=1) for tokens in modality_tokens], dim=1)
        observed = observed_mask.bool()
        x = torch.where(observed.unsqueeze(-1), x, self.missing_embed.expand(x.shape[0], -1, -1))
        x = x + self.modality_embed

        local_logits = torch.stack([head(x[:, i]) for i, head in enumerate(self.local_heads)], dim=1)
        graph_x, graph_weights = self.graph(x, observed)
        regional_feat = self._pool_observed(graph_x, observed)
        regional_logits = self.regional_head(regional_feat)

        global_x = self.global_encoder(graph_x, observed)
        global_feat = self._pool_observed(global_x, observed)
        global_logits = self.global_head(global_feat)

        gate = self.uncertainty_gate(torch.cat([regional_logits, global_logits], dim=-1))
        fused_feat = torch.cat([gate * regional_feat, (1.0 - gate) * global_feat], dim=-1)
        logits = self.final_head(fused_feat)

        if not return_aux:
            return {"logits": logits}

        teacher_prob = torch.softmax(global_logits.detach(), dim=-1)
        local_log_prob = F.log_softmax(local_logits, dim=-1)
        local_kd = -(teacher_prob.unsqueeze(1) * local_log_prob).sum(dim=-1)
        local_kd = (local_kd * observed.float()).sum() / observed.float().sum().clamp_min(1)
        regional_kd = F.kl_div(
            F.log_softmax(regional_logits, dim=-1),
            teacher_prob,
            reduction='batchmean',
        )
        entropy = -(teacher_prob * teacher_prob.clamp_min(1e-8).log()).sum(dim=-1).mean()
        graph_reg = (graph_weights * graph_weights.clamp_min(1e-8).log()).sum(dim=-1).abs().mean()
        aux_loss = local_kd + regional_kd + 0.1 * entropy + 0.05 * graph_reg
        return {"logits": logits, "aux_loss": aux_loss}
