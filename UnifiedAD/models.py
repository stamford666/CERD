import math
from typing import List

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


class SharedProjectionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor):
        z = self.cross_norm(q_tokens)
        msg, _ = self.cross_attn(z, kv_tokens, kv_tokens, need_weights=False)
        z = q_tokens + self.dropout(msg)
        msg, _ = self.self_attn(self.self_norm(z), self.self_norm(z), self.self_norm(z), need_weights=False)
        z = z + self.dropout(msg)
        z = z + self.dropout(self.ffn(self.ffn_norm(z)))
        return z


class UnifiedAnyModAD(nn.Module):
    """
    Lightweight reproduction of MICCAI 2024 AnyMod / Unified Multi-Modal AD.

    Core pieces:
      - modality-specific query tokens with shared projection Transformer
      - task-anchor clustering to a fixed number of task factors
      - fusion Transformer and class-anchor task alignment
    """
    def __init__(
        self,
        num_modalities: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 4,
        num_query_tokens: int = 8,
        num_task_tokens: int = 8,
        num_proj_layers: int = 1,
        num_fusion_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_query_tokens = num_query_tokens
        self.num_task_tokens = num_task_tokens

        self.modality_queries = nn.Parameter(torch.randn(num_modalities, num_query_tokens, hidden_dim) * 0.02)
        self.modality_embed = nn.Parameter(torch.zeros(1, num_modalities, 1, hidden_dim))
        self.proj_blocks = nn.ModuleList([
            SharedProjectionBlock(hidden_dim, num_heads, dropout)
            for _ in range(max(1, num_proj_layers))
        ])

        self.task_anchors = nn.Parameter(torch.randn(num_task_tokens, hidden_dim) * 0.02)
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(fusion_layer, num_layers=max(1, num_fusion_layers))
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        self.class_anchors = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.02)

    def _project_modalities(self, modality_tokens: List[torch.Tensor], observed_mask: torch.Tensor):
        B = modality_tokens[0].shape[0]
        projected = []
        for m, tokens in enumerate(modality_tokens):
            q = self.modality_queries[m].unsqueeze(0).expand(B, -1, -1)
            kv = tokens + self.modality_embed[:, m, :, :]
            z = q
            for block in self.proj_blocks:
                z = block(z, kv)
            z = z * observed_mask[:, m].float().view(B, 1, 1)
            projected.append(z)
        return torch.cat(projected, dim=1)

    def _cluster_to_task_tokens(self, tokens: torch.Tensor, token_mask: torch.Tensor):
        anchors = F.normalize(self.task_anchors, dim=-1)
        norm_tokens = F.normalize(tokens, dim=-1)
        sim = torch.matmul(norm_tokens, anchors.t())
        sim = sim.masked_fill(~token_mask.unsqueeze(-1), -1e4)

        assign = sim.argmax(dim=-1)
        weights = torch.zeros_like(sim)
        weights.scatter_(-1, assign.unsqueeze(-1), 1.0)
        weights = weights * token_mask.unsqueeze(-1).float()
        soft = torch.softmax(sim, dim=1) * weights
        denom = soft.sum(dim=1).transpose(1, 0).transpose(0, 1).clamp_min(1e-6)
        task_tokens = torch.einsum('btn,btd->bnd', soft, tokens) / denom.unsqueeze(-1)

        empty = soft.sum(dim=1) <= 1e-6
        if empty.any():
            fallback = self.task_anchors.unsqueeze(0).expand(tokens.shape[0], -1, -1)
            task_tokens = torch.where(empty.unsqueeze(-1), fallback, task_tokens)
        return task_tokens, sim

    def forward(self, *modality_tokens, observed_mask: torch.Tensor, return_aux: bool = False):
        observed_mask = observed_mask.bool()
        tokens = self._project_modalities(list(modality_tokens), observed_mask)
        B = tokens.shape[0]
        token_mask = observed_mask.unsqueeze(-1).expand(-1, -1, self.num_query_tokens).reshape(B, -1)
        no_obs = ~token_mask.any(dim=1)
        if no_obs.any():
            token_mask[no_obs] = True

        task_tokens, sim = self._cluster_to_task_tokens(tokens, token_mask)
        fused = self.fusion(task_tokens)
        embedding = fused.mean(dim=1)
        logits = self.classifier(embedding)

        if not return_aux:
            return {"logits": logits}

        align_prob = torch.softmax(sim, dim=-1).max(dim=-1).values.clamp_min(1e-8)
        align_loss = -(torch.log(align_prob) * token_mask.float()).sum() / token_mask.float().sum().clamp_min(1.0)

        norm_embed = F.normalize(embedding, dim=-1)
        norm_class = F.normalize(self.class_anchors, dim=-1)
        anchor_logits = torch.matmul(norm_embed, norm_class.t())
        return {
            "logits": logits,
            "embedding": embedding,
            "align_loss": align_loss,
            "anchor_logits": anchor_logits,
        }
