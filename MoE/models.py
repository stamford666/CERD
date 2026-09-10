import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from moe_module import *
from itertools import combinations
from typing import Optional, List, NamedTuple, Tuple


# =========================
# Basic blocks (keep style)
# =========================
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


class ConcatTransformerBaseline(nn.Module):
    """Plain early-fusion Transformer over concatenated modality tokens.

    Missing modalities are excluded with the Transformer's key-padding mask.
    The model intentionally contains no MoE routing, completion module,
    provenance embedding, branch bank, or reliability-aware fusion.
    """

    def __init__(
        self,
        *,
        num_modalities: int,
        num_patches: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        num_layers_pred: int,
    ) -> None:
        super().__init__()
        if num_modalities <= 0 or num_patches <= 0 or num_layers <= 0:
            raise ValueError("Transformer dimensions and layer count must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_modalities = int(num_modalities)
        self.num_patches = int(num_patches)
        sequence_length = self.num_modalities * self.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length + 1, hidden_dim)
        )
        self.modality_embedding = nn.Parameter(
            torch.zeros(1, self.num_modalities, 1, hidden_dim)
        )
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.modality_embedding, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=2 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = MLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers_pred,
            activation=nn.GELU(),
            dropout=dropout,
        )

    @staticmethod
    def provenance() -> dict:
        return {
            "implementation": "plain concatenation Transformer baseline",
            "fusion": "concatenate all modality tokens before self-attention",
            "missingness": "key-padding mask excludes every token of a missing modality",
            "pooling": "learned CLS token",
            "sparse_moe": False,
            "completion": False,
            "provenance_embedding": False,
            "reliability_aware_branch_fusion": False,
        }

    def forward(
        self,
        *tokens: torch.Tensor,
        observed_mask: torch.Tensor,
        return_aux: bool = False,
    ) -> dict[str, torch.Tensor]:
        if len(tokens) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modality tensors, got {len(tokens)}"
            )
        if observed_mask.ndim != 2 or observed_mask.shape[1] != self.num_modalities:
            raise ValueError(
                "observed_mask must have shape (batch, num_modalities); got "
                f"{tuple(observed_mask.shape)}"
            )
        if any(
            token.ndim != 3 or token.shape[1] != self.num_patches
            for token in tokens
        ):
            raise ValueError(
                "Each modality must have shape (batch, num_patches, hidden_dim)"
            )
        observed_mask = observed_mask.bool()
        if (~observed_mask).all(dim=1).any():
            raise ValueError("At least one modality must be observed per sample")

        stacked = torch.stack(tokens, dim=1)
        stacked = stacked + self.modality_embedding
        batch_size = stacked.shape[0]
        fused = stacked.reshape(batch_size, -1, stacked.shape[-1])
        cls = self.cls_token.expand(batch_size, -1, -1)
        fused = torch.cat((cls, fused), dim=1) + self.position_embedding

        missing_tokens = (~observed_mask).unsqueeze(-1).expand(
            -1, -1, self.num_patches
        )
        padding_mask = torch.cat(
            (
                torch.zeros(
                    batch_size, 1, dtype=torch.bool, device=observed_mask.device
                ),
                missing_tokens.reshape(batch_size, -1),
            ),
            dim=1,
        )
        encoded = self.transformer(fused, src_key_padding_mask=padding_mask)
        logits = self.classifier(self.norm(encoded[:, 0]))
        output = {"logits": logits}
        if return_aux:
            output["aux_loss"] = logits.new_zeros(())
        return output


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, kv, attn_mask=None):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        kv = self.kv(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Custom3DCNN(nn.Module):
    # Architecture provided by: End-To-End Alzheimer's Disease Diagnosis and Biomarker Identification
    def __init__(self, hidden_dim=128):
        super(Custom3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        self.dropout1 = nn.Dropout3d(0.2)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=3)
        self.dropout2 = nn.Dropout3d(0.2)

        self.conv5 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv6 = nn.Conv3d(128, hidden_dim, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(4, 4, 4))
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

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.
    """
    def __init__(self, feature_size, num_patches, embed_dim, dropout=0.25):
        super().__init__()
        patch_size = math.ceil(feature_size / num_patches)
        pad_size = num_patches * patch_size - feature_size
        self.pad_size = pad_size
        self.num_patches = num_patches
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        x = F.pad(x, (0, self.pad_size)).view(x.shape[0], self.num_patches, self.patch_size)
        x = self.projection(x)
        return x


class IndependentPatchEmbeddings(nn.Module):
    """ABCD tabular tokenizer with a separate projection per feature block."""

    def __init__(self, feature_size, num_patches, embed_dim, dropout=0.25):
        super().__init__()
        patch_size = math.ceil(feature_size / num_patches)
        self.pad_size = num_patches * patch_size - feature_size
        self.num_patches = num_patches
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.projections = nn.ModuleList([
            nn.Linear(patch_size, embed_dim) for _ in range(num_patches)
        ])

    def forward(self, x):
        patches = F.pad(x, (0, self.pad_size)).view(
            x.shape[0], self.num_patches, self.patch_size
        )
        return torch.stack([
            projection(patches[:, patch_idx, :])
            for patch_idx, projection in enumerate(self.projections)
        ], dim=1)


class LowRankAdaptivePatchEmbeddings(nn.Module):
    """Shared patch projection with parameter-efficient patch-specific adapters.

    The historical tokenizer applies one linear map to every contiguous feature
    block.  That is economical, but it assumes that coordinate ``j`` has the
    same meaning in every block.  Tabular and genomic feature blocks do not
    satisfy that assumption.  This module keeps the shared projection and adds
    a low-rank residual ``x_p A_p B_p`` for each patch ``p``.

    ``B`` is initialized to exact zero, so the initial forward pass is exactly
    the historical shared projection.  The additional parameter count is
    ``num_patches * patch_size * rank + num_patches * rank * embed_dim``
    instead of the much larger fully independent
    ``num_patches * patch_size * embed_dim`` tokenizer.
    """

    def __init__(
        self,
        feature_size: int,
        num_patches: int,
        embed_dim: int,
        dropout: float = 0.25,
        adapter_rank: int = 4,
    ):
        super().__init__()
        del dropout  # Kept for drop-in compatibility with the other tokenizers.
        if adapter_rank <= 0:
            raise ValueError("adapter_rank must be positive")
        patch_size = math.ceil(feature_size / num_patches)
        self.pad_size = num_patches * patch_size - feature_size
        self.num_patches = int(num_patches)
        self.feature_size = int(feature_size)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.adapter_rank = int(adapter_rank)

        self.projection = nn.Linear(self.patch_size, self.embed_dim)
        self.adapter_down = nn.Parameter(
            torch.empty(
                self.num_patches,
                self.patch_size,
                self.adapter_rank,
            )
        )
        self.adapter_up = nn.Parameter(
            torch.zeros(
                self.num_patches,
                self.adapter_rank,
                self.embed_dim,
            )
        )
        # ``kaiming_uniform_`` interprets a 3-D tensor as a convolutional
        # kernel and would incorrectly fold ``adapter_rank`` into fan-in.
        # Initialize every [patch_size, rank] matrix with the exact fan-in
        # bound used by ``nn.Linear(patch_size, rank)`` instead.
        adapter_bound = 1.0 / math.sqrt(float(self.patch_size))
        nn.init.uniform_(self.adapter_down, -adapter_bound, adapter_bound)
        self.adapter_scale = 1.0 / math.sqrt(float(self.adapter_rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = F.pad(x, (0, self.pad_size)).view(
            x.shape[0], self.num_patches, self.patch_size
        )
        shared = self.projection(patches)
        adapted = torch.einsum(
            "bpf,pfr,pre->bpe",
            patches,
            self.adapter_down,
            self.adapter_up,
        )
        return shared + self.adapter_scale * adapted



class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        num_experts,
        num_routers,
        d_model,
        num_head,
        dropout=0.1,
        activation=nn.GELU,
        hidden_times=2,
        mlp_sparse=False,
        self_attn=True,
        full_modality_index=4,
        top_k=2,
        standard_residual=False,
        gated_residual=False,
        normalized_gate_loss=False,
        dense_backbone=False,
        **kwargs
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation()
        self.attn = Attention(d_model, num_heads=num_head, qkv_bias=False, attn_drop=dropout, proj_drop=dropout)

        self.mlp_sparse = mlp_sparse
        self.dense_backbone = bool(dense_backbone)
        self.self_attn = self_attn
        self.standard_residual = bool(standard_residual)
        self.gated_residual = bool(gated_residual)
        if self.standard_residual and self.gated_residual:
            raise ValueError("standard_residual and gated_residual are mutually exclusive")
        if self.gated_residual:
            # Per-channel, zero-initialized recovery of the input residual.  The
            # candidate starts exactly on the historical path and can learn only
            # the residual dimensions supported by the task loss.
            self.residual_gate = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_parameter("residual_gate", None)
        self.expert_index = None
        self.full_modality_index = full_modality_index

        if self.mlp_sparse:
            self.mlp = FMoETransformerMLP(
                num_expert=num_experts,
                n_router=num_routers,
                d_model=d_model,
                d_hidden=d_model * hidden_times,
                activation=nn.GELU(),
                top_k=top_k,
                normalized_gate_loss=normalized_gate_loss,
                **kwargs
            )
            if self.dense_backbone:
                cpu_rng = torch.get_rng_state()
                self.dense_mlp = MLP(
                    input_dim=d_model, hidden_dim=d_model * hidden_times,
                    output_dim=d_model, num_layers=2, activation=nn.GELU(),
                    dropout=dropout,
                )
                torch.set_rng_state(cpu_rng)
        else:
            self.mlp = MLP(input_dim=d_model, hidden_dim=d_model * hidden_times, output_dim=d_model,
                           num_layers=2, activation=nn.GELU(), dropout=dropout)

    def forward(self, x, attn_mask=None):
        if self.self_attn:
            chunk_size = [item.shape[1] for item in x]
            residual = torch.cat(x, dim=1)
            normalized = self.norm1(residual)
            attended = self.attn(normalized, normalized, attn_mask)
            if self.standard_residual:
                x = residual + self.dropout1(attended)
            else:
                # Historical behavior retained for old ADNI configurations.
                x = attended + self.dropout1(attended)
                if self.residual_gate is not None:
                    x = x + torch.tanh(self.residual_gate).view(1, 1, -1) * residual
            x = torch.split(x, chunk_size, dim=1)
            x = [item for item in x]

            if self.mlp_sparse and not self.dense_backbone:
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i]), self.expert_index))
            else:
                mlp = self.dense_mlp if self.mlp_sparse else self.mlp
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout2(mlp(self.norm2(x[i])))

        else:
            chunk_size = [item.shape[1] for item in x]
            x = [item for item in x]
            for i in range(len(chunk_size)):
                other_m = [x[j] for j in range(len(chunk_size)) if j != i]
                other_m = torch.cat([x[i], *other_m], dim=1)
                x[i] = self.attn(x[i], other_m, attn_mask)
            x = [x[i] + self.dropout1(x[i]) for i in range(len(chunk_size))]

            if self.mlp_sparse and not self.dense_backbone:
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i]), self.expert_index))
            else:
                mlp = self.dense_mlp if self.mlp_sparse else self.mlp
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout2(mlp(self.norm2(x[i])))

        return x

    def set_expert_index(self, expert_index):
        self.expert_index = expert_index

    def set_full_modality(self, is_full_modality):
        if hasattr(self.mlp, 'set_full_modality'):
            self.mlp.set_full_modality(is_full_modality)


class FlexMoE(nn.Module):
    def __init__(self, num_modalities, full_modality_index, num_patches, hidden_dim,
                 num_layers, num_experts, num_routers, top_k, num_heads=2, dropout=0.5,
                 standard_residual=False, gated_residual=False,
                 normalized_gate_loss=False, dense_backbone=False):
        super().__init__()
        layers = []
        _sparse = True
        layers.append(TransformerEncoderLayer(num_experts, num_routers, hidden_dim, num_head=num_heads,
                                              dropout=dropout, hidden_times=2, mlp_sparse=_sparse,
                                              full_modality_index=full_modality_index, top_k=top_k,
                                              standard_residual=standard_residual,
                                              gated_residual=gated_residual,
                                              normalized_gate_loss=normalized_gate_loss,
                                              dense_backbone=dense_backbone))
        for _ in range(num_layers - 1):
            _sparse = not _sparse
            layers.append(TransformerEncoderLayer(num_experts, num_routers, hidden_dim, num_head=num_heads,
                                                  dropout=dropout, hidden_times=2, mlp_sparse=_sparse,
                                                  full_modality_index=full_modality_index, top_k=top_k,
                                                  standard_residual=standard_residual,
                                                  gated_residual=gated_residual,
                                                  normalized_gate_loss=normalized_gate_loss,
                                                  dense_backbone=dense_backbone))
        self.layers = nn.Sequential(*layers)

        self.pos_embed = nn.Parameter(torch.zeros(1, np.sum([num_patches] * num_modalities), hidden_dim))
        self.combination_to_index = self._create_combination_index(num_modalities)

    def forward(self, *inputs, expert_indices=None, return_pooled: bool = True):
        chunk_size = [x.shape[1] for x in inputs]
        x = torch.cat(inputs, dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = torch.split(x, chunk_size, dim=1)

        for layer in self.layers:
            if expert_indices is not None and hasattr(layer, 'set_expert_index'):
                layer.set_expert_index(expert_indices)
            x = layer(x)

        if return_pooled:
            pooled = [t.mean(dim=1) for t in x]  # list[(B,D)]
            return pooled
        return x

    def gate_loss(self):
        g_loss = []
        for _, mm in self.named_modules():
            if hasattr(mm, 'all_gates'):
                for i in range(len(mm.all_gates)):
                    i_loss = mm.all_gates[f'{i}'].get_loss()
                    if i_loss is not None:
                        g_loss.append(i_loss)
        return sum(g_loss) if len(g_loss) > 0 else 0.0

    def _create_combination_index(self, num_modalities):
        combinations_list = []
        for r in range(1, num_modalities + 1):
            combinations_list.extend(combinations(range(num_modalities), r))
        return {tuple(sorted(comb)): idx for idx, comb in enumerate(combinations_list)}

    def assign_expert(self, combination):
        return self.combination_to_index.get(tuple(sorted(combination)))

    def set_full_modality(self, is_full_modality):
        for layer in self.layers:
            if hasattr(layer, 'set_full_modality'):
                layer.set_full_modality(is_full_modality)


# =========================
# Generators + flags + recon
# =========================
class ConditionalGenerator(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_patches: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_output_gate: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches

        self.query_embed = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)
        self.pos_q = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "ln_q": nn.LayerNorm(hidden_dim),
                "ln_kv": nn.LayerNorm(hidden_dim),
                "xattn": nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True),
                "drop": nn.Dropout(dropout),
                "ln_ffn": nn.LayerNorm(hidden_dim),
                "ffn": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                ),
            }))

        # Always construct the gate so disabling it does not alter parameter
        # names, RNG consumption, or initialization of later modules.
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.use_output_gate = bool(use_output_gate)

    def forward(self, ctx_tokens: torch.Tensor):
        B = ctx_tokens.shape[0]
        q = self.query_embed.expand(B, -1, -1) + self.pos_q
        kv = ctx_tokens
        for layer in self.layers:
            q_norm = layer["ln_q"](q)
            kv_norm = layer["ln_kv"](kv)
            x, _ = layer["xattn"](q_norm, kv_norm, kv_norm, need_weights=False)
            q = q + layer["drop"](x)
            q = q + layer["ffn"](layer["ln_ffn"](q))
        if not self.use_output_gate:
            return q
        g = self.gate(q)
        return q * g


class ModalityFlagEmbedding(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.emb = nn.Embedding(2, hidden_dim)
        nn.init.zeros_(self.emb.weight)

    def forward(self, x: torch.Tensor, flag: torch.Tensor):
        return x + self.emb(flag.long()).unsqueeze(1)


class ReconProjector(nn.Module):
    def __init__(self, hidden_dim: int, proj_dim: Optional[int] = None):
        super().__init__()
        if proj_dim is None:
            proj_dim = hidden_dim
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class AttentiveTokenPooler(nn.Module):
    """
    Token-level importance pooling for each modality.
    Inspired by the moe_eating_disorder model, but kept modality-local.
    """
    def __init__(
        self,
        hidden_dim: int,
        temperature: float = 0.5,
        dropout: float = 0.1,
        attention_mix_init: float = -4.0,
        mean_pooling_only: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.mean_pooling_only = bool(mean_pooling_only)
        self.attn_mix_logit = nn.Parameter(torch.tensor(float(attention_mix_init)))
        self.score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(tokens).squeeze(-1)
        weights = F.softmax(scores / max(self.temperature, 1e-6), dim=1)
        attn_pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        mean_pooled = tokens.mean(dim=1)
        mix = torch.sigmoid(self.attn_mix_logit)
        pooled = mean_pooled if self.mean_pooling_only else mean_pooled + mix * (attn_pooled - mean_pooled)
        return pooled, weights


class TabMJointHeadOutput(NamedTuple):
    """Vectorized member predictions from :class:`TabMJointEnsembleHead`."""

    log_probabilities: torch.Tensor
    probabilities: torch.Tensor
    member_logits: torch.Tensor
    member_probabilities: torch.Tensor


class TabMJointEnsembleHead(nn.Module):
    """A small TabMmini-style ensemble for the multimodal joint branch.

    Members share every hidden layer.  Diversity comes from a learned
    member-specific multiplicative adapter on the pooled input and a tiny
    member-specific prediction layer.  Classification probabilities are
    averaged across members; logits are never averaged.

    This module is deliberately used only when ``ensemble_size > 1``.  The
    disabled path in :class:`ReliabilityBranchFusion` continues to construct
    the historical :class:`MLP` object, preserving its parameters, RNG draws,
    state-dict keys, and numerical path.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        ensemble_size: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        input_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        output_dim = int(output_dim)
        num_layers = int(num_layers)
        ensemble_size = int(ensemble_size)
        if ensemble_size <= 1:
            raise ValueError("TabM joint-head ensemble_size must be greater than one")
        if num_layers < 2:
            raise ValueError("TabM joint-head ensemble requires num_layers >= 2")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.ensemble_size = ensemble_size

        # Match the historical MLP initialization order for the shared trunk
        # and one prediction head.  Consequently, member zero starts as an
        # exact anchored copy of the old joint MLP when its input adapter is 1.
        layers = []
        activation = nn.ReLU()
        drop = nn.Dropout(dropout)
        layers += [nn.Linear(input_dim, hidden_dim), activation, drop]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation, drop]
        self.shared_trunk = nn.Sequential(*layers)

        legacy_prediction_head = nn.Linear(hidden_dim, output_dim)
        self.member_weight = nn.Parameter(
            legacy_prediction_head.weight.detach()
            .unsqueeze(0)
            .repeat(ensemble_size, 1, 1)
        )
        self.member_bias = nn.Parameter(
            legacy_prediction_head.bias.detach()
            .unsqueeze(0)
            .repeat(ensemble_size, 1)
        )

        # The first member is the legacy anchor.  Learned representations enter
        # this head after nonlinear feature extraction, for which TabM uses a
        # standard-normal first adapter rather than the random-sign variant.
        input_scale = torch.ones(ensemble_size, input_dim)
        with torch.no_grad():
            input_scale[1:].normal_(mean=0.0, std=1.0)
        self.input_scale = nn.Parameter(input_scale)

    def forward_with_members(self, x: torch.Tensor) -> TabMJointHeadOutput:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                "TabM joint head expects a rank-2 tensor with "
                f"feature dimension {self.input_dim}, got {tuple(x.shape)}"
            )
        batch_size = x.shape[0]
        adapted = x.unsqueeze(1) * self.input_scale.unsqueeze(0)
        hidden = self.shared_trunk(
            adapted.reshape(batch_size * self.ensemble_size, self.input_dim)
        ).reshape(batch_size, self.ensemble_size, self.hidden_dim)
        member_logits = torch.einsum(
            "bkh,kch->bkc", hidden, self.member_weight
        ) + self.member_bias.unsqueeze(0)

        # Keep the probability mixture stable under autocast while preserving
        # float64 precision in numerical tests and analysis code.
        probability_dtype = (
            torch.float32
            if member_logits.dtype in {torch.float16, torch.bfloat16}
            else member_logits.dtype
        )
        member_probabilities = F.softmax(
            member_logits, dim=-1, dtype=probability_dtype
        )
        probabilities = member_probabilities.mean(dim=1)
        min_probability = torch.finfo(probabilities.dtype).tiny
        log_probabilities = probabilities.clamp_min(min_probability).log()
        return TabMJointHeadOutput(
            log_probabilities=log_probabilities,
            probabilities=probabilities,
            member_logits=member_logits,
            member_probabilities=member_probabilities,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_members(x).log_probabilities



class ReliabilityBranchFusion(nn.Module):
    """
    Missing-aware reliability mixture of diagnostic branches.

    Branches:
      0: joint branch over all filled modality features
      1..M: unimodal branches
      remaining: pairwise branches

    The classifier uses stable mean-pooled backbone features; token attention is
    kept for interpretation only.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_modalities: int,
        output_dim: int,
        num_layers_pred: int = 1,
        dropout: float = 0.5,
        dynamic_gating: bool = False,
        dynamic_joint_prior_boost: bool = True,
        dynamic_use_observed_mask: bool = False,
        supervised_router_observed_specialists_only: bool = False,
        complete_joint_only: bool = False,
        complete_specialist_weight: float = 1.0,
        confidence_mode: str = "evidence",
        centered_evidence_confidence: bool = False,
        class_conditional_fusion: bool = False,
        joint_head_ensemble_size: int = 1,
        missing_family_normalized_router: bool = False,
        missing_family_residual_gate: bool = False,
        uniform_branch_weights: bool = False,
        joint_branch_only: bool = False,
        prediction_observed_specialists_only: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        self.output_dim = output_dim
        self.uniform_branch_weights = bool(uniform_branch_weights)
        self.joint_branch_only = bool(joint_branch_only)
        self.pairs = list(combinations(range(num_modalities), 2))
        self.num_branches = 1 + num_modalities + len(self.pairs)
        self.dynamic_gating = bool(dynamic_gating)
        self.dynamic_joint_prior_boost = bool(dynamic_joint_prior_boost)
        self.dynamic_use_observed_mask = bool(dynamic_use_observed_mask)
        self.prediction_observed_specialists_only = bool(
            prediction_observed_specialists_only
        )
        self.supervised_router_observed_specialists_only = bool(
            supervised_router_observed_specialists_only
        )
        self.clean_dynamic_gating = bool(
            self.dynamic_gating
            and (
                not self.dynamic_joint_prior_boost
                or self.dynamic_use_observed_mask
                or self.supervised_router_observed_specialists_only
            )
        )
        if not self.dynamic_gating and (
            not self.dynamic_joint_prior_boost
            or self.dynamic_use_observed_mask
            or self.supervised_router_observed_specialists_only
        ):
            raise ValueError(
                "clean dynamic-router options require dynamic gating"
            )
        self.complete_joint_only = bool(complete_joint_only)
        if not 0.0 <= complete_specialist_weight <= 1.0:
            raise ValueError("complete_specialist_weight must be in [0, 1]")
        self.complete_specialist_weight = float(complete_specialist_weight)
        self.missing_family_normalized_router = bool(
            missing_family_normalized_router
        )
        self.missing_family_residual_gate = bool(
            missing_family_residual_gate
        )
        if (
            self.missing_family_residual_gate
            and not self.missing_family_normalized_router
        ):
            raise ValueError(
                "missing_family_residual_gate requires "
                "missing_family_normalized_router"
            )
        if self.missing_family_normalized_router and self.num_modalities < 2:
            raise ValueError(
                "missing-family routing requires at least two modalities"
            )
        if self.missing_family_normalized_router and self.dynamic_gating:
            raise ValueError(
                "missing-family routing v1 is incompatible with dynamic gating"
            )
        if confidence_mode not in {"evidence", "entropy_detached", "entropy_exp_detached"}:
            raise ValueError(f"Unsupported branch confidence mode: {confidence_mode}")
        self.confidence_mode = confidence_mode
        self.centered_evidence_confidence = bool(centered_evidence_confidence)
        self.class_conditional_fusion = bool(class_conditional_fusion)
        self.joint_head_ensemble_size = int(joint_head_ensemble_size)
        if self.joint_head_ensemble_size < 1:
            raise ValueError("joint_head_ensemble_size must be positive")
        if self.missing_family_normalized_router and self.class_conditional_fusion:
            raise ValueError(
                "missing-family routing v1 is incompatible with "
                "class-conditional fusion"
            )
        if (
            self.missing_family_normalized_router
            and self.joint_head_ensemble_size != 1
        ):
            raise ValueError(
                "missing-family routing v1 requires joint_head_ensemble_size=1"
            )

        if self.joint_head_ensemble_size == 1:
            # Do not wrap or reconstruct this path: old checkpoints depend on
            # the exact module type, initialization order, and state-dict keys.
            self.joint_head = MLP(
                hidden_dim * num_modalities,
                hidden_dim,
                output_dim,
                num_layers_pred,
                activation=nn.ReLU(),
                dropout=dropout,
            )
        else:
            self.joint_head = TabMJointEnsembleHead(
                hidden_dim * num_modalities,
                hidden_dim,
                output_dim,
                num_layers_pred,
                self.joint_head_ensemble_size,
                dropout=dropout,
            )
        self.unimodal_heads = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, output_dim, max(1, num_layers_pred), activation=nn.ReLU(), dropout=dropout)
            for _ in range(num_modalities)
        ])
        self.pair_heads = nn.ModuleList([
            MLP(hidden_dim * 4, hidden_dim, output_dim, max(1, num_layers_pred), activation=nn.ReLU(), dropout=dropout)
            for _ in self.pairs
        ])

        # Low-capacity prior only; sample-wise branch weights come from evidence
        # confidence and AGMG reliability instead of a free MLP gate.
        self.branch_prior = nn.Parameter(torch.zeros(self.num_branches))
        if self.class_conditional_fusion:
            # A bounded, branch-specific class correction.  Zero initialization
            # makes the initial prediction exactly equal to the scalar-weighted
            # fusion, while allowing different branches to specialize by class.
            self.branch_class_prior = nn.Parameter(
                torch.zeros(self.num_branches, self.output_dim)
            )
        else:
            self.register_parameter("branch_class_prior", None)
        if self.dynamic_gating:
            gate_input_dim = hidden_dim * num_modalities + 2 * num_modalities
            # Clean routing must not perturb initialization of any shared or
            # subsequently constructed module.  Legacy dynamic routing keeps
            # its historical construction-time RNG consumption.  Only CPU RNG
            # is isolated; training-mode gate dropout still consumes RNG.
            cpu_rng_state = (
                torch.random.get_rng_state()
                if self.clean_dynamic_gating
                else None
            )
            try:
                self.dynamic_gate = nn.Sequential(
                    nn.LayerNorm(gate_input_dim),
                    nn.Linear(gate_input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(min(dropout, 0.2)),
                    nn.Linear(hidden_dim, self.num_branches),
                )
                # Start at an exact zero correction while retaining trainable
                # hidden and output layers.
                nn.init.zeros_(self.dynamic_gate[-1].weight)
                nn.init.zeros_(self.dynamic_gate[-1].bias)
            finally:
                if cpu_rng_state is not None:
                    torch.random.set_rng_state(cpu_rng_state)
            self.dynamic_gate_scale = nn.Parameter(torch.tensor(0.5))
            if self.dynamic_joint_prior_boost:
                with torch.no_grad():
                    self.branch_prior[0] = math.log(2.0)

        # Optional missing-row family residual.  Parameter-free family
        # normalization is controlled independently by
        # ``missing_family_normalized_router``.  Constructing this gate is an
        # explicit second factor; its RNG is isolated so every historical and
        # subsequently constructed shared tensor retains its exact
        # initialization.
        self.missing_family_gate = None
        if self.missing_family_residual_gate:
            cpu_rng_state = torch.random.get_rng_state()
            try:
                gate_input_dim = 2 * self.num_modalities
                gate_hidden_dim = max(4, self.hidden_dim // 4)
                self.missing_family_gate = nn.Sequential(
                    nn.LayerNorm(gate_input_dim),
                    nn.Linear(gate_input_dim, gate_hidden_dim),
                    nn.GELU(),
                    nn.Linear(gate_hidden_dim, 3),
                )
                nn.init.zeros_(self.missing_family_gate[-1].weight)
                nn.init.zeros_(self.missing_family_gate[-1].bias)
            finally:
                torch.random.set_rng_state(cpu_rng_state)

        # Non-persistent, graph-carrying diagnostics for the optional ensemble.
        # They intentionally are not parameters or buffers, so K=1 state dicts
        # and checkpoints remain byte-for-byte structurally compatible.
        self.last_joint_member_logits = None
        self.last_joint_member_probabilities = None
        self.last_joint_member_evidence = None
        self.last_joint_probability = None
        self.last_joint_log_probability = None
        self.last_supervision_branch_mask = None
        self.last_missing_family_weights = None
        self.last_missing_family_mask = None

    def _evidence_confidence(self, branch_logits: torch.Tensor) -> torch.Tensor:
        confidence_logits = branch_logits
        if self.centered_evidence_confidence:
            # Softmax is invariant to a common class-logit shift.  Centering
            # removes the otherwise arbitrary shift from the evidence weight.
            confidence_logits = confidence_logits - confidence_logits.mean(
                dim=-1, keepdim=True
            )
        evidence_strength = F.softplus(confidence_logits).sum(dim=-1)
        return evidence_strength / (evidence_strength + float(self.output_dim))

    def _mix_probabilities(
        self,
        branch_probs: torch.Tensor,
        branch_weights: torch.Tensor,
    ) -> torch.Tensor:
        weighted = branch_probs * branch_weights.unsqueeze(-1)
        if self.branch_class_prior is None:
            # Keep the historical/default numerical path bit-for-bit compatible.
            return weighted.sum(dim=1).clamp_min(1e-8)
        # Bound the correction to exp(+/-0.5), preventing a tiny calibration
        # layer from overwhelming branch predictions or reliability weights.
        correction = torch.exp(0.5 * torch.tanh(self.branch_class_prior))
        probs = (weighted * correction.unsqueeze(0)).sum(dim=1).clamp_min(1e-8)
        return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def _branch_mask(self, usable_mask: torch.Tensor) -> torch.Tensor:
        B = usable_mask.shape[0]
        mask = torch.zeros(B, self.num_branches, device=usable_mask.device, dtype=torch.bool)
        any_usable = usable_mask.any(dim=1)
        mask[:, 0] = any_usable
        mask[:, 1:1 + self.num_modalities] = usable_mask
        offset = 1 + self.num_modalities
        for k, (i, j) in enumerate(self.pairs):
            mask[:, offset + k] = usable_mask[:, i] & usable_mask[:, j]
        no_branch = ~mask.any(dim=1)
        if no_branch.any():
            mask[no_branch, 0] = True
        return mask

    def _branch_quality(self, modality_reliability: torch.Tensor, branch_mask: torch.Tensor) -> torch.Tensor:
        quality = torch.zeros(
            modality_reliability.shape[0],
            self.num_branches,
            device=modality_reliability.device,
            dtype=modality_reliability.dtype,
        )
        usable = modality_reliability > 0
        denom = usable.float().sum(dim=1).clamp_min(1.0)
        quality[:, 0] = (modality_reliability * usable.float()).sum(dim=1) / denom
        quality[:, 1:1 + self.num_modalities] = modality_reliability
        offset = 1 + self.num_modalities
        for k, (i, j) in enumerate(self.pairs):
            quality[:, offset + k] = torch.sqrt(
                (modality_reliability[:, i] * modality_reliability[:, j]).clamp_min(0.0)
            )
        quality = quality.masked_fill(~branch_mask, 0.0)
        return quality

    def _missing_family_weights(
        self,
        branch_log_scores: torch.Tensor,
        branch_mask: torch.Tensor,
        observed_mask: torch.Tensor,
        modality_reliability: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return two-level branch/family weights for missing rows only.

        The family base score is the log-mean-exp of its active branches.
        Consequently equal branch scores allocate equal mass to joint,
        unimodal, and pairwise families instead of inheriting a 1:M:C(M,2)
        branch-count prior.  The optional learned gate is a separately
        controlled, zero-initialized residual on the three family scores.
        """

        if branch_log_scores.ndim != 2 or branch_log_scores.shape[1] != self.num_branches:
            raise ValueError("branch_log_scores has an invalid shape")
        if branch_mask.shape != branch_log_scores.shape or branch_mask.dtype != torch.bool:
            raise ValueError("branch_mask must be a boolean branch-score mask")
        expected_modality_shape = (
            branch_log_scores.shape[0],
            self.num_modalities,
        )
        if observed_mask.shape != expected_modality_shape or observed_mask.dtype != torch.bool:
            raise ValueError("observed_mask must be a boolean modality mask")
        if modality_reliability.shape != expected_modality_shape:
            raise ValueError("modality_reliability has an invalid shape")
        if not bool(observed_mask.any(dim=1).all()):
            raise ValueError(
                "missing-family routing requires at least one observed modality"
            )
        if not bool(branch_mask.any(dim=1).all()):
            raise ValueError(
                "missing-family routing requires at least one active branch"
            )

        family_slices = (
            slice(0, 1),
            slice(1, 1 + self.num_modalities),
            slice(1 + self.num_modalities, self.num_branches),
        )
        family_active_parts = []
        family_base_parts = []
        within_family_parts = []
        minimum_score = torch.finfo(branch_log_scores.dtype).min
        for location in family_slices:
            scores = branch_log_scores[:, location]
            active = branch_mask[:, location]
            family_active = active.any(dim=1)
            safe_scores = scores.masked_fill(~active, minimum_score)
            within = torch.softmax(safe_scores, dim=1)
            within = within.masked_fill(~active, 0.0)
            within = within / within.sum(dim=1, keepdim=True).clamp_min(1e-8)
            active_count = active.sum(dim=1).to(dtype=scores.dtype)
            family_base = torch.logsumexp(safe_scores, dim=1) - torch.log(
                active_count.clamp_min(1.0)
            )
            family_active_parts.append(family_active)
            family_base_parts.append(family_base)
            within_family_parts.append(within)

        family_mask = torch.stack(family_active_parts, dim=1)
        if not bool(family_mask.any(dim=1).all()):
            raise ValueError("missing-family routing found no active family")
        family_log_scores = torch.stack(family_base_parts, dim=1)
        if self.missing_family_gate is not None:
            gate_input = torch.cat(
                [
                    observed_mask.to(dtype=modality_reliability.dtype),
                    modality_reliability,
                ],
                dim=1,
            )
            family_log_scores = family_log_scores + self.missing_family_gate(
                gate_input
            )
        family_log_scores = family_log_scores.masked_fill(
            ~family_mask, minimum_score
        )
        family_weights = torch.softmax(family_log_scores, dim=1)
        family_weights = family_weights.masked_fill(~family_mask, 0.0)
        family_weights = family_weights / family_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)

        branch_weights = torch.zeros_like(branch_log_scores)
        for family_index, (location, within) in enumerate(
            zip(family_slices, within_family_parts)
        ):
            branch_weights[:, location] = (
                family_weights[:, family_index : family_index + 1] * within
            )
        return branch_weights, family_weights, family_mask

    def forward(
        self,
        modality_feats: List[torch.Tensor],
        usable_mask: torch.Tensor,
        modality_reliability: torch.Tensor,
        complete_mask: Optional[torch.Tensor] = None,
        observed_mask: Optional[torch.Tensor] = None,
    ):
        feats = torch.stack(modality_feats, dim=1)  # (B,M,D)
        flat = feats.flatten(1)

        joint_output = None
        if self.joint_head_ensemble_size == 1:
            joint_logits = self.joint_head(flat)
            self.last_joint_member_logits = None
            self.last_joint_member_probabilities = None
            self.last_joint_member_evidence = None
            self.last_joint_probability = None
            self.last_joint_log_probability = None
        else:
            joint_output = self.joint_head.forward_with_members(flat)
            joint_logits = joint_output.log_probabilities
            self.last_joint_member_logits = joint_output.member_logits
            self.last_joint_member_probabilities = (
                joint_output.member_probabilities
            )
            self.last_joint_probability = joint_output.probabilities
            self.last_joint_log_probability = joint_output.log_probabilities

        branch_logits = [joint_logits]
        for m in range(self.num_modalities):
            branch_logits.append(self.unimodal_heads[m](feats[:, m, :]))
        for head, (i, j) in zip(self.pair_heads, self.pairs):
            fi, fj = feats[:, i, :], feats[:, j, :]
            pair_feat = torch.cat([fi, fj, fi * fj, torch.abs(fi - fj)], dim=-1)
            branch_logits.append(head(pair_feat))
        branch_logits = torch.stack(branch_logits, dim=1)  # (B,branches,C)

        usable_branch_mask = self._branch_mask(usable_mask)
        if (
            self.dynamic_use_observed_mask
            or self.prediction_observed_specialists_only
            or self.supervised_router_observed_specialists_only
            or self.missing_family_normalized_router
        ):
            if observed_mask is None:
                raise ValueError(
                    "observed_mask is required by clean dynamic-router options"
                )
            if observed_mask.shape != usable_mask.shape:
                raise ValueError(
                    "observed_mask and usable_mask must have identical shapes"
                )
            if (
                self.missing_family_normalized_router
                and observed_mask.dtype != torch.bool
            ):
                raise ValueError(
                    "missing-family routing requires a boolean observed_mask"
                )
            observed_mask = observed_mask.bool()
        if self.prediction_observed_specialists_only:
            observed_branch_mask = self._branch_mask(observed_mask)
            # Completion remains available to the joint classifier.  Only the
            # unimodal and pairwise specialists are restricted to modalities
            # that were present in the original input.
            branch_mask = torch.cat(
                [usable_branch_mask[:, :1], observed_branch_mask[:, 1:]],
                dim=1,
            )
        else:
            branch_mask = usable_branch_mask
        if self.supervised_router_observed_specialists_only:
            observed_branch_mask = self._branch_mask(observed_mask)
            # The joint branch remains the same prediction-time joint branch;
            # only unimodal/pair teachers containing generated modalities are
            # excluded from label-supervised routing.
            supervision_branch_mask = torch.cat(
                [branch_mask[:, :1], observed_branch_mask[:, 1:]], dim=1
            )
        else:
            supervision_branch_mask = branch_mask
        self.last_supervision_branch_mask = supervision_branch_mask
        branch_quality = self._branch_quality(modality_reliability, branch_mask)

        if joint_output is None:
            # Preserve the historical/default numerical path exactly.
            branch_probs = F.softmax(branch_logits, dim=-1)
        else:
            # The joint expert is a probability ensemble.  Reusing its exact
            # mean avoids silently turning it into a mean-logit ensemble via a
            # second softmax; all specialist branches retain their old path.
            specialist_probs = F.softmax(branch_logits[:, 1:, :], dim=-1)
            joint_probability = joint_output.probabilities.to(
                dtype=specialist_probs.dtype
            )
            branch_probs = torch.cat(
                [joint_probability.unsqueeze(1), specialist_probs], dim=1
            )

            # Evidence for the joint branch is the mean evidence of its members,
            # not evidence computed from log(mean probability), whose arbitrary
            # log scale is incompatible with the existing reliability score.
            self.last_joint_member_evidence = self._evidence_confidence(
                joint_output.member_logits
            )
        if self.confidence_mode in {"entropy_detached", "entropy_exp_detached"}:
            entropy = -(branch_probs.clamp_min(1e-8) * branch_probs.clamp_min(1e-8).log()).sum(dim=-1)
            if self.confidence_mode == "entropy_exp_detached":
                branch_confidence = torch.exp(-entropy).detach()
            else:
                branch_confidence = (
                    1.0 - entropy / math.log(float(self.output_dim))
                ).clamp_min(1e-3).detach()
        else:
            if joint_output is None:
                branch_confidence = self._evidence_confidence(branch_logits)
            else:
                specialist_confidence = self._evidence_confidence(
                    branch_logits[:, 1:, :]
                )
                branch_confidence = torch.cat(
                    [
                        self.last_joint_member_evidence.mean(
                            dim=1, keepdim=True
                        ),
                        specialist_confidence,
                    ],
                    dim=1,
                )
        base_branch_log_scores = (
            torch.log(branch_quality.clamp_min(1e-8))
            + torch.log(branch_confidence.clamp_min(1e-8))
            + self.branch_prior.unsqueeze(0)
        )
        if self.dynamic_gating:
            gate_mask = (
                observed_mask
                if self.dynamic_use_observed_mask
                else usable_mask
            )
            gate_input = torch.cat(
                [flat, gate_mask.float(), modality_reliability], dim=1
            )
            dynamic_offset = torch.tanh(self.dynamic_gate(gate_input))
            dynamic_correction = self.dynamic_gate_scale * dynamic_offset
            branch_log_scores = base_branch_log_scores + dynamic_correction
            # The supervised-router target is label-derived.  Detach the
            # heuristic evidence/quality base so that this auxiliary cannot be
            # reduced through arbitrary common shifts of branch class logits.
            # The dynamic correction remains differentiable through both its
            # gate and the pooled backbone features used as gate input.
            supervision_log_scores = (
                base_branch_log_scores.detach() + dynamic_correction
            )
        else:
            # Keep the historical prediction path bit-for-bit: no extra add is
            # performed while dynamic routing is disabled.
            branch_log_scores = base_branch_log_scores
            supervision_log_scores = base_branch_log_scores.detach()
        branch_log_scores = branch_log_scores.masked_fill(~branch_mask, -1e4)
        supervision_log_scores = supervision_log_scores.masked_fill(
            ~supervision_branch_mask, -1e4
        )
        branch_weights = F.softmax(branch_log_scores, dim=1)
        if self.uniform_branch_weights:
            active = branch_mask.to(branch_log_scores.dtype)
            branch_weights = active / active.sum(dim=1, keepdim=True).clamp_min(1.0)
        if self.joint_branch_only:
            branch_weights = torch.zeros_like(branch_weights)
            branch_weights[:, 0] = 1.0

        probs = self._mix_probabilities(branch_probs, branch_weights)
        self.last_missing_family_weights = None
        self.last_missing_family_mask = None
        if self.missing_family_normalized_router:
            if complete_mask is None:
                raise ValueError(
                    "complete_mask is required by missing-family routing"
                )
            if complete_mask.ndim != 1 or complete_mask.shape[0] != branch_weights.shape[0]:
                raise ValueError("complete_mask has an invalid shape")
            if complete_mask.dtype != torch.bool:
                raise ValueError(
                    "missing-family routing requires a boolean complete_mask"
                )
            expected_complete_mask = observed_mask.all(dim=1)
            if not torch.equal(complete_mask, expected_complete_mask):
                raise ValueError(
                    "missing-family routing requires complete_mask to equal "
                    "raw observed_mask.all(dim=1)"
                )
            missing_mask = ~complete_mask
            if missing_mask.any():
                missing_weights, family_weights, family_mask = (
                    self._missing_family_weights(
                        branch_log_scores[missing_mask],
                        branch_mask[missing_mask],
                        observed_mask[missing_mask],
                        modality_reliability[missing_mask],
                    )
                )
                rerouted_probs = self._mix_probabilities(
                    branch_probs[missing_mask], missing_weights
                )
                # CopySlices preserves complete rows from the already-executed
                # historical path exactly; the gate is never called when a
                # batch contains complete rows only.
                branch_weights = branch_weights.clone()
                branch_weights[missing_mask] = missing_weights
                probs = probs.clone()
                probs[missing_mask] = rerouted_probs
                self.last_missing_family_weights = family_weights
                self.last_missing_family_mask = family_mask
        if self.complete_joint_only:
            if complete_mask is None:
                raise ValueError("complete_mask is required when complete_joint_only is enabled")
            complete_mask = complete_mask.bool()
            if complete_mask.any():
                probs = torch.where(
                    complete_mask.unsqueeze(1),
                    branch_probs[:, 0, :],
                    probs,
                )
                branch_weights = branch_weights.clone()
                branch_weights[complete_mask] = 0.0
                branch_weights[complete_mask, 0] = 1.0
        elif self.complete_specialist_weight < 1.0:
            if complete_mask is None:
                raise ValueError("complete_mask is required for anchored complete-sample fusion")
            complete_mask = complete_mask.bool()
            if complete_mask.any():
                specialist_weights = branch_weights[:, 1:]
                specialist_weights = specialist_weights / specialist_weights.sum(
                    dim=1, keepdim=True
                ).clamp_min(1e-8)
                anchored_weights = torch.cat(
                    [
                        torch.full_like(branch_weights[:, :1], 1.0 - self.complete_specialist_weight),
                        self.complete_specialist_weight * specialist_weights,
                    ],
                    dim=1,
                )
                anchored_probs = self._mix_probabilities(branch_probs, anchored_weights)
                probs = torch.where(complete_mask.unsqueeze(1), anchored_probs, probs)
                branch_weights = torch.where(
                    complete_mask[:, None], anchored_weights, branch_weights
                )
        logits = torch.log(probs)
        return (
            logits,
            branch_logits,
            branch_weights,
            branch_mask,
            branch_quality,
            branch_log_scores,
            supervision_log_scores,
        )

    def branch_aux_loss(self, branch_logits: torch.Tensor, branch_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        losses = []
        for k in range(branch_logits.shape[1]):
            mask_k = branch_mask[:, k]
            if mask_k.any():
                losses.append(F.cross_entropy(branch_logits[mask_k, k, :], labels[mask_k]))
        if len(losses) == 0:
            return torch.tensor(0.0, device=branch_logits.device)
        return torch.stack(losses).mean()


class OrdinalFusionHead(nn.Module):
    """Two monotonic cumulative logits for the ordered 0 / 1 / 2+ target."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(min(dropout, 0.2)),
            nn.Linear(hidden_dim, 1),
        )
        self.first_cut = nn.Parameter(torch.tensor(-0.5))
        # softplus(0.5413) is approximately one.
        self.cut_gap_raw = nn.Parameter(torch.tensor(0.5413))

    def forward(self, flat_features: torch.Tensor):
        score = self.score(flat_features)
        cuts = torch.stack((
            self.first_cut,
            self.first_cut + F.softplus(self.cut_gap_raw),
        ))
        cumulative_logits = score - cuts.unsqueeze(0)
        cumulative = torch.sigmoid(cumulative_logits)
        probabilities = torch.stack((
            1.0 - cumulative[:, 0],
            cumulative[:, 0] - cumulative[:, 1],
            cumulative[:, 1],
        ), dim=1).clamp_min(1e-8)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
        return cumulative_logits, probabilities


class OrdinalContinuationHead(nn.Module):
    """Non-proportional continuation-ratio head for ordered 0 / 1 / 2+ labels.

    The first logit predicts P(y >= 1); the second predicts
    P(y == 2 | y >= 1).  Unlike a proportional-odds head, the two boundaries
    can use different feature directions while always yielding valid class
    probabilities.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(min(dropout, 0.2)),
        )
        self.boundaries = nn.Linear(hidden_dim, 2)

    def forward(self, flat_features: torch.Tensor):
        continuation_logits = self.boundaries(self.shared(flat_features))
        probability_ge_one = torch.sigmoid(continuation_logits[:, 0])
        probability_two_given_ge_one = torch.sigmoid(continuation_logits[:, 1])
        probabilities = torch.stack(
            (
                1.0 - probability_ge_one,
                probability_ge_one * (1.0 - probability_two_given_ge_one),
                probability_ge_one * probability_two_given_ge_one,
            ),
            dim=1,
        ).clamp_min(1e-8)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
        return continuation_logits, probabilities


def uncertainty_weighted_ordinal_mix(
    categorical_logits: torch.Tensor,
    ordinal_probabilities: torch.Tensor,
    max_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Blend an ordinal prediction only where the categorical head is uncertain.

    The detached per-sample blend weight is
    ``max_weight * (H(softmax(logits)) / log(num_classes)) ** 2``.  Detaching
    the weight prevents the categorical head from increasing its entropy merely
    to route more probability mass through the ordinal head; gradients still
    flow through both probability paths.
    """
    if categorical_logits.ndim != 2:
        raise ValueError("categorical_logits must have shape (batch, classes)")
    if ordinal_probabilities.shape != categorical_logits.shape:
        raise ValueError(
            "ordinal_probabilities must match categorical_logits shape"
        )
    num_classes = categorical_logits.shape[1]
    if num_classes < 2:
        raise ValueError("uncertainty-aware ordinal fusion needs at least two classes")
    max_weight = float(max_weight)
    if not math.isfinite(max_weight) or not 0.0 <= max_weight <= 1.0:
        raise ValueError("max_weight must be finite and in [0, 1]")

    categorical_probabilities = torch.softmax(categorical_logits, dim=1)
    entropy = torch.special.entr(categorical_probabilities).sum(
        dim=1, keepdim=True
    )
    normalized_entropy = entropy / math.log(float(num_classes))
    blend_weight = (
        max_weight * normalized_entropy.clamp(min=0.0, max=1.0).square()
    ).detach()
    probabilities = (
        (1.0 - blend_weight) * categorical_probabilities
        + blend_weight * ordinal_probabilities
    ).clamp_min(1e-8)
    probabilities = probabilities / probabilities.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-8)
    return probabilities, blend_weight


def sample_observed_reconstruction_groups(
    observed_mask: torch.Tensor,
    targets_per_sample: int,
    context_dropout_probability: float = 0.0,
    disable_stochastic_context_masking: bool = False,
) -> dict[tuple[int, tuple[int, ...]], torch.Tensor]:
    """Group per-sample observed reconstruction tasks by target and context.

    Every eligible sample has at least two naturally observed modalities.  At
    zero context dropout, each target is reconstructed from all other observed
    modalities (the backward-compatible leave-one-out path).  With positive
    dropout, one shared non-empty proper subset of the observed modalities is
    sampled as context and the masked observed modalities become reconstruction
    targets.  Grouping equal ``(target, context-pattern)`` tasks preserves
    batched generator execution without making eligibility depend on other
    samples in the minibatch.
    """
    if observed_mask.ndim != 2:
        raise ValueError("observed_mask must have shape (batch, modalities)")
    targets_per_sample = int(targets_per_sample)
    if targets_per_sample < 0:
        raise ValueError("targets_per_sample must be non-negative")
    if targets_per_sample == 0 or observed_mask.shape[0] == 0:
        return {}

    context_dropout_probability = float(context_dropout_probability)
    if (
        not math.isfinite(context_dropout_probability)
        or not 0.0 <= context_dropout_probability <= 1.0
    ):
        raise ValueError(
            "context_dropout_probability must be finite and in [0, 1]"
        )

    observed_mask = observed_mask.bool()
    batch_size, num_modalities = observed_mask.shape
    patterns = observed_mask.detach().cpu().tolist()
    needs_sampling = any(
        2 <= sum(pattern) and targets_per_sample < sum(pattern)
        for pattern in patterns
    )
    # Draw all priorities in one operation only when a strict target subset is
    # requested.  With the default K=M, exhaustive targets do not consume an
    # otherwise unrelated RNG draw.
    priorities = (
        torch.rand(
            batch_size,
            num_modalities,
            device=observed_mask.device,
        ).detach().cpu().tolist()
        if needs_sampling
        else None
    )
    # Only the opt-in stochastic-subset path consumes this RNG draw.  One row
    # defines a shared observed subset for every reconstruction target from the
    # sample; fallbacks retain at least one context and at least one target.
    subset_priorities = (
        torch.rand(
            batch_size,
            num_modalities,
            device=observed_mask.device,
        ).detach().cpu().tolist()
        if context_dropout_probability > 0
        else None
    )
    grouped_rows: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for sample_index, pattern in enumerate(patterns):
        observed_modalities = [
            modality_index
            for modality_index, is_observed in enumerate(pattern)
            if is_observed
        ]
        if len(observed_modalities) < 2:
            continue
        if subset_priorities is not None:
            row_priorities = subset_priorities[sample_index]
            context_modalities = [
                modality_index for modality_index in observed_modalities
                if row_priorities[modality_index] >= context_dropout_probability
            ]
            if not context_modalities:
                context_modalities = [max(observed_modalities, key=lambda m: row_priorities[m])]
            if len(context_modalities) == len(observed_modalities):
                context_modalities.remove(min(observed_modalities, key=lambda m: row_priorities[m]))
            target_modalities = [m for m in observed_modalities if m not in context_modalities]
            target_modalities = sorted(target_modalities, key=lambda m: row_priorities[m])[:targets_per_sample]
            context_modalities = tuple(sorted(context_modalities))
        else:
            target_count = min(targets_per_sample, len(observed_modalities))
            if target_count == len(observed_modalities):
                target_modalities = observed_modalities
            else:
                if priorities is None:
                    raise RuntimeError("Target priorities were not initialized")
                target_modalities = sorted(observed_modalities, key=lambda m: priorities[sample_index][m])[:target_count]
        for target_modality in target_modalities:
            if disable_stochastic_context_masking or subset_priorities is None:
                current_context_modalities = tuple(
                    modality_index
                    for modality_index in observed_modalities
                    if modality_index != target_modality
                )
            else:
                current_context_modalities = context_modalities
            # Eligibility above guarantees a real observed context.
            grouped_rows.setdefault(
                (target_modality, current_context_modalities), []
            ).append(sample_index)

    return {
        key: torch.as_tensor(
            rows,
            dtype=torch.long,
            device=observed_mask.device,
        )
        for key, rows in grouped_rows.items()
    }


# =========================
# Final Model
# =========================
class AGMGFlexMoE(nn.Module):
    def __init__(
        self,
        num_modalities: int,
        full_modality_index: int,
        num_patches: int,
        hidden_dim: int,
        output_dim: int,
        num_layers_fus: int,
        num_layers_pred: int,
        num_experts: int,
        num_routers: int,
        top_k: int,
        num_heads: int = 4,
        dropout: float = 0.5,
        gen_num_layers: int = 2,
        gen_num_heads: int = 4,
        recon_use_token_mse: bool = False,
        recon_token_mse_weight: float = 0.05,
        pattern_aware_reconstruction: bool = False,
        recon_normalized_token_loss_weight: float = 0.0,
        recon_context_dropout_probability: float = 0.0,
        recon_encoder_gradient_scale: float = 1.0,
        vectorized_generation: bool = False,
        recon_targets_per_sample: int = 0,
        use_generators: bool = True,
        dynamic_branch_fusion: bool = False,
        dynamic_branch_joint_prior_boost: bool = True,
        dynamic_branch_use_observed_mask: bool = False,
        supervised_router_observed_specialists_only: bool = False,
        complete_joint_only: bool = False,
        complete_specialist_weight: float = 1.0,
        branch_confidence_mode: str = "evidence",
        token_attention_init: float = -4.0,
        generator_task_grad: bool = False,
        generator_only_task_grad: bool = False,
        generator_output_gate: bool = True,
        standard_transformer_residual: bool = False,
        gated_transformer_residual: bool = False,
        ordinal_fusion_weight: float = 0.0,
        ordinal_aux_loss_weight: float = 0.0,
        ordinal_head_type: str = "proportional",
        uncertainty_aware_ordinal_fusion: bool = False,
        enable_class1_aux_head: bool = False,
        learn_observed_reliability: bool = False,
        centered_evidence_confidence: bool = False,
        class_conditional_fusion: bool = False,
        normalized_gate_loss: bool = False,
        enable_supervised_contrastive: bool = False,
        supervised_contrastive_projection_dim: int = 64,

        # Kept for backward-compatible constructor calls; current model uses hidden_dim features directly.
        unique_dim: Optional[int] = None,

        # Kept for backward-compatible constructor calls; ignored by current token-importance model.
        ortho_reg_weight: float = 0.0,
        shared_align_weight: float = 0.0,

        # Default one preserves the historical joint MLP exactly.  Values above
        # one opt into the parameter-efficient TabM-style joint ensemble.
        joint_head_ensemble_size: int = 1,

        # Optional MORE-inspired low-rank adaptation of the final fused logits.
        # Zero is deliberately the default so historical construction consumes
        # no additional RNG and creates no additional state-dict entries.
        more_tail_rank: int = 0,

        # Optional dual local-boundary residual (DLBR) on the final fused
        # three-class logits.  A zero loss weight constructs no module and
        # therefore preserves the historical module tree, RNG trajectory, and
        # state dict exactly.
        dual_local_boundary_loss_weight: float = 0.0,

        # Optional ADHD-presentation residual on the final fused three-class
        # logits.  The neutral loss weight constructs no module, preserving
        # historical state dictionaries and RNG trajectories exactly.
        presentation_axis_loss_weight: float = 0.0,
        presentation_axis_residual_cap: float = 0.5,

        # Optional pooled-capacity residual applied only to rows that are
        # missing at least one modality in the current forward view.  Zero
        # constructs no module, so historical checkpoints and RNG trajectories
        # remain exact.
        missing_capacity_residual_width: int = 0,

        # Optional two-level joint/unimodal/pairwise normalization on missing
        # rows only.  The residual gate is a separately controlled extension
        # and requires the parameter-free normalized router.
        missing_family_normalized_router: bool = False,
        missing_family_residual_gate: bool = False,
        dense_backbone: bool = False,
        disable_provenance: bool = False,
        uniform_branch_weights: bool = False,
        joint_branch_only: bool = False,
        mean_pooling_only: bool = False,
        disable_stochastic_context_masking: bool = False,
        disable_completion: bool = False,
        no_output_gate: bool = False,
        prediction_observed_specialists_only: bool = False,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim
        self.dense_backbone = bool(dense_backbone)
        self.disable_provenance = bool(disable_provenance)
        self.uniform_branch_weights = bool(uniform_branch_weights)
        self.joint_branch_only = bool(joint_branch_only)
        self.mean_pooling_only = bool(mean_pooling_only)
        self.disable_stochastic_context_masking = bool(disable_stochastic_context_masking)
        self.disable_completion = bool(disable_completion)
        self.no_output_gate = bool(no_output_gate)

        self.recon_use_token_mse = recon_use_token_mse
        self.recon_token_mse_weight = recon_token_mse_weight
        self.pattern_aware_reconstruction = bool(pattern_aware_reconstruction)
        if (
            not math.isfinite(float(recon_normalized_token_loss_weight))
            or recon_normalized_token_loss_weight < 0
        ):
            raise ValueError(
                "recon_normalized_token_loss_weight must be finite and non-negative"
            )
        self.recon_normalized_token_loss_weight = float(
            recon_normalized_token_loss_weight
        )
        if (
            not math.isfinite(float(recon_context_dropout_probability))
            or not 0.0 <= float(recon_context_dropout_probability) <= 1.0
        ):
            raise ValueError(
                "recon_context_dropout_probability must be finite and in [0, 1]"
            )
        self.recon_context_dropout_probability = float(
            recon_context_dropout_probability
        )
        if (
            not math.isfinite(float(recon_encoder_gradient_scale))
            or not 0.0 <= float(recon_encoder_gradient_scale) <= 1.0
        ):
            raise ValueError(
                "recon_encoder_gradient_scale must be finite and in [0, 1]"
            )
        # This affects only the reconstruction context's backward pass.  The
        # historical value one is an exact identity, consumes no RNG, and
        # creates no parameter or buffer.  Classification-time generation is
        # controlled independently by generator_task_grad below.
        self.recon_encoder_gradient_scale = float(
            recon_encoder_gradient_scale
        )
        self.vectorized_generation = vectorized_generation
        self.recon_targets_per_sample = max(0, int(recon_targets_per_sample))
        self.use_generators = bool(use_generators)
        self.generator_task_grad = bool(generator_task_grad)
        self.generator_only_task_grad = bool(generator_only_task_grad)
        self.generator_output_gate = bool(generator_output_gate) and not self.no_output_gate
        if self.generator_task_grad and self.generator_only_task_grad:
            raise ValueError(
                "generator_task_grad and generator_only_task_grad are mutually "
                "exclusive"
            )
        if self.generator_only_task_grad and not self.use_generators:
            raise ValueError(
                "generator_only_task_grad requires use_generators=True"
            )
        self.ordinal_fusion_weight = float(ordinal_fusion_weight)
        self.ordinal_aux_loss_weight = float(ordinal_aux_loss_weight)
        if ordinal_head_type not in {"proportional", "continuation"}:
            raise ValueError(f"Unsupported ordinal head type: {ordinal_head_type}")
        self.ordinal_head_type = ordinal_head_type
        self.uncertainty_aware_ordinal_fusion = bool(
            uncertainty_aware_ordinal_fusion
        )
        if (
            self.uncertainty_aware_ordinal_fusion
            and self.ordinal_head_type != "continuation"
        ):
            raise ValueError(
                "uncertainty-aware ordinal fusion requires the continuation head"
            )
        self.enable_class1_aux_head = bool(enable_class1_aux_head)
        if self.enable_class1_aux_head and output_dim != 3:
            raise ValueError("class-1 auxiliary supervision requires three classes")
        self.learn_observed_reliability = bool(learn_observed_reliability)
        self.missing_family_normalized_router = bool(
            missing_family_normalized_router
        )
        self.missing_family_residual_gate = bool(
            missing_family_residual_gate
        )
        self.enable_supervised_contrastive = bool(enable_supervised_contrastive)
        self.more_tail_rank = int(more_tail_rank)
        if self.more_tail_rank < 0:
            raise ValueError("more_tail_rank must be non-negative")
        if self.more_tail_rank > 0 and output_dim < 2:
            raise ValueError(
                "MORE fused-logit tail adaptation requires at least two classes"
            )
        self.dual_local_boundary_loss_weight = float(
            dual_local_boundary_loss_weight
        )
        if (
            not math.isfinite(self.dual_local_boundary_loss_weight)
            or self.dual_local_boundary_loss_weight < 0
        ):
            raise ValueError(
                "DLBR loss weight must be finite and non-negative"
            )
        if self.dual_local_boundary_loss_weight > 0 and output_dim != 3:
            raise ValueError("DLBR requires exactly three output classes")
        self.presentation_axis_loss_weight = float(
            presentation_axis_loss_weight
        )
        self.presentation_axis_residual_cap = float(
            presentation_axis_residual_cap
        )
        if (
            not math.isfinite(self.presentation_axis_loss_weight)
            or self.presentation_axis_loss_weight < 0
        ):
            raise ValueError(
                "presentation-axis loss weight must be finite and non-negative"
            )
        if (
            not math.isfinite(self.presentation_axis_residual_cap)
            or self.presentation_axis_residual_cap <= 0
        ):
            raise ValueError(
                "presentation-axis residual cap must be finite and positive"
            )
        if self.presentation_axis_loss_weight > 0:
            if output_dim != 3:
                raise ValueError(
                    "presentation-axis residual requires exactly three output classes"
                )
            incompatible = []
            if self.ordinal_fusion_weight > 0 or self.ordinal_aux_loss_weight > 0:
                incompatible.append("ordinal fusion/auxiliary")
            if self.enable_class1_aux_head:
                incompatible.append("class-1 auxiliary")
            if self.dual_local_boundary_loss_weight > 0:
                incompatible.append("DLBR")
            if incompatible:
                raise ValueError(
                    "presentation-axis residual is mutually exclusive with "
                    + ", ".join(incompatible)
                )
        if (
            type(missing_capacity_residual_width) is not int
            or missing_capacity_residual_width < 0
        ):
            raise ValueError(
                "missing_capacity_residual_width must be a non-negative integer"
            )
        self.missing_capacity_residual_width = missing_capacity_residual_width
        if self.missing_capacity_residual_width > 0:
            if output_dim != 3:
                raise ValueError(
                    "missing-capacity residual requires exactly three output classes"
                )
            if num_modalities != 4:
                raise ValueError(
                    "missing-capacity residual requires exactly four modalities"
                )
        supervised_contrastive_projection_dim = int(
            supervised_contrastive_projection_dim
        )
        if (
            self.enable_supervised_contrastive
            and supervised_contrastive_projection_dim <= 0
        ):
            raise ValueError(
                "supervised contrastive projection dimension must be positive"
            )

        self.unique_dim = hidden_dim
        self.ortho_reg_weight = ortho_reg_weight
        self.shared_align_weight = shared_align_weight

        self.backbone = FlexMoE(
            num_modalities=num_modalities,
            full_modality_index=full_modality_index,
            num_patches=num_patches,
            hidden_dim=hidden_dim,
            num_layers=num_layers_fus,
            num_experts=num_experts,
            num_routers=num_routers,
            top_k=top_k,
            num_heads=num_heads,
            dropout=dropout,
            standard_residual=standard_transformer_residual,
            gated_residual=gated_transformer_residual,
            normalized_gate_loss=normalized_gate_loss,
            dense_backbone=self.dense_backbone,
        )

        self.generators = nn.ModuleList([
            ConditionalGenerator(
                hidden_dim,
                num_patches,
                num_heads=gen_num_heads,
                num_layers=gen_num_layers,
                dropout=0.1,
                use_output_gate=self.generator_output_gate,
            )
            for _ in range(num_modalities)
        ]) if self.use_generators else nn.ModuleList()
        self.flag_embeds = nn.ModuleList([ModalityFlagEmbedding(hidden_dim) for _ in range(num_modalities)])
        self.recon_projectors = nn.ModuleList([
            ReconProjector(hidden_dim) for _ in range(num_modalities)
        ]) if self.use_generators else nn.ModuleList()
        self.token_poolers = nn.ModuleList([
            AttentiveTokenPooler(
                hidden_dim,
                temperature=0.5,
                dropout=0.1,
                attention_mix_init=token_attention_init,
                mean_pooling_only=self.mean_pooling_only,
            )
            for _ in range(num_modalities)
        ])
        self.reliability_scorers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(num_modalities)
        ])
        if self.learn_observed_reliability:
            # Start observed reliabilities at exactly one.  The bounded mapping
            # in forward then learns small, sample-specific deviations without
            # permitting branch collapse.
            for scorer in self.reliability_scorers:
                nn.init.zeros_(scorer[-1].weight)
                nn.init.zeros_(scorer[-1].bias)
        self.generated_reliability_bias = nn.Parameter(torch.full((num_modalities,), -1.0))

        # ===== classification with reliability-gated diagnostic branches =====
        self.output_dim = output_dim
        self.num_layers_pred = num_layers_pred
        self.branch_fusion = ReliabilityBranchFusion(
            hidden_dim=hidden_dim,
            num_modalities=num_modalities,
            output_dim=output_dim,
            num_layers_pred=num_layers_pred,
            dropout=dropout,
            dynamic_gating=dynamic_branch_fusion,
            dynamic_joint_prior_boost=dynamic_branch_joint_prior_boost,
            dynamic_use_observed_mask=dynamic_branch_use_observed_mask,
            prediction_observed_specialists_only=(
                prediction_observed_specialists_only
            ),
            supervised_router_observed_specialists_only=(
                supervised_router_observed_specialists_only
            ),
            complete_joint_only=complete_joint_only,
            complete_specialist_weight=complete_specialist_weight,
            confidence_mode=branch_confidence_mode,
            centered_evidence_confidence=centered_evidence_confidence,
            class_conditional_fusion=class_conditional_fusion,
            joint_head_ensemble_size=joint_head_ensemble_size,
            missing_family_normalized_router=(
                self.missing_family_normalized_router
            ),
            missing_family_residual_gate=self.missing_family_residual_gate,
            uniform_branch_weights=self.uniform_branch_weights,
            joint_branch_only=self.joint_branch_only,
        )
        use_ordinal_head = output_dim == 3 and (
            self.ordinal_fusion_weight > 0 or self.ordinal_aux_loss_weight > 0
        )
        if use_ordinal_head and self.ordinal_head_type == "continuation":
            self.ordinal_head = OrdinalContinuationHead(
                hidden_dim * num_modalities, hidden_dim, dropout
            )
        elif use_ordinal_head:
            self.ordinal_head = OrdinalFusionHead(
                hidden_dim * num_modalities, hidden_dim, dropout
            )
        else:
            self.ordinal_head = None

        # optional temperature for analysis-only importance
        self.imp_temp = nn.Parameter(torch.tensor(1.0))

        # Training-only one-vs-rest supervision for the heterogeneous middle
        # class.  This module is deliberately appended after every historical
        # parameter so the default-off construction and state dict remain
        # exactly checkpoint-compatible.  Its logit is never fused into the
        # categorical prediction.
        self.class1_aux_head = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim * num_modalities),
                nn.Linear(hidden_dim * num_modalities, max(1, hidden_dim // 2)),
                nn.GELU(),
                nn.Dropout(min(dropout, 0.2)),
                nn.Linear(max(1, hidden_dim // 2), 1),
            )
            if self.enable_class1_aux_head
            else None
        )

        # Training-only projection of the exact pooled feature vector consumed
        # by the joint diagnostic branch.  Appending the optional module after
        # every historical parameter and constructing nothing while disabled
        # preserves the old state dict and RNG path exactly.
        self.supervised_contrastive_projector = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim * num_modalities),
                nn.Linear(hidden_dim * num_modalities, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, supervised_contrastive_projection_dim),
            )
            if self.enable_supervised_contrastive
            else None
        )

        # Classifier-level low-rank residual W = W_g + BA, applied to the exact
        # flattened pooled feature consumed by the joint diagnostic head.  A
        # keeps nn.Linear's standard initialization; B is zero initialized so
        # enabling the adapter starts with final_logits == base_logits exactly.
        # These modules are constructed only when enabled.  In particular, the
        # rank-zero path neither changes the module tree/state dict nor consumes
        # an RNG draw, preserving historical checkpoint replay.
        if self.more_tail_rank > 0:
            self.more_tail_A = nn.Linear(
                hidden_dim * num_modalities,
                self.more_tail_rank,
                bias=False,
            )
            self.more_tail_B = nn.Linear(
                self.more_tail_rank,
                output_dim,
                bias=False,
            )
            nn.init.zeros_(self.more_tail_B.weight)

        # Two independent, intercept-free local boundary directions operate on
        # the exact flattened pooled feature used by the joint diagnostic
        # branch.  LayerNorm has no affine parameters, and both heads are
        # exactly zero initialized.  Consequently an enabled DLBR model starts
        # with the identical final logits while learning sample-dependent
        # corrections rather than a global class bias.  The optional modules
        # are appended after all historical modules and are not constructed at
        # the neutral default.
        if self.dual_local_boundary_loss_weight > 0:
            flat_dim = hidden_dim * num_modalities
            self.dual_local_boundary_norm = nn.LayerNorm(
                flat_dim,
                elementwise_affine=False,
            )
            self.dual_local_boundary_heads = nn.ModuleList(
                (
                    nn.Linear(flat_dim, 1, bias=False),
                    nn.Linear(flat_dim, 1, bias=False),
                )
            )
            for head in self.dual_local_boundary_heads:
                nn.init.zeros_(head.weight)

        # A single intercept-free two-axis head models inattentive (IA) and
        # hyperactive/impulsive (HI) presentation evidence from the exact
        # pooled multimodal feature used by the joint classifier.  For ABCD's
        # four modalities this is precisely Linear(4 * hidden_dim, 2).  The
        # affine-free normalization and exactly zero-initialized projection
        # ensure that enabling the candidate starts from identical logits.
        if self.presentation_axis_loss_weight > 0:
            flat_dim = hidden_dim * num_modalities
            self.presentation_axis_norm = nn.LayerNorm(
                flat_dim,
                elementwise_affine=False,
            )
            # nn.Linear performs a random default initialization before the
            # explicit zeroing.  Preserve/restore the CPU generator so this
            # zero-initialized opt-in head cannot perturb any later dropout or
            # caller-side RNG sequence relative to the disabled anchor.
            cpu_rng_state = torch.random.get_rng_state()
            try:
                self.presentation_axis_head = nn.Linear(
                    flat_dim,
                    2,
                    bias=False,
                )
                nn.init.zeros_(self.presentation_axis_head.weight)
            finally:
                torch.random.set_rng_state(cpu_rng_state)

        # This intercept-free MLP adds capacity only for current-view missing
        # rows.  Its final projection is exactly zero initialized, making every
        # initial logit bit-identical to the H64 anchor.  Preserve/restore the
        # CPU RNG around the complete optional construction so enabling the
        # candidate cannot perturb any shared state or caller-side RNG draw.
        if self.missing_capacity_residual_width > 0:
            flat_dim = hidden_dim * num_modalities
            cpu_rng_state = torch.random.get_rng_state()
            try:
                self.missing_capacity_residual = nn.Sequential(
                    nn.LayerNorm(flat_dim, elementwise_affine=False),
                    nn.Linear(
                        flat_dim,
                        self.missing_capacity_residual_width,
                        bias=False,
                    ),
                    nn.GELU(),
                    nn.Linear(
                        self.missing_capacity_residual_width,
                        output_dim,
                        bias=False,
                    ),
                )
                nn.init.zeros_(self.missing_capacity_residual[-1].weight)
            finally:
                torch.random.set_rng_state(cpu_rng_state)

    def gate_loss(self):
        return self.backbone.gate_loss()

    def set_full_modality(self, is_full_modality):
        self.backbone.set_full_modality(is_full_modality)

    def _collect_context_tokens_full(self, per_sample_tokens: List[torch.Tensor], target_m: int):
        ctx = []
        for j in range(self.num_modalities):
            if j == target_m:
                continue
            ctx.append(per_sample_tokens[j])
        return torch.cat(ctx, dim=0)

    def _collect_context_tokens_general(self, per_sample_tokens: List[torch.Tensor], obs_row: torch.Tensor, target_m: int):
        ctx = []
        for j in range(self.num_modalities):
            if j == target_m:
                continue
            if bool(obs_row[j]):
                ctx.append(per_sample_tokens[j])
        if len(ctx) == 0:
            return None
        return torch.cat(ctx, dim=0)

    def _scale_reconstruction_context_gradient(
        self, context_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Preserve context values while scaling only encoder-side gradients."""
        scale = self.recon_encoder_gradient_scale
        if scale == 1.0:
            # Preserve the exact historical graph and numerical path.
            return context_tokens
        if scale == 0.0:
            return context_tokens.detach()
        detached = context_tokens.detach()
        return detached + scale * (context_tokens - detached)

    def _recon_loss_disc(self, m: int, pred_tokens: torch.Tensor, target_tokens: torch.Tensor):
        pred_pool = pred_tokens.mean(dim=0)
        tgt_pool = target_tokens.mean(dim=0)
        z_pred = self.recon_projectors[m](pred_pool)
        z_tgt = self.recon_projectors[m](tgt_pool).detach()
        cos = F.cosine_similarity(z_pred.unsqueeze(0), z_tgt.unsqueeze(0), dim=-1)
        loss = (1.0 - cos).mean()
        if self.recon_use_token_mse:
            loss = loss + self.recon_token_mse_weight * F.mse_loss(pred_tokens, target_tokens, reduction="mean")
        if self.recon_normalized_token_loss_weight > 0:
            pred_normalized = F.layer_norm(
                pred_tokens, (pred_tokens.shape[-1],)
            )
            target_normalized = F.layer_norm(
                target_tokens.detach(), (target_tokens.shape[-1],)
            )
            loss = loss + self.recon_normalized_token_loss_weight * F.smooth_l1_loss(
                pred_normalized,
                target_normalized,
                reduction="mean",
            )
        return loss

    def _recon_loss_disc_batch(self, m: int, pred_tokens: torch.Tensor, target_tokens: torch.Tensor):
        """Per-sample reconstruction loss for batched generator outputs."""
        pred_pool = pred_tokens.mean(dim=1)
        tgt_pool = target_tokens.mean(dim=1)
        z_pred = self.recon_projectors[m](pred_pool)
        z_tgt = self.recon_projectors[m](tgt_pool).detach()
        losses = 1.0 - F.cosine_similarity(z_pred, z_tgt, dim=-1)
        if self.recon_use_token_mse:
            token_mse = (pred_tokens - target_tokens).pow(2).mean(dim=(1, 2))
            losses = losses + self.recon_token_mse_weight * token_mse
        if self.recon_normalized_token_loss_weight > 0:
            pred_normalized = F.layer_norm(
                pred_tokens, (pred_tokens.shape[-1],)
            )
            target_normalized = F.layer_norm(
                target_tokens.detach(), (target_tokens.shape[-1],)
            )
            token_smooth_l1 = F.smooth_l1_loss(
                pred_normalized,
                target_normalized,
                reduction="none",
            ).mean(dim=(1, 2))
            losses = (
                losses
                + self.recon_normalized_token_loss_weight * token_smooth_l1
            )
        return losses

    def _generate_missing_batched(self, tokens_list, observed_mask, filled_for_cls, generated_mask):
        """Group equal missingness patterns so each generator runs on batches."""
        device = observed_mask.device
        for m in range(self.num_modalities):
            missing_idx = (~observed_mask[:, m]).nonzero(as_tuple=False).view(-1)
            if missing_idx.numel() == 0:
                continue

            pattern_rows = observed_mask.index_select(0, missing_idx).detach().cpu().tolist()
            groups = {}
            for position, pattern in enumerate(pattern_rows):
                context_modalities = tuple(j for j, is_observed in enumerate(pattern) if is_observed and j != m)
                if context_modalities:
                    groups.setdefault(context_modalities, []).append(position)

            for context_modalities, positions in groups.items():
                position_idx = torch.as_tensor(positions, device=device, dtype=torch.long)
                sample_idx = missing_idx.index_select(0, position_idx)
                context = torch.cat(
                    [tokens_list[j].index_select(0, sample_idx) for j in context_modalities],
                    dim=1,
                )
                if self.generator_only_task_grad:
                    context = context.detach()
                generated = self.generators[m](context)
                if not (
                    self.generator_task_grad or self.generator_only_task_grad
                ):
                    generated = generated.detach()
                filled_for_cls[m][sample_idx] = generated
                generated_mask[sample_idx, m] = True

    def _reconstruct_random_targets_batched(self, tokens_list, observed_mask):
        """Reconstruct a configurable number of targets per complete sample in batches."""
        device = observed_mask.device
        available_modalities = observed_mask.any(dim=0).nonzero(as_tuple=False).view(-1)
        if available_modalities.numel() == 0:
            return torch.zeros((), device=device)
        full_idx = observed_mask[:, available_modalities].all(dim=1).nonzero(as_tuple=False).view(-1)
        if full_idx.numel() == 0:
            return torch.zeros((), device=device)

        targets_per_sample = min(self.recon_targets_per_sample, int(available_modalities.numel()))
        random_order = torch.rand(
            full_idx.numel(), available_modalities.numel(), device=device
        ).argsort(dim=1)[:, :targets_per_sample]

        losses = []
        for available_position, modality_tensor in enumerate(available_modalities):
            selected_rows = (random_order == available_position).nonzero(as_tuple=False)[:, 0]
            if selected_rows.numel() == 0:
                continue
            sample_idx = full_idx.index_select(0, selected_rows)
            m = int(modality_tensor.item())
            context = torch.cat(
                [
                    self._scale_reconstruction_context_gradient(
                        tokens_list[j].index_select(0, sample_idx)
                    )
                    for j in range(self.num_modalities)
                    if j != m
                ],
                dim=1,
            )
            prediction = self.generators[m](context)
            target = tokens_list[m].index_select(0, sample_idx)
            losses.append(self._recon_loss_disc_batch(m, prediction, target))

        return torch.cat(losses).mean() if losses else torch.zeros((), device=device)

    def _reconstruct_observed_targets_batched(self, tokens_list, observed_mask):
        """Reconstruct naturally observed targets from per-sample contexts."""
        device = observed_mask.device
        groups = sample_observed_reconstruction_groups(
            observed_mask,
            self.recon_targets_per_sample,
            self.recon_context_dropout_probability,
            self.disable_stochastic_context_masking,
        )
        losses = []
        for (target_modality, context_modalities), sample_idx in groups.items():
            context = torch.cat(
                [
                    self._scale_reconstruction_context_gradient(
                        tokens_list[modality_index].index_select(0, sample_idx)
                    )
                    for modality_index in context_modalities
                ],
                dim=1,
            )
            prediction = self.generators[target_modality](context)
            target = tokens_list[target_modality].index_select(0, sample_idx)
            losses.append(
                self._recon_loss_disc_batch(
                    target_modality,
                    prediction,
                    target,
                )
            )
        return torch.cat(losses).mean() if losses else torch.zeros((), device=device)

    def forward(
        self,
        *modality_tokens,
        observed_mask: torch.Tensor,
        expert_indices: Optional[torch.Tensor] = None,
        return_importance: bool = False,
        return_recon_loss: bool = True,
        return_joint_ensemble_diagnostics: bool = False,
    ):
        assert len(modality_tokens) == self.num_modalities, "输入模态数与模型不一致"
        device = modality_tokens[0].device
        if (
            self.missing_family_normalized_router
            and observed_mask.dtype != torch.bool
        ):
            raise ValueError(
                "missing-family routing requires a raw boolean observed_mask"
            )
        observed_mask = observed_mask.bool()

        tokens_list = [t for t in modality_tokens]  # list[(B,P,D)]
        B = tokens_list[0].shape[0]

        # ===== 分类路径：缺失补全（detach）+ flag embedding =====
        filled_for_cls = [t.clone() for t in tokens_list]
        generated_mask = torch.zeros((B, self.num_modalities), device=device, dtype=torch.bool)

        gen_flag = [torch.ones((B,), device=device, dtype=torch.long) for _ in range(self.num_modalities)]
        for m in range(self.num_modalities):
            gen_flag[m][observed_mask[:, m]] = 0

        if self.use_generators and not self.disable_completion and self.vectorized_generation:
            self._generate_missing_batched(tokens_list, observed_mask, filled_for_cls, generated_mask)
        elif self.use_generators and not self.disable_completion:
            for m in range(self.num_modalities):
                missing_idx = (~observed_mask[:, m]).nonzero(as_tuple=False).view(-1)
                for idx in missing_idx.tolist():
                    per_sample_tokens = [t[idx] for t in tokens_list]
                    ctx = self._collect_context_tokens_general(per_sample_tokens, observed_mask[idx], m)
                    if ctx is None:
                        gen_flag[m][idx] = 1
                        continue
                    if self.generator_only_task_grad:
                        ctx = ctx.detach()
                    gen = self.generators[m](ctx.unsqueeze(0)).squeeze(0)
                    filled_for_cls[m][idx] = (
                        gen
                        if (
                            self.generator_task_grad
                            or self.generator_only_task_grad
                        )
                        else gen.detach()
                    )
                    gen_flag[m][idx] = 1
                    generated_mask[idx, m] = True

        for m in range(self.num_modalities):
            flagged = self.flag_embeds[m](filled_for_cls[m], gen_flag[m])
            if not self.disable_provenance:
                filled_for_cls[m] = flagged

        # ===== generator training: legacy complete-only or observed-only grouped targets =====
        recon_loss = None
        if return_recon_loss:
            if not self.use_generators or self.disable_completion:
                recon_loss = torch.zeros((), device=device)
            elif self.pattern_aware_reconstruction:
                recon_loss = self._reconstruct_observed_targets_batched(
                    tokens_list, observed_mask
                )
            elif self.vectorized_generation and self.recon_targets_per_sample > 0:
                recon_loss = self._reconstruct_random_targets_batched(tokens_list, observed_mask)
            else:
                avail_mask = observed_mask.any(dim=0)  # (M,) bool
                full_idx = (observed_mask[:, avail_mask].all(dim=1)).nonzero(as_tuple=False).view(-1)
                if full_idx.numel() == 0:
                    recon_loss = torch.tensor(0.0, device=device)
                else:
                    recon_losses = []
                    for idx in full_idx.tolist():
                        per_sample_tokens = [t[idx] for t in tokens_list]  # 用原始真实 token（不注入flag）
                        for m in range(self.num_modalities):
                            reconstruction_context_tokens = [
                                self._scale_reconstruction_context_gradient(token)
                                for token in per_sample_tokens
                            ]
                            ctx = self._collect_context_tokens_full(
                                reconstruction_context_tokens, m
                            )
                            pred = self.generators[m](ctx.unsqueeze(0)).squeeze(0)
                            target = per_sample_tokens[m]
                            recon_losses.append(self._recon_loss_disc(m, pred, target))
                    recon_loss = torch.stack(recon_losses).mean() if len(recon_losses) > 0 else torch.tensor(0.0, device=device)

        # ===== stable classification features + interpretation-only token attention =====
        token_features = self.backbone(*filled_for_cls, expert_indices=expert_indices, return_pooled=False)
        token_importance = []
        pooled_feats = []
        reliability_scores = []
        for m in range(self.num_modalities):
            pooled_m, tok_w_m = self.token_poolers[m](token_features[m])
            token_importance.append(tok_w_m)
            pooled_feats.append(pooled_m)
            raw_reliability = self.reliability_scorers[m](pooled_m).squeeze(-1)
            gen_rel = torch.sigmoid(
                raw_reliability + self.generated_reliability_bias[m]
            )
            observed_rel = (
                0.75 + 0.5 * torch.sigmoid(raw_reliability)
                if self.learn_observed_reliability
                else torch.ones_like(gen_rel)
            )
            rel_m = torch.where(
                observed_mask[:, m],
                observed_rel,
                torch.where(generated_mask[:, m], gen_rel, torch.zeros_like(gen_rel)),
            )
            reliability_scores.append(rel_m)

        modality_reliability = torch.stack(reliability_scores, dim=1)
        usable_mask = observed_mask | generated_mask
        no_usable = ~usable_mask.any(dim=1)
        if no_usable.any():
            usable_mask[no_usable] = True
            modality_reliability[no_usable] = 1.0 / float(self.num_modalities)

        (
            logits,
            branch_logits,
            branch_weights,
            branch_mask,
            branch_quality,
            branch_log_scores,
            supervision_log_scores,
        ) = self.branch_fusion(
            pooled_feats,
            usable_mask,
            modality_reliability,
            complete_mask=observed_mask.all(dim=1),
            observed_mask=observed_mask,
        )
        supervision_branch_mask = (
            self.branch_fusion.last_supervision_branch_mask
        )
        ordinal_logits = None
        class1_aux_logit = None
        supervised_contrastive_embedding = None
        flat_features = None
        if self.ordinal_head is not None:
            flat_features = torch.cat(pooled_feats, dim=1)
            ordinal_logits, ordinal_probs = self.ordinal_head(flat_features)
            branch_probs = torch.softmax(logits, dim=1)
            blend = min(max(self.ordinal_fusion_weight, 0.0), 1.0)
            if self.uncertainty_aware_ordinal_fusion:
                mixed_probs, _ = uncertainty_weighted_ordinal_mix(
                    logits, ordinal_probs, blend
                )
                logits = torch.log(mixed_probs.clamp_min(1e-8))
            else:
                logits = torch.log(
                    (
                        (1.0 - blend) * branch_probs
                        + blend * ordinal_probs
                    ).clamp_min(1e-8)
                )
        base_logits = None
        tail_logits = None
        if self.more_tail_rank > 0:
            if flat_features is None:
                flat_features = torch.cat(pooled_feats, dim=1)
            base_logits = logits
            tail_logits = self.more_tail_B(self.more_tail_A(flat_features))
            logits = base_logits + tail_logits
        dlbr_base_logits = None
        dlbr_residuals = None
        if self.dual_local_boundary_loss_weight > 0:
            if flat_features is None:
                flat_features = torch.cat(pooled_feats, dim=1)
            dlbr_base_logits = logits
            normalized_features = self.dual_local_boundary_norm(flat_features)
            dlbr_residuals = 0.5 * torch.tanh(
                torch.cat(
                    tuple(
                        head(normalized_features)
                        for head in self.dual_local_boundary_heads
                    ),
                    dim=1,
                )
            )
            residual_10 = dlbr_residuals[:, 0]
            residual_12 = dlbr_residuals[:, 1]
            # This is the unique minimum-L2, zero-sum three-logit correction
            # whose two class-1 gaps change by residual_10 and residual_12.
            dlbr_delta = torch.stack(
                (
                    (-2.0 * residual_10 + residual_12) / 3.0,
                    (residual_10 + residual_12) / 3.0,
                    (residual_10 - 2.0 * residual_12) / 3.0,
                ),
                dim=1,
            )
            logits = dlbr_base_logits + dlbr_delta
        presentation_axis_base_logits = None
        presentation_axis_raw_logits = None
        presentation_axis_residuals = None
        if self.presentation_axis_loss_weight > 0:
            if flat_features is None:
                flat_features = torch.cat(pooled_feats, dim=1)
            presentation_axis_base_logits = logits
            presentation_axis_raw_logits = self.presentation_axis_head(
                self.presentation_axis_norm(flat_features)
            )
            presentation_axis_residuals = (
                self.presentation_axis_residual_cap
                * torch.tanh(presentation_axis_raw_logits)
            )
            residual_ia = presentation_axis_residuals[:, 0]
            residual_hi = presentation_axis_residuals[:, 1]
            # IA evidence raises the inattentive and combined presentations
            # relative to hyperactive/impulsive; HI evidence symmetrically
            # raises hyperactive/impulsive and combined relative to inattentive.
            # The correction is zero-sum for every sample.
            presentation_axis_delta = torch.stack(
                (
                    (residual_ia - 2.0 * residual_hi) / 3.0,
                    (-2.0 * residual_ia + residual_hi) / 3.0,
                    (residual_ia + residual_hi) / 3.0,
                ),
                dim=1,
            )
            logits = presentation_axis_base_logits + presentation_axis_delta
        missing_capacity_base_logits = None
        missing_capacity_raw_delta = None
        missing_capacity_delta = None
        current_view_missing_mask = None
        if self.missing_capacity_residual_width > 0:
            if flat_features is None:
                flat_features = torch.cat(pooled_feats, dim=1)
            missing_capacity_base_logits = logits
            missing_capacity_raw_delta = self.missing_capacity_residual(
                flat_features
            )
            current_view_missing_mask = ~observed_mask.all(dim=1)
            missing_capacity_delta = torch.where(
                current_view_missing_mask.unsqueeze(1),
                missing_capacity_raw_delta,
                torch.zeros_like(missing_capacity_raw_delta),
            )
            logits = missing_capacity_base_logits + missing_capacity_delta
        if self.class1_aux_head is not None:
            if flat_features is None:
                flat_features = torch.cat(pooled_feats, dim=1)
            class1_aux_logit = self.class1_aux_head(flat_features).squeeze(-1)
        if self.supervised_contrastive_projector is not None:
            if flat_features is None:
                flat_features = torch.cat(pooled_feats, dim=1)
            supervised_contrastive_embedding = F.normalize(
                self.supervised_contrastive_projector(flat_features),
                p=2,
                dim=1,
            )
        token_importance = torch.stack(token_importance, dim=1) # (B,M,P)

        # Convert branch reliabilities into modality weights for normalized token-level explanations.
        modality_w = branch_weights[:, 1:1 + self.num_modalities].clone()
        pair_offset = 1 + self.num_modalities
        for pair_idx, (i, j) in enumerate(self.branch_fusion.pairs):
            pair_w = 0.5 * branch_weights[:, pair_offset + pair_idx]
            modality_w[:, i] = modality_w[:, i] + pair_w
            modality_w[:, j] = modality_w[:, j] + pair_w
        joint_share = branch_weights[:, 0:1] / float(self.num_modalities)
        modality_w = modality_w + joint_share
        modality_w = modality_w.masked_fill(~usable_mask, 0.0)
        modality_w = modality_w / modality_w.sum(dim=1, keepdim=True).clamp_min(1e-8)

        w_entropy = -(modality_w.clamp(min=1e-8) * torch.log(modality_w.clamp(min=1e-8))).sum(dim=1).mean()

        aux_loss = torch.tensor(0.0, device=device)

        importance = None
        if return_importance:
            importance = modality_w.unsqueeze(-1) * token_importance

        output = {
            "logits": logits,
            "branch_logits": branch_logits,
            "branch_weights": branch_weights,
            "branch_log_scores": branch_log_scores,
            "supervision_log_scores": supervision_log_scores,
            "branch_mask": branch_mask,
            "supervision_branch_mask": supervision_branch_mask,
            "branch_quality": branch_quality,
            "ordinal_logits": ordinal_logits,
            "class1_aux_logit": class1_aux_logit,
            "supervised_contrastive_embedding": (
                supervised_contrastive_embedding
            ),
            "modality_reliability": modality_reliability,
            "generated_mask": generated_mask,
            "usable_mask": usable_mask,
            "w": modality_w,
            "token_importance": token_importance,
            "w_entropy": w_entropy,
            "recon_loss": recon_loss,
            "importance": importance,
            "aux_loss": aux_loss,
        }
        if self.more_tail_rank > 0:
            output.update(
                {
                    "base_logits": base_logits,
                    "tail_logits": tail_logits,
                }
            )
        if self.dual_local_boundary_loss_weight > 0:
            output.update(
                {
                    "dlbr_base_logits": dlbr_base_logits,
                    "dlbr_residuals": dlbr_residuals,
                }
            )
        if self.presentation_axis_loss_weight > 0:
            output.update(
                {
                    "presentation_axis_base_logits": (
                        presentation_axis_base_logits
                    ),
                    "presentation_axis_raw_logits": (
                        presentation_axis_raw_logits
                    ),
                    "presentation_axis_residuals": (
                        presentation_axis_residuals
                    ),
                }
            )
        if self.missing_capacity_residual_width > 0:
            output.update(
                {
                    "missing_capacity_base_logits": (
                        missing_capacity_base_logits
                    ),
                    "missing_capacity_raw_delta": missing_capacity_raw_delta,
                    "missing_capacity_delta": missing_capacity_delta,
                    "current_view_missing_mask": current_view_missing_mask,
                }
            )
        if return_joint_ensemble_diagnostics:
            output.update(
                {
                    "joint_member_logits": (
                        self.branch_fusion.last_joint_member_logits
                    ),
                    "joint_member_probabilities": (
                        self.branch_fusion.last_joint_member_probabilities
                    ),
                    "joint_member_evidence": (
                        self.branch_fusion.last_joint_member_evidence
                    ),
                    "joint_probability": (
                        self.branch_fusion.last_joint_probability
                    ),
                    "joint_log_probability": (
                        self.branch_fusion.last_joint_log_probability
                    ),
                }
            )
        return output
