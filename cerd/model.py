import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .moe import *
from itertools import combinations
from typing import Optional, List, Tuple


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

            if self.mlp_sparse:
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i]), self.expert_index))
            else:
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i])))

        else:
            chunk_size = [item.shape[1] for item in x]
            x = [item for item in x]
            for i in range(len(chunk_size)):
                other_m = [x[j] for j in range(len(chunk_size)) if j != i]
                other_m = torch.cat([x[i], *other_m], dim=1)
                x[i] = self.attn(x[i], other_m, attn_mask)
            x = [x[i] + self.dropout1(x[i]) for i in range(len(chunk_size))]

            if self.mlp_sparse:
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i]), self.expert_index))
            else:
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i])))

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
                 normalized_gate_loss=False):
        super().__init__()
        layers = []
        _sparse = True
        layers.append(TransformerEncoderLayer(num_experts, num_routers, hidden_dim, num_head=num_heads,
                                              dropout=dropout, hidden_times=2, mlp_sparse=_sparse,
                                              full_modality_index=full_modality_index, top_k=top_k,
                                              standard_residual=standard_residual,
                                              gated_residual=gated_residual,
                                              normalized_gate_loss=normalized_gate_loss))
        for _ in range(num_layers - 1):
            _sparse = not _sparse
            layers.append(TransformerEncoderLayer(num_experts, num_routers, hidden_dim, num_head=num_heads,
                                                  dropout=dropout, hidden_times=2, mlp_sparse=_sparse,
                                                  full_modality_index=full_modality_index, top_k=top_k,
                                                  standard_residual=standard_residual,
                                                  gated_residual=gated_residual,
                                                  normalized_gate_loss=normalized_gate_loss))
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
    def __init__(self, hidden_dim: int, num_patches: int, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
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

        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

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
    Uses modality-local encoding before the shared fusion stack.
    """
    def __init__(
        self,
        hidden_dim: int,
        temperature: float = 0.5,
        dropout: float = 0.1,
        attention_mix_init: float = -4.0,
    ):
        super().__init__()
        self.temperature = temperature
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
        pooled = mean_pooled + mix * (attn_pooled - mean_pooled)
        return pooled, weights


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
        complete_joint_only: bool = False,
        complete_specialist_weight: float = 1.0,
        confidence_mode: str = "evidence",
        centered_evidence_confidence: bool = False,
        class_conditional_fusion: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        self.output_dim = output_dim
        self.pairs = list(combinations(range(num_modalities), 2))
        self.num_branches = 1 + num_modalities + len(self.pairs)
        self.dynamic_gating = bool(dynamic_gating)
        self.complete_joint_only = bool(complete_joint_only)
        if not 0.0 <= complete_specialist_weight <= 1.0:
            raise ValueError("complete_specialist_weight must be in [0, 1]")
        self.complete_specialist_weight = float(complete_specialist_weight)
        if confidence_mode not in {"evidence", "entropy_detached", "entropy_exp_detached"}:
            raise ValueError(f"Unsupported branch confidence mode: {confidence_mode}")
        self.confidence_mode = confidence_mode
        self.centered_evidence_confidence = bool(centered_evidence_confidence)
        self.class_conditional_fusion = bool(class_conditional_fusion)
        self.joint_head = MLP(
            hidden_dim * num_modalities,
            hidden_dim,
            output_dim,
            num_layers_pred,
            activation=nn.ReLU(),
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
            self.dynamic_gate = nn.Sequential(
                nn.LayerNorm(gate_input_dim),
                nn.Linear(gate_input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(min(dropout, 0.2)),
                nn.Linear(hidden_dim, self.num_branches),
            )
            # Start close to the old reliability fusion and learn a bounded,
            # sample-specific correction rather than replacing it abruptly.
            nn.init.zeros_(self.dynamic_gate[-1].weight)
            nn.init.zeros_(self.dynamic_gate[-1].bias)
            self.dynamic_gate_scale = nn.Parameter(torch.tensor(0.5))
            with torch.no_grad():
                self.branch_prior[0] = math.log(2.0)

        # Non-persistent, graph-carrying diagnostics for the optional ensemble.
        # They intentionally are not parameters or buffers, so K=1 state dicts
        # and checkpoints remain byte-for-byte structurally compatible.
        self.last_joint_member_logits = None
        self.last_joint_member_probabilities = None
        self.last_joint_member_evidence = None
        self.last_joint_probability = None
        self.last_joint_log_probability = None

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

    def forward(
        self,
        modality_feats: List[torch.Tensor],
        usable_mask: torch.Tensor,
        modality_reliability: torch.Tensor,
        complete_mask: Optional[torch.Tensor] = None,
    ):
        feats = torch.stack(modality_feats, dim=1)  # (B,M,D)
        flat = feats.flatten(1)

        joint_logits = self.joint_head(flat)

        branch_logits = [joint_logits]
        for m in range(self.num_modalities):
            branch_logits.append(self.unimodal_heads[m](feats[:, m, :]))
        for head, (i, j) in zip(self.pair_heads, self.pairs):
            fi, fj = feats[:, i, :], feats[:, j, :]
            pair_feat = torch.cat([fi, fj, fi * fj, torch.abs(fi - fj)], dim=-1)
            branch_logits.append(head(pair_feat))
        branch_logits = torch.stack(branch_logits, dim=1)  # (B,branches,C)

        branch_mask = self._branch_mask(usable_mask)
        branch_quality = self._branch_quality(modality_reliability, branch_mask)

        branch_probs = F.softmax(branch_logits, dim=-1)
        if self.confidence_mode in {"entropy_detached", "entropy_exp_detached"}:
            entropy = -(branch_probs.clamp_min(1e-8) * branch_probs.clamp_min(1e-8).log()).sum(dim=-1)
            if self.confidence_mode == "entropy_exp_detached":
                branch_confidence = torch.exp(-entropy).detach()
            else:
                branch_confidence = (
                    1.0 - entropy / math.log(float(self.output_dim))
                ).clamp_min(1e-3).detach()
        else:
            branch_confidence = self._evidence_confidence(branch_logits)
        base_branch_log_scores = (
            torch.log(branch_quality.clamp_min(1e-8))
            + torch.log(branch_confidence.clamp_min(1e-8))
            + self.branch_prior.unsqueeze(0)
        )
        if self.dynamic_gating:
            gate_input = torch.cat(
                [flat, usable_mask.float(), modality_reliability], dim=1
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
            ~branch_mask, -1e4
        )
        branch_weights = F.softmax(branch_log_scores, dim=1)

        probs = self._mix_probabilities(branch_probs, branch_weights)
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
) -> dict[tuple[int, tuple[int, ...]], torch.Tensor]:
    """Group per-sample observed reconstruction tasks by target and context.

    Every eligible sample has at least two naturally observed modalities.  A
    target is sampled only from those observed modalities, and all remaining
    observed modalities form a non-empty conditioning context.  Grouping equal
    ``(target, context-pattern)`` tasks preserves batched generator execution
    without making eligibility depend on the other samples in the minibatch.
    """
    if observed_mask.ndim != 2:
        raise ValueError("observed_mask must have shape (batch, modalities)")
    targets_per_sample = int(targets_per_sample)
    if targets_per_sample < 0:
        raise ValueError("targets_per_sample must be non-negative")
    if targets_per_sample == 0 or observed_mask.shape[0] == 0:
        return {}

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
    grouped_rows: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for sample_index, pattern in enumerate(patterns):
        observed_modalities = [
            modality_index
            for modality_index, is_observed in enumerate(pattern)
            if is_observed
        ]
        if len(observed_modalities) < 2:
            continue
        target_count = min(targets_per_sample, len(observed_modalities))
        if target_count == len(observed_modalities):
            target_modalities = observed_modalities
        else:
            if priorities is None:
                raise RuntimeError("Target priorities were not initialized")
            target_modalities = sorted(
                observed_modalities,
                key=lambda modality_index: priorities[sample_index][modality_index],
            )[:target_count]
        for target_modality in target_modalities:
            context_modalities = tuple(
                modality_index
                for modality_index in observed_modalities
                if modality_index != target_modality
            )
            # Eligibility above guarantees a real observed context.
            grouped_rows.setdefault(
                (target_modality, context_modalities), []
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
        vectorized_generation: bool = False,
        recon_targets_per_sample: int = 0,
        use_generators: bool = True,
        dynamic_branch_fusion: bool = False,
        complete_joint_only: bool = False,
        complete_specialist_weight: float = 1.0,
        branch_confidence_mode: str = "evidence",
        token_attention_init: float = -4.0,
        generator_task_grad: bool = False,
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

        # Optional MORE-inspired low-rank adaptation of the final fused logits.
        # Zero is deliberately the default so historical construction consumes
        # no additional RNG and creates no additional state-dict entries.
        more_tail_rank: int = 0,

        # Optional dual local-boundary residual (DLBR) on the final fused
        # three-class logits.  A zero loss weight constructs no module and
        # therefore preserves the historical module tree, RNG trajectory, and
        # state dict exactly.
        dual_local_boundary_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim

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
        self.vectorized_generation = vectorized_generation
        self.recon_targets_per_sample = max(0, int(recon_targets_per_sample))
        self.use_generators = bool(use_generators)
        self.generator_task_grad = bool(generator_task_grad)
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
        )

        self.generators = nn.ModuleList([
            ConditionalGenerator(hidden_dim, num_patches, num_heads=gen_num_heads, num_layers=gen_num_layers, dropout=0.1)
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
            complete_joint_only=complete_joint_only,
            complete_specialist_weight=complete_specialist_weight,
            confidence_mode=branch_confidence_mode,
            centered_evidence_confidence=centered_evidence_confidence,
            class_conditional_fusion=class_conditional_fusion,
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
                generated = self.generators[m](context)
                if not self.generator_task_grad:
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
                [tokens_list[j].index_select(0, sample_idx) for j in range(self.num_modalities) if j != m],
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
        )
        losses = []
        for (target_modality, context_modalities), sample_idx in groups.items():
            context = torch.cat(
                [
                    tokens_list[modality_index].index_select(0, sample_idx)
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
        observed_mask = observed_mask.bool()

        tokens_list = [t for t in modality_tokens]  # list[(B,P,D)]
        B = tokens_list[0].shape[0]

        # ===== 分类路径：缺失补全（detach）+ flag embedding =====
        filled_for_cls = [t.clone() for t in tokens_list]
        generated_mask = torch.zeros((B, self.num_modalities), device=device, dtype=torch.bool)

        gen_flag = [torch.ones((B,), device=device, dtype=torch.long) for _ in range(self.num_modalities)]
        for m in range(self.num_modalities):
            gen_flag[m][observed_mask[:, m]] = 0

        if self.use_generators and self.vectorized_generation:
            self._generate_missing_batched(tokens_list, observed_mask, filled_for_cls, generated_mask)
        elif self.use_generators:
            for m in range(self.num_modalities):
                missing_idx = (~observed_mask[:, m]).nonzero(as_tuple=False).view(-1)
                for idx in missing_idx.tolist():
                    per_sample_tokens = [t[idx] for t in tokens_list]
                    ctx = self._collect_context_tokens_general(per_sample_tokens, observed_mask[idx], m)
                    if ctx is None:
                        gen_flag[m][idx] = 1
                        continue
                    gen = self.generators[m](ctx.unsqueeze(0)).squeeze(0)
                    filled_for_cls[m][idx] = gen if self.generator_task_grad else gen.detach()
                    gen_flag[m][idx] = 1
                    generated_mask[idx, m] = True

        for m in range(self.num_modalities):
            filled_for_cls[m] = self.flag_embeds[m](filled_for_cls[m], gen_flag[m])

        # ===== generator training: legacy complete-only or observed-only grouped targets =====
        recon_loss = None
        if return_recon_loss:
            if not self.use_generators:
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
                            ctx = self._collect_context_tokens_full(per_sample_tokens, m)
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
