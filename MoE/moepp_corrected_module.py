"""MoE++ adapter with an isolated correction for an upstream residual bug.

The vendored I2MoE implementation calls its MoE feed-forward block, but then
immediately overwrites the returned expert features with a second copy of the
attention output.  Consequently the classifier has no computational path to
any expert or router parameter.  This adapter leaves the pinned upstream files
untouched and replaces only that encoder ``forward`` method with the standard
pre-norm residual update::

    residual + dropout2(moe_output)

The unmodified adapter remains available as ``moepp``.  The corrected variant
is deliberately exposed as a different model name so results cannot be mixed.
"""

from __future__ import annotations

from types import MethodType
from typing import Any

from copy import deepcopy

import torch

from moepp_module import MoEPlusPlusOfficialAdapter


def _corrected_moe_layer_forward(self, *inputs, gate_residual=None, **kwargs):
    """Preserve upstream routing while avoiding its autograd-breaking in-place scale."""

    del kwargs
    d_model = inputs[0].shape[-1]
    reshaped_input = inputs[0].reshape(-1, d_model)
    output = torch.zeros_like(reshaped_input)
    expert_info, gate_residual = self.gate(reshaped_input, gate_residual)
    if not (self.moe_use_mixtral_gating or self.moe_feature_no_mul_topk):
        routed_input = reshaped_input * 2
    else:
        routed_input = reshaped_input
    for expert_index, token_indices_and_gates in expert_info.items():
        indices, gates = token_indices_and_gates
        tokens = routed_input.index_select(dim=0, index=indices)
        expert_output = self.experts.experts[expert_index](tokens)
        weighted_output = expert_output * gates.unsqueeze(-1)
        output.index_add_(dim=0, index=indices, source=weighted_output)
    return output.reshape(inputs[0].shape), gate_residual


def _corrected_encoder_forward(self, x, gate_residual=None):
    """Upstream forward with only the discarded-MoE-output line corrected."""

    chunk_size = [item.shape[1] for item in x]

    residual = torch.cat(x, dim=1)
    normed_x = self.norm1(residual)
    attended_x = self.attn(normed_x, normed_x)
    hidden_states = residual + self.dropout1(attended_x)

    residual = hidden_states
    moe_output, gate_residual = self.mlp(
        self.norm2(hidden_states), gate_residual=gate_residual
    )
    hidden_states = residual + self.dropout2(moe_output)

    return list(torch.split(hidden_states, chunk_size, dim=1)), gate_residual


def _corrected_fusion_forward(self, inputs, return_latent=False):
    """Run encoder layers sequentially; upstream repeatedly feeds the raw input."""

    chunk_size = [item.shape[1] for item in inputs]
    hidden_states = torch.cat(inputs, dim=1)
    if self.pos_embed is not None:
        hidden_states = hidden_states + self.pos_embed
    layer_inputs = list(torch.split(hidden_states, chunk_size, dim=1))

    gate_residual = None
    for layer in self.layers:
        layer_inputs, gate_residual = layer(
            layer_inputs, gate_residual=gate_residual
        )
    pooled = torch.cat([item.mean(dim=1) for item in layer_inputs], dim=1)
    latent = deepcopy(pooled) if return_latent else None
    logits = self.classification_head(pooled)
    return (logits, latent) if return_latent else logits


class MoEPlusPlusCorrectedAdapter(MoEPlusPlusOfficialAdapter):
    """Official MoE++ adapter with the one-line feed-forward residual fix."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fusion_model.forward = MethodType(
            _corrected_fusion_forward, self.fusion_model
        )
        for layer in self.fusion_model.layers:
            layer.forward = MethodType(_corrected_encoder_forward, layer)
            layer.mlp.moe.forward = MethodType(
                _corrected_moe_layer_forward, layer.mlp.moe
            )

    @staticmethod
    def provenance() -> dict[str, Any]:
        base = MoEPlusPlusOfficialAdapter.provenance()
        return {
            **base,
            "implementation": "official-code adapter with isolated upstream bug correction",
            "upstream_bug": (
                "I2MoE/src/common/modules/moepp_layer.py discards the MoE output "
                "and adds attended_x twice; fusion_models/moepp.py also feeds the "
                "raw input to every encoder layer instead of chaining layer outputs"
            ),
            "isolated_patch": (
                "second residual update is residual + dropout2(moe_output); "
                "the upstream in-place routed-input scale is made out-of-place "
                "to preserve autograd; encoder outputs are passed sequentially; "
                "vendored upstream source is unchanged"
            ),
        }
