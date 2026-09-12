"""Thin adapter around the official I2MoE interaction architecture.

The interaction experts, random-modality perturbations, interaction losses,
and MLP reweighting are imported from the local official I2MoE checkout.  This
module only adapts the shared runner's token/missingness interface to that
implementation and converts its tuple output to the runner's dictionary API.

Official source pinned in this workspace:
  repository: https://github.com/Raina-Xin/I2MoE
  commit: 75b578e1f7ec20ebe990f6e6e5680b1a8860046f
  interaction implementation: src/imoe/InteractionMoE.py
  fusion implementation: src/common/fusion_models/transformer.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


OFFICIAL_ROOT = Path(
    os.environ.get(
        "I2MOE_OFFICIAL_ROOT",
        str(Path(__file__).resolve().parent.parent / "I2MoE"),
    )
).expanduser().resolve()
OFFICIAL_COMMIT = "75b578e1f7ec20ebe990f6e6e5680b1a8860046f"
OFFICIAL_REPOSITORY = "https://github.com/Raina-Xin/I2MoE"


def _load_official_classes():
    """Import the pinned local official classes without copying their logic."""

    required = (
        OFFICIAL_ROOT / "src" / "imoe" / "InteractionMoE.py",
        OFFICIAL_ROOT / "src" / "common" / "fusion_models" / "transformer.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The official I2MoE checkout is required by this adapter; missing: "
            + ", ".join(missing)
        )

    official_root = str(OFFICIAL_ROOT)
    if official_root not in sys.path:
        # The upstream package uses absolute imports rooted at ``src``.
        sys.path.insert(0, official_root)

    try:
        from src.common.fusion_models.transformer import Transformer
        from src.imoe.InteractionMoE import InteractionMoE
    except Exception as error:  # pragma: no cover - preserves the root cause
        raise ImportError(
            f"Could not import the official I2MoE implementation from {OFFICIAL_ROOT}"
        ) from error

    imported_sources = {
        Path(sys.modules[Transformer.__module__].__file__).resolve(),
        Path(sys.modules[InteractionMoE.__module__].__file__).resolve(),
    }
    expected_sources = {path.resolve() for path in required}
    if imported_sources != expected_sources:
        raise ImportError(
            "A conflicting top-level 'src' package shadowed the official I2MoE checkout: "
            f"loaded {sorted(map(str, imported_sources))}"
        )
    return Transformer, InteractionMoE


OfficialTransformer, OfficialInteractionMoE = _load_official_classes()


class I2MoEOfficialAdapter(nn.Module):
    """Official dense-Transformer I2MoE with the shared runner interface.

    Each of the ``M + 2`` branches is an independent deep copy of the official
    fusion network: ``M`` uniqueness experts, one synergy expert, and one
    redundancy expert.  Training calls the upstream perturbation/loss path;
    evaluation calls its unperturbed ``inference`` path.
    """

    def __init__(
        self,
        *,
        num_modalities: int,
        num_patches: int,
        hidden_dim: int,
        output_dim: int,
        num_layers_fus: int,
        num_layers_pred: int,
        num_experts: int,
        num_routers: int,
        top_k: int,
        num_heads: int,
        dropout: float,
        hidden_dim_rw: int,
        num_layer_rw: int,
        temperature_rw: float,
    ) -> None:
        super().__init__()
        if num_modalities < 2:
            raise ValueError("I2MoE requires at least two modalities")
        self.num_modalities = int(num_modalities)
        self.num_branches = self.num_modalities + 2

        # This is the exact fusion class used by the official ADNI Transformer
        # experiment.  Its official script sets fusion_sparse=False, so the
        # num_experts/router arguments are retained for constructor parity but
        # intentionally do not enable a nested sparse-MoE layer.
        fusion_model = OfficialTransformer(
            num_modalities=self.num_modalities,
            num_patches=num_patches,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers_fus,
            num_layers_pred=num_layers_pred,
            num_experts=num_experts,
            num_routers=num_routers,
            top_k=top_k,
            num_heads=num_heads,
            dropout=dropout,
            mlp_sparse=False,
            gate="None",
        )
        self.interaction_moe = OfficialInteractionMoE(
            num_modalities=self.num_modalities,
            fusion_model=fusion_model,
            fusion_sparse=False,
            hidden_dim=hidden_dim,
            hidden_dim_rw=hidden_dim_rw,
            num_layer_rw=num_layer_rw,
            temperature_rw=temperature_rw,
        )

    @staticmethod
    def provenance() -> dict[str, Any]:
        return {
            "implementation": "official-code adapter",
            "official_repository": OFFICIAL_REPOSITORY,
            "official_commit": OFFICIAL_COMMIT,
            "official_sources": [
                "I2MoE/src/imoe/InteractionMoE.py",
                "I2MoE/src/common/fusion_models/transformer.py",
            ],
            "preserved_core": [
                "M uniqueness + 1 synergy + 1 redundancy independent interaction experts",
                "official random-modality replacement forward passes",
                "official triplet/cosine interaction losses",
                "official sample-wise MLP softmax reweighting",
            ],
            "adapter_boundary": [
                "shared MoE encoders/data/splits/metrics replace the official data pipeline",
                "naturally missing modalities enter as zero tokens from the shared encoder path",
                "official dense Transformer fusion is used (fusion_sparse=False)",
                "the shared runner owns optimizer, checkpoint selection, and class balancing",
            ],
        }

    def _adapt_inputs(
        self,
        tokens: tuple[torch.Tensor, ...],
        observed_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        if len(tokens) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modality tensors, got {len(tokens)}"
            )
        if observed_mask.ndim != 2 or observed_mask.shape[1] != self.num_modalities:
            raise ValueError(
                "observed_mask must have shape (batch, num_modalities); got "
                f"{tuple(observed_mask.shape)}"
            )
        if any(token.ndim != 3 for token in tokens):
            raise ValueError("Each I2MoE input must have shape (batch, patches, hidden_dim)")

        # ``encode_batch`` already emits zero tokens for missing inputs.  Apply
        # the mask again here so the adapter contract remains explicit and no
        # caller can leak a missing modality's value into the official model.
        return [
            token.masked_fill(~observed_mask[:, index, None, None], 0.0)
            for index, token in enumerate(tokens)
        ]

    def forward(
        self,
        *tokens: torch.Tensor,
        observed_mask: torch.Tensor,
        return_aux: bool = False,
    ) -> dict[str, torch.Tensor]:
        inputs = self._adapt_inputs(tokens, observed_mask.bool())

        if return_aux:
            (
                expert_outputs,
                interaction_weights,
                weighted_logits,
                interaction_losses,
            ) = self.interaction_moe(inputs)
            per_interaction_loss = torch.stack(
                [loss.reshape(()) for loss in interaction_losses]
            )
            return {
                "logits": weighted_logits,
                "interaction_loss": per_interaction_loss.mean(),
                "interaction_losses": per_interaction_loss,
                "interaction_weights": interaction_weights,
                "branch_logits": torch.stack(
                    [outputs[0] for outputs in expert_outputs], dim=1
                ),
            }

        branch_logits, interaction_weights, weighted_logits = (
            self.interaction_moe.inference(inputs)
        )
        return {
            "logits": weighted_logits,
            "interaction_weights": interaction_weights,
            "branch_logits": torch.stack(branch_logits, dim=1),
        }
