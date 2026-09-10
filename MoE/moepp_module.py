"""Thin adapter around the official I2MoE repository's MoE++ baseline.

The fusion network, residual gate routing, learned/constant/copy/zero experts,
and classifier are imported from the pinned local I2MoE checkout.  This file
only adapts the shared runner's token/missingness interface and defers device
placement until the runner calls ``.to(device)``.

Official source pinned in this workspace:
  repository: https://github.com/Raina-Xin/I2MoE
  commit: 75b578e1f7ec20ebe990f6e6e5680b1a8860046f
  implementation: src/common/fusion_models/moepp.py
  MoE layer: src/common/modules/moepp_layer.py
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn


OFFICIAL_ROOT = Path(__file__).resolve().parent.parent / "I2MoE"
OFFICIAL_COMMIT = "75b578e1f7ec20ebe990f6e6e5680b1a8860046f"
OFFICIAL_REPOSITORY = "https://github.com/Raina-Xin/I2MoE"


def _load_official_class():
    """Import the exact pinned MoE++ class rather than copying its logic."""

    required = (
        OFFICIAL_ROOT / "src" / "common" / "fusion_models" / "moepp.py",
        OFFICIAL_ROOT / "src" / "common" / "modules" / "moepp_layer.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The official I2MoE checkout is required by this adapter; missing: "
            + ", ".join(missing)
        )

    official_root = str(OFFICIAL_ROOT)
    if official_root not in sys.path:
        # Upstream uses absolute imports rooted at the top-level ``src``.
        sys.path.insert(0, official_root)

    try:
        from src.common.fusion_models.moepp import MoEPlusPlusTransformer
        from src.common.modules.moepp_layer import MoEPlusPlusEncoderLayer
    except Exception as error:  # pragma: no cover - preserves the root cause
        raise ImportError(
            f"Could not import the official MoE++ implementation from {OFFICIAL_ROOT}"
        ) from error

    imported_sources = {
        Path(sys.modules[MoEPlusPlusTransformer.__module__].__file__).resolve(),
        Path(sys.modules[MoEPlusPlusEncoderLayer.__module__].__file__).resolve(),
    }
    expected_sources = {path.resolve() for path in required}
    if imported_sources != expected_sources:
        raise ImportError(
            "A conflicting top-level 'src' package shadowed the official I2MoE checkout: "
            f"loaded {sorted(map(str, imported_sources))}"
        )
    return MoEPlusPlusTransformer


OfficialMoEPlusPlusTransformer = _load_official_class()


@contextmanager
def _defer_upstream_cuda_placement() -> Iterator[None]:
    """Neutralize one eager ``.cuda()`` call in the upstream constructor.

    The official class moves only its classifier to CUDA while constructing
    the model.  That eager placement is unnecessary because both the official
    trainer and the shared runner subsequently call ``model.to(device)``.  A
    short-lived constructor shim keeps the vendored source untouched, permits
    CPU verification, and has no effect on model parameters or computation.
    """

    original_cuda = nn.Module.cuda

    def identity_cuda(module: nn.Module, device: Any = None) -> nn.Module:
        del device
        return module

    nn.Module.cuda = identity_cuda
    try:
        yield
    finally:
        nn.Module.cuda = original_cuda


class MoEPlusPlusOfficialAdapter(nn.Module):
    """Official MoE++ fusion baseline with the shared runner interface."""

    def __init__(
        self,
        *,
        num_modalities: int,
        num_patches: int,
        hidden_dim: int,
        output_dim: int,
        num_layers_fus: int,
        num_experts: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_modalities < 2:
            raise ValueError("MoE++ requires at least two modalities")
        if num_experts < 4:
            raise ValueError(
                "The official MoE++ expert set needs at least four experts "
                "(two constant, one copy, and one zero expert)"
            )
        self.num_modalities = int(num_modalities)

        with _defer_upstream_cuda_placement():
            self.fusion_model = OfficialMoEPlusPlusTransformer(
                num_modalities=self.num_modalities,
                num_patches=num_patches,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers_fus,
                num_experts=num_experts,
                num_heads=num_heads,
                dropout=dropout,
            )

    @staticmethod
    def provenance() -> dict[str, Any]:
        return {
            "implementation": "official-code adapter",
            "official_repository": OFFICIAL_REPOSITORY,
            "official_commit": OFFICIAL_COMMIT,
            "official_sources": [
                "I2MoE/src/common/fusion_models/moepp.py",
                "I2MoE/src/common/modules/moepp_layer.py",
            ],
            "preserved_core": [
                "official residual gate routing across MoE++ encoder layers",
                "official learned SwiGLU, constant, copy, and zero experts",
                "official top-2 routing and classifier",
                "official task-only cross-entropy objective (no auxiliary gate loss)",
            ],
            "adapter_boundary": [
                "shared MoE encoders/data/splits/metrics replace the official data pipeline",
                "naturally missing modalities enter as zero tokens from the shared encoder path",
                "the shared runner owns optimizer, checkpoint selection, and class balancing",
                "the upstream classifier's eager .cuda() is deferred to runner device placement",
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
            raise ValueError("Each MoE++ input must have shape (batch, patches, hidden_dim)")

        # ``encode_batch`` already emits zeros for missing inputs.  Mask again
        # at this boundary so no alternate caller can expose a missing value.
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
        logits = self.fusion_model(inputs)
        result = {"logits": logits}
        if return_aux:
            # The official baseline trainer optimizes task CE only.  Its
            # residual router exposes no load-balancing auxiliary objective.
            result["aux_loss"] = logits.new_zeros(())
        return result
