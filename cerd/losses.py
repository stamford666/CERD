"""Training losses used by the released CERD configurations.

This module intentionally contains only objectives used by the main method.
Baseline adapters and experimental teacher/SAM/R-Drop objectives are excluded.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitAdjustedCrossEntropy(nn.Module):
    """Training-only CE on logits plus a training-prior adjustment."""

    def __init__(
        self,
        class_prior: Sequence[float] | torch.Tensor,
        tau: float = 0.0,
        label_smoothing: float = 0.0,
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        prior = torch.as_tensor(class_prior, dtype=torch.float32)
        if prior.ndim != 1 or prior.numel() < 2 or float(prior.sum()) <= 0:
            raise ValueError("class_prior must be a non-empty class vector")
        prior = prior / prior.sum().clamp_min(1e-12)
        self.register_buffer("log_prior", torch.log(prior.clamp_min(1e-12)))
        self.register_buffer("weight", weight)
        self.tau = float(tau)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.tau:
            logits = logits + self.tau * self.log_prior.to(logits.device)
        return F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
        )


def per_sample_cross_entropy(
    criterion: nn.Module,
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Apply the configured CE without reducing across samples."""

    if isinstance(criterion, LogitAdjustedCrossEntropy):
        adjusted = logits
        if criterion.tau:
            adjusted = logits + criterion.tau * criterion.log_prior.to(logits.device)
        return F.cross_entropy(
            adjusted,
            target,
            weight=criterion.weight,
            reduction="none",
            label_smoothing=criterion.label_smoothing,
        )
    if isinstance(criterion, nn.CrossEntropyLoss):
        return F.cross_entropy(
            logits,
            target,
            weight=criterion.weight,
            ignore_index=criterion.ignore_index,
            reduction="none",
            label_smoothing=criterion.label_smoothing,
        )
    raise TypeError("criterion must be CrossEntropyLoss or LogitAdjustedCrossEntropy")


def modality_dropout_mask(
    observed_mask: torch.Tensor,
    drop_probability: float,
) -> torch.Tensor:
    """Drop observed modalities while retaining at least one per sample."""

    if drop_probability <= 0:
        return observed_mask
    augmented = observed_mask.bool().clone()
    dropped = (torch.rand_like(augmented.float()) < drop_probability) & augmented
    augmented &= ~dropped
    for row in (~augmented.any(dim=1)).nonzero(as_tuple=False).view(-1).tolist():
        available = observed_mask[row].nonzero(as_tuple=False).view(-1)
        if available.numel():
            keep = available[
                torch.randint(available.numel(), (1,), device=observed_mask.device)
            ]
            augmented[row, keep] = True
    return augmented


def combination_indices_from_mask(
    observed_mask: torch.Tensor,
    modality_codes: str,
    *,
    sort_codes_before_enumeration: bool,
) -> torch.Tensor:
    """Map masks to the data loader's modality-combination ordering."""

    from itertools import combinations

    num_modalities = observed_mask.shape[1]
    codes = list(dict.fromkeys(str(modality_codes).upper()))
    if len(codes) != num_modalities:
        raise ValueError("modality_codes must match the observed-mask width")
    enumeration_codes = sorted(codes) if sort_codes_before_enumeration else codes
    mapping: dict[str, int] = {}
    index = 0
    for size in range(num_modalities, 0, -1):
        for subset in combinations(enumeration_codes, size):
            mapping["".join(sorted(subset))] = index
            index += 1

    lookup = torch.full(
        (2**num_modalities,), -1, dtype=torch.long, device=observed_mask.device
    )
    for bitmask in range(1, 2**num_modalities):
        key = "".join(
            sorted(
                codes[m]
                for m in range(num_modalities)
                if bitmask & (1 << m)
            )
        )
        lookup[bitmask] = mapping[key]
    powers = 1 << torch.arange(num_modalities, device=observed_mask.device)
    encoded = (observed_mask.long() * powers.unsqueeze(0)).sum(dim=1)
    result = lookup[encoded]
    if bool((result < 0).any()):
        raise ValueError("at least one modality must remain")
    return result


def more_fewer_rank_loss(
    more_logits: torch.Tensor,
    fewer_logits: torch.Tensor,
    labels: torch.Tensor,
    more_observed_mask: torch.Tensor,
    fewer_observed_mask: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Mean max(0, CE_more - CE_fewer) over strict reduced views."""

    if more_logits.shape != fewer_logits.shape:
        raise ValueError("more/fewer logits must have identical shapes")
    more_mask = more_observed_mask.bool()
    fewer_mask = fewer_observed_mask.bool()
    if bool((fewer_mask & ~more_mask).any()):
        raise ValueError("fewer modalities must be a subset of more modalities")
    strict = fewer_mask.sum(dim=1) < more_mask.sum(dim=1)
    if not bool(strict.any()):
        return (more_logits.sum() + fewer_logits.sum()) * 0.0
    more_ce = per_sample_cross_entropy(criterion, more_logits, labels)
    fewer_ce = per_sample_cross_entropy(criterion, fewer_logits, labels)
    return F.relu(more_ce[strict] - fewer_ce[strict]).mean()


def dual_boundary_rank_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.20,
    class1_vs_0_weight: float = 2.0 / 3.0,
) -> torch.Tensor:
    """Rank the middle class against both adjacent classes."""

    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError("dual-boundary ranking expects (batch, 3) logits")
    if not 0.0 <= class1_vs_0_weight <= 1.0 or margin < 0:
        raise ValueError("invalid dual-boundary hyperparameters")
    class1 = labels == 1
    terms: list[tuple[float, torch.Tensor]] = []
    if bool(class1.any()):
        for negative_class, weight in (
            (0, class1_vs_0_weight),
            (2, 1.0 - class1_vs_0_weight),
        ):
            negative = labels == negative_class
            if weight <= 0 or not bool(negative.any()):
                continue
            scores = logits[:, 1] - logits[:, negative_class]
            gaps = scores[class1, None] - scores[None, negative]
            terms.append((weight, F.softplus(margin - gaps).mean()))
    if not terms:
        return logits.sum() * 0.0
    if len(terms) == 1:
        return terms[0][1]
    normalizer = sum(weight for weight, _ in terms)
    return torch.stack([weight * loss for weight, loss in terms]).sum() / normalizer


def branch_auxiliary_loss(
    branch_logits: torch.Tensor,
    branch_mask: torch.Tensor,
    branch_quality: torch.Tensor | None,
    labels: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Reliability-weighted auxiliary CE over active diagnostic branches."""

    losses: list[torch.Tensor] = []
    for branch_index in range(branch_logits.shape[1]):
        active = branch_mask[:, branch_index]
        if not bool(active.any()):
            continue
        if branch_quality is None:
            losses.append(criterion(branch_logits[active, branch_index], labels[active]))
            continue
        per_sample = F.cross_entropy(
            branch_logits[active, branch_index],
            labels[active],
            reduction="none",
        )
        quality = branch_quality[active, branch_index].detach().clamp_min(1e-3)
        losses.append((per_sample * quality).sum() / quality.sum().clamp_min(1e-6))
    if not losses:
        return branch_logits.sum() * 0.0
    return torch.stack(losses).mean()


def self_distillation_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Detached full-view teacher to reduced-view student KL."""

    temperature = max(float(temperature), 1e-6)
    teacher = torch.softmax(teacher_logits.detach() / temperature, dim=1)
    student = torch.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(student, teacher, reduction="batchmean") * temperature**2


class BranchDistillationContext(NamedTuple):
    correct_teacher_mask: torch.Tensor
    active_branch_mask: torch.Tensor
    sample_compensation: torch.Tensor


def branch_distillation_context(
    branch_logits: torch.Tensor,
    branch_mask: torch.Tensor,
    labels: torch.Tensor,
    class_accuracy_ema: Sequence[float] | torch.Tensor,
) -> BranchDistillationContext:
    """Freeze correct-teacher gates and sqrt-error class compensation."""

    class_accuracy = torch.as_tensor(
        class_accuracy_ema, device=branch_logits.device, dtype=torch.float64
    )
    if class_accuracy.shape != (branch_logits.shape[2],):
        raise ValueError("class_accuracy_ema must contain one value per class")
    with torch.no_grad():
        active = branch_mask.bool().detach().clone()
        correct = active & branch_logits.detach().argmax(dim=2).eq(labels[:, None])
        compensation = torch.sqrt(
            (1.0 - class_accuracy[labels]).clamp_min(0.0)
        ).to(branch_logits.dtype)
    return BranchDistillationContext(correct, active, compensation.detach())


def masked_branch_tcl_loss(
    branch_logits: torch.Tensor,
    context: BranchDistillationContext,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Correct active branches teach every other active branch."""

    temperature = float(temperature)
    batch_size, branch_count, _ = branch_logits.shape
    teacher_gate = context.correct_teacher_mask.to(branch_logits.device)
    active = context.active_branch_mask.to(branch_logits.device)
    compensation = context.sample_compensation.to(branch_logits.device)
    not_self = ~torch.eye(branch_count, device=branch_logits.device, dtype=torch.bool)
    pair_mask = teacher_gate[:, :, None] & active[:, None, :] & not_self[None]
    pair_counts = pair_mask.sum(dim=(1, 2))
    weights = compensation * (pair_counts > 0).to(compensation.dtype)
    if not bool((weights > 0).any()):
        return branch_logits.sum() * 0.0
    teacher_log = F.log_softmax(branch_logits / temperature, dim=2).detach()
    teacher_prob = teacher_log.exp()
    student_log = F.log_softmax(branch_logits / temperature, dim=2)
    per_pair = (
        teacher_prob[:, :, None, :]
        * (teacher_log[:, :, None, :] - student_log[:, None, :, :])
    ).sum(dim=3) * temperature**2
    per_sample = (per_pair * pair_mask).sum(dim=(1, 2)) / pair_counts.clamp_min(1)
    return (per_sample * weights).sum() / weights.sum()


def trusted_branch_fusion_distillation_loss(
    fused_logits: torch.Tensor,
    branch_logits: torch.Tensor,
    context: BranchDistillationContext,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Distill the mean trusted-branch distribution into fused logits."""

    temperature = float(temperature)
    teacher_gate = context.correct_teacher_mask.to(branch_logits.device)
    compensation = context.sample_compensation.to(branch_logits.device)
    teacher_counts = teacher_gate.sum(dim=1)
    weights = compensation * (teacher_counts > 0).to(compensation.dtype)
    if not bool((weights > 0).any()):
        return fused_logits.sum() * 0.0
    branch_prob = torch.softmax(branch_logits / temperature, dim=2).detach()
    teacher = (
        (branch_prob * teacher_gate[:, :, None]).sum(dim=1)
        / teacher_counts.clamp_min(1)[:, None]
    ).detach()
    student_log = F.log_softmax(fused_logits / temperature, dim=1)
    per_sample = F.kl_div(student_log, teacher, reduction="none").sum(dim=1)
    per_sample = per_sample * temperature**2
    return (per_sample * weights).sum() / weights.sum()


class BranchClassAccuracyEMA:
    """Epoch-lagged active-branch class accuracy used by TCL/TBFD."""

    def __init__(self, num_classes: int, decay: float = 0.9) -> None:
        self.num_classes = int(num_classes)
        self.decay = float(decay)
        self.class_accuracy = np.zeros(self.num_classes, dtype=np.float64)
        self.initialized = np.zeros(self.num_classes, dtype=bool)
        self.reset()

    def reset(self) -> None:
        self.correct = np.zeros(self.num_classes, dtype=np.int64)
        self.valid = np.zeros(self.num_classes, dtype=np.int64)

    def collect(
        self,
        branch_logits: torch.Tensor,
        branch_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            active = branch_mask.bool()
            correct = active & branch_logits.argmax(dim=2).eq(labels[:, None])
            for class_index in range(self.num_classes):
                rows = labels == class_index
                self.correct[class_index] += int(correct[rows].sum().cpu())
                self.valid[class_index] += int(active[rows].sum().cpu())

    def context(
        self,
        branch_logits: torch.Tensor,
        branch_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> BranchDistillationContext:
        return branch_distillation_context(
            branch_logits, branch_mask, labels, self.class_accuracy
        )

    def update(self) -> list[float]:
        for class_index in range(self.num_classes):
            if self.valid[class_index] == 0:
                continue
            observed = self.correct[class_index] / self.valid[class_index]
            if self.initialized[class_index]:
                self.class_accuracy[class_index] = (
                    self.decay * self.class_accuracy[class_index]
                    + (1.0 - self.decay) * observed
                )
            else:
                self.class_accuracy[class_index] = observed
                self.initialized[class_index] = True
        result = self.class_accuracy.tolist()
        self.reset()
        return result
