import torch

from cerd.losses import (
    BranchClassAccuracyEMA,
    dual_boundary_rank_loss,
    masked_branch_tcl_loss,
    more_fewer_rank_loss,
    trusted_branch_fusion_distillation_loss,
)


def test_dual_boundary_prefers_correct_ordering():
    labels = torch.tensor([0, 1, 2])
    good = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
    bad = torch.tensor([[0.0, 3.0, 0.0], [3.0, 0.0, 3.0], [0.0, 3.0, 0.0]])
    assert dual_boundary_rank_loss(good, labels) < dual_boundary_rank_loss(bad, labels)


def test_more_fewer_matches_per_sample_ce_formula():
    labels = torch.tensor([0, 1])
    more = torch.tensor([[3.0, 0.0], [0.0, 3.0]], requires_grad=True)
    fewer = torch.tensor([[0.0, 3.0], [3.0, 0.0]], requires_grad=True)
    more_mask = torch.tensor([[1, 1], [1, 1]], dtype=torch.bool)
    fewer_mask = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)
    loss = more_fewer_rank_loss(
        more,
        fewer,
        labels,
        more_mask,
        fewer_mask,
        torch.nn.CrossEntropyLoss(),
    )
    assert loss.item() == 0.0
    loss.backward()


def test_tcl_and_tbfd_are_differentiable():
    branch_logits = torch.tensor(
        [[[3.0, 0.0], [2.0, 0.0], [0.0, 2.0]], [[0.0, 3.0], [0.0, 2.0], [2.0, 0.0]]],
        requires_grad=True,
    )
    fused_logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    branch_mask = torch.ones(2, 3, dtype=torch.bool)
    labels = torch.tensor([0, 1])
    ema = BranchClassAccuracyEMA(2)
    context = ema.context(branch_logits, branch_mask, labels)
    tcl = masked_branch_tcl_loss(branch_logits, context)
    tbfd = trusted_branch_fusion_distillation_loss(fused_logits, branch_logits, context)
    assert torch.isfinite(tcl)
    assert torch.isfinite(tbfd)
    (tcl + tbfd).backward()
    assert branch_logits.grad is not None
    assert fused_logits.grad is not None
