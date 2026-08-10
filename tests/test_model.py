import torch

from cerd.model import AGMGFlexMoE


def test_forward_generates_only_missing_modalities():
    torch.manual_seed(0)
    model = AGMGFlexMoE(
        num_modalities=4,
        full_modality_index=0,
        num_patches=4,
        hidden_dim=16,
        output_dim=3,
        num_layers_fus=1,
        num_layers_pred=1,
        num_experts=4,
        num_routers=1,
        top_k=2,
        num_heads=4,
        dropout=0.0,
        gen_num_layers=1,
        gen_num_heads=4,
        vectorized_generation=True,
        recon_targets_per_sample=4,
    )
    tokens = [torch.randn(3, 4, 16) for _ in range(4)]
    observed = torch.tensor(
        [[1, 1, 1, 1], [1, 0, 1, 1], [0, 1, 1, 1]], dtype=torch.bool
    )
    output = model(
        *tokens,
        observed_mask=observed,
        expert_indices=torch.tensor([0, 1, 2]),
        return_recon_loss=True,
    )
    assert output["logits"].shape == (3, 3)
    assert output["branch_logits"].shape == (3, 11, 3)
    assert torch.equal(output["generated_mask"], ~observed)
    assert torch.isfinite(output["recon_loss"])
