import torch
import torch.nn.functional as F
import torch.nn as nn
from copy import deepcopy
from fmoe.transformer import _Expert
from fmoe.layers import FMoE, _fmoe_general_global_forward, mark_module_parallel_comm
from fmoe.functions import ensure_comm, Slice, AllGather
from fmoe.gates import NaiveGate

import tree

from fmoe.gates import NoisyGate

import math

class FixedFMoE(nn.Module):
    def __init__(self, num_expert=32, d_model=1024, world_size=1, mp_group=None, slice_group=None, moe_group=None, top_k=2, gate=NaiveGate, expert=None, gate_hook=None, mask=None, mask_dict=None, **kwargs):
        super().__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.world_size = world_size
        self.slice_group = slice_group
        if mp_group is not None:
            print("[Warning] mp_group is being deprecated")
            self.slice_group = mp_group
        if self.slice_group is None:
            self.slice_size = 1
            self.slice_rank = 0
        else:
            self.slice_size = self.slice_group.size()
            self.slice_rank = self.slice_group.rank()
        self.top_k = top_k
        if type(expert) is list:
            self.experts = nn.ModuleList([e(d_model) for e in expert])
            self.experts_fused = False
            self.num_expert = num_expert = len(expert)
        elif expert is not None:
            self.experts = nn.ModuleList([expert(d_model) for _ in range(num_expert)])
            self.experts_fused = False
        else:
            self.experts_fused = True
        self.gate = gate(d_model, num_expert, world_size, top_k)
        self.gate_hook = gate_hook
        self.mask = mask
        self.mask_dict = mask_dict
        self.moe_group = moe_group

    def expert_fn(self, inp, fwd_expert_count):
        if self.experts_fused:
            return self.experts(inp, fwd_expert_count)
        if isinstance(fwd_expert_count, torch.Tensor):
            fwd_expert_count = fwd_expert_count.cpu().numpy()
        outputs = []
        base_idx = 0
        for i in range(self.num_expert):
            batch_size = fwd_expert_count[i]
            inp_slice = inp[base_idx : base_idx + batch_size]
            outputs.append(self.experts[i](inp_slice))
            base_idx += batch_size
        return torch.cat(outputs, dim=0)

    def mark_parallel_comm(self, expert_dp_comm="none"):
        if self.experts is not None:
            comm = expert_dp_comm
            if isinstance(self.experts, list):
                for e in self.experts:
                    mark_module_parallel_comm(e, comm)
            else:
                mark_module_parallel_comm(self.experts, comm)
        mark_module_parallel_comm(self.gate, "gate")

    def forward(self, moe_inp, expert_indices=None):
        moe_inp_batch_size = tree.flatten(tree.map_structure(lambda tensor: tensor.shape[0], moe_inp))
        assert all([batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]), "MoE inputs must have the same batch size"

        if self.world_size > 1:
            def ensure_comm_func(tensor):
                ensure_comm(tensor, self.moe_group)
            tree.map_structure(ensure_comm_func, moe_inp)

        if self.slice_size > 1:
            def slice_func(tensor):
                return Slice.apply(tensor, self.slice_rank, self.slice_size, self.slice_group)
            moe_inp = tree.map_structure(slice_func, moe_inp)

        gate_top_k_idx, gate_score = self.gate(moe_inp, expert_indices)

        # Reshape gate_top_k_idx to be 2-dimensional
        gate_top_k_idx = gate_top_k_idx.view(moe_inp.shape[0], self.top_k)

        if self.gate_hook is not None:
            self.gate_hook(gate_top_k_idx, gate_score, None)
        self.gate.set_topk_indicates(gate_top_k_idx)

        if self.mask is not None and self.mask_dict is not None:
            def delete_mask_func(tensor):
                tensor = tensor[self.mask == 0, :]
                return tensor
            mask = self.mask.view(-1)
            moe_inp = tree.map_structure(delete_mask_func, moe_inp)
            gate_top_k_idx = gate_top_k_idx[mask == 0, :]

        fwd = _fmoe_general_global_forward(moe_inp, gate_top_k_idx, self.expert_fn, self.num_expert, self.world_size, experts=self.experts)

        if self.mask is not None and self.mask_dict is not None:
            def recover_func(tensor):
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                x = torch.zeros(mask.shape[0], self.top_k, dim, device=tensor.device, dtype=tensor.dtype)
                x[mask == 0] = tensor
                for k, v in self.mask_dict.items():
                    x[mask == k] = v
                return x
            moe_outp = tree.map_structure(recover_func, fwd)
        else:
            def view_func(tensor):
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                return tensor
            moe_outp = tree.map_structure(view_func, fwd)

        gate_score = gate_score.view(-1, 1, self.top_k)
        def bmm_func(tensor):
            dim = tensor.shape[-1]
            tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
            return tensor
        moe_outp = tree.map_structure(bmm_func, moe_outp)

        if self.slice_size > 1:
            def all_gather_func(tensor):
                return AllGather.apply(tensor, self.slice_rank, self.slice_size, self.slice_group)
            moe_outp = tree.map_structure(all_gather_func, moe_outp)

        moe_outp_batch_size = tree.flatten(tree.map_structure(lambda tensor: tensor.shape[0], moe_outp))
        assert all([batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]), "MoE outputs must have the same batch size"
        return moe_outp


class FMoETransformerMLP(FixedFMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        expert_dp_comm="none",
        expert_rank=0,
        n_router = 1,
        gate='AddtionalNoisyGate', # NaiveGate
        world_size=1,
        top_k=2,
        normalized_gate_loss=False,
        **kwargs
    ):
        if type(gate) is str:
            gate = eval(gate)
        super().__init__(num_expert=num_expert, d_model=d_model, gate=gate, world_size=world_size, top_k=top_k, **kwargs)
        self.experts = _Expert(
            num_expert, d_model, d_hidden, activation, rank=expert_rank
        )
        self.cpu_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_hidden), deepcopy(activation), nn.Linear(d_hidden, d_model))
            for _ in range(num_expert)
        ]) if not torch.cuda.is_available() else None
        self.n_router = n_router
        self.all_gates = nn.ModuleDict({f'{i}': gate(d_model, num_expert, world_size, top_k) for i in range(n_router)})
        self.gate = self.all_gates[f'{0}']
        for gate_module in self.all_gates.values():
            if hasattr(gate_module, 'set_normalized_gate_loss'):
                gate_module.set_normalized_gate_loss(normalized_gate_loss)

        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp: torch.Tensor, expert_indices=None):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)

        if inp.device.type == 'cpu' and self.cpu_experts is not None:
            top_idx, top_score = self.gate(inp, expert_indices)
            top_idx = top_idx.view(inp.shape[0], self.top_k)
            top_score = top_score.view(inp.shape[0], self.top_k)
            self.gate.set_topk_indicates(top_idx)
            output = torch.zeros_like(inp)
            for rank in range(self.top_k):
                for expert_id, expert in enumerate(self.cpu_experts):
                    selected = top_idx[:, rank] == expert_id
                    if selected.any():
                        output[selected] += top_score[selected, rank, None] * expert(inp[selected])
        else:
            output = super().forward(inp, expert_indices=expert_indices)

        return output.reshape(original_shape)

    def set_full_modality(self, is_full_modality):
        for gate in self.all_gates.values():
            if hasattr(gate, 'set_full_modality'):
                gate.set_full_modality(is_full_modality)


class AddtionalNoisyGate(NoisyGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(d_model, num_expert, world_size, top_k)
        self.topk_logits = []
        self.indicates = None
        self.is_full_modality = False
        self.normalized_gate_loss = False
        self._normalized_loss_call_count = 0

    def set_topk_logit(self, logit):
        self.topk_logits.append(logit)

    def get_topk_logit(self, clear = True):
        topk_logit = self.topk_logits
        if clear:
            self.topk_logits = None
        return topk_logit

    def set_topk_indicates(self, indicate):
        self.indicates = indicate

    def get_topk_indicate(self, clear = True):
        topk_indicate = self.indicates
        if clear:
            self.indicates = None
        return topk_indicate

    def set_loss(self, loss):
        if self.normalized_gate_loss:
            self._normalized_loss_call_count += 1
        if self.loss is None:
            self.loss = loss
        else:
            self.loss += loss

    def get_loss(self, clear=True):
        if not self.normalized_gate_loss:
            # Preserve the historical NoisyGate retrieval path exactly.
            return super().get_loss(clear=clear)
        loss = super().get_loss(clear=False)
        if loss is None:
            return None
        if self._normalized_loss_call_count <= 0:
            raise RuntimeError("Normalized gate loss has no recorded router calls")
        normalized = loss / float(self._normalized_loss_call_count)
        if clear:
            self.loss = None
            self._normalized_loss_call_count = 0
        return normalized

    def set_normalized_gate_loss(self, enabled):
        if type(enabled) is not bool:
            raise TypeError("normalized_gate_loss must be a bool")
        if self.loss is not None:
            raise RuntimeError("Cannot change gate-loss normalization with pending loss")
        self.normalized_gate_loss = enabled
        self._normalized_loss_call_count = 0

    def set_full_modality(self, is_full_modality):
        self.is_full_modality = is_full_modality

    def forward(self, inp, expert_indices=None):
        clean_logits = inp @ self.w_gate
        raw_noise_stddev = inp @ self.w_noise
        noise_stddev = (
            self.softplus(raw_noise_stddev) + self.noise_epsilon
        ) * self.training
        noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        logits = noisy_logits
        loss = 0

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(
            min(self.top_k + 1, self.tot_expert), dim=1
        )
        top_k_logits = top_logits[:, : self.top_k]
        top_k_indices = top_indices[:, : self.top_k]
        top_k_gates = self.softmax(top_k_logits)

        if (expert_indices != None) & (expert_indices.sum() > 0):
            batch_size = inp.shape[0]
            num_experts = expert_indices.shape[0]

            repeats = batch_size // num_experts
            remainder = batch_size % num_experts

            if repeats > 0:
                expert_indices_expanded = expert_indices.repeat(repeats, 1).T.reshape(-1)

            else:
                expert_indices_expanded = torch.tensor([], dtype=expert_indices.dtype, device=expert_indices.device)

            if remainder > 0:
                expert_indices_expanded = torch.cat([expert_indices_expanded, torch.tensor([expert_indices[-1]]*remainder).to(expert_indices.device)])

            full_modality_mask_expanded = expert_indices_expanded == 0

        if expert_indices is not None and expert_indices.sum() > 0:
            scalable_targets = expert_indices_expanded.remainder(self.tot_expert)
            if self.normalized_gate_loss:
                # Preserve prevalence: a rare missing-pattern row should not
                # receive the same total router supervision as a fully missing
                # minibatch merely because CE normally averages its subset.
                expert_idx_loss = F.cross_entropy(
                    logits[~full_modality_mask_expanded],
                    scalable_targets[~full_modality_mask_expanded],
                    reduction='sum',
                ) / float(logits.shape[0])
            else:
                expert_idx_loss = F.cross_entropy(logits[~full_modality_mask_expanded], scalable_targets[~full_modality_mask_expanded])
            loss += expert_idx_loss

        self.set_topk_logit(top_k_indices)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.top_k < self.tot_expert and self.training:
            load = (
                self._prob_in_top_k(
                    clean_logits, noisy_logits, noise_stddev, top_logits
                )
            )
        else:
            load = self._gates_to_load(gates)

        if (expert_indices != None):
            full_modality_mask = expert_indices == 0
            if full_modality_mask.sum() == len(full_modality_mask):
                load = load.sum(0) if self.training else load
                importance = gates.sum(0) if self.training else gates.sum(0)
                loss += self.cv_squared(importance) + self.cv_squared(load)
            else:
                importance_1 = gates[full_modality_mask_expanded, :].sum(0) if self.training else gates.sum(0)
                load_1 = load[full_modality_mask_expanded, :].sum(0) if self.training else load
                loss_1 = self.cv_squared(importance_1) + self.cv_squared(load_1)

                importance_2 = gates[~full_modality_mask_expanded, 1:].sum(0) if self.training else gates.sum(0)
                load_2 = load[~full_modality_mask_expanded, 1:].sum(0) if self.training else load
                loss_2 = self.cv_squared(importance_2) + self.cv_squared(load_2)

                if self.normalized_gate_loss:
                    full_fraction = full_modality_mask_expanded.to(
                        dtype=logits.dtype
                    ).mean()
                    loss = (
                        loss
                        + full_fraction * loss_1
                        + (1.0 - full_fraction) * loss_2
                    )
                else:
                    loss = loss + loss_1 + loss_2
        else:
            load = load.sum(0) if self.training else load
            importance = gates.sum(0) if self.training else gates.sum(0)
            loss += self.cv_squared(importance) + self.cv_squared(load)

        self.set_loss(loss)

        return (
            top_k_indices.contiguous().view(-1),
            top_k_gates.contiguous().unsqueeze(1),
        )


# =====================================================================================
# Innovation modules (for single-node training, avoids FMoE dispatch constraints)
# - Expert-Choice routing (experts pick tokens) to eliminate load imbalance
# - Slot-based Soft-MoE fusion (no token dropping) to stabilize small datasets
# =====================================================================================


class _RouterLossMixin:
    """Minimal interface so FlexMoE.gate_loss() can aggregate router losses."""

    def __init__(self):
        super().__init__()
        self._router_loss = None

    def set_loss(self, loss: torch.Tensor | None):
        self._router_loss = loss

    def get_loss(self):
        # return and clear to match the original behavior
        if self._router_loss is None:
            return None
        loss = self._router_loss
        self._router_loss = None
        return loss


def _cv_squared(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    # Same spirit as Switch / NoisyGate balance objective
    if x.numel() == 0:
        return torch.zeros([], device=x.device)
    mean = x.mean()
    var = x.var(unbiased=False)
    return var / (mean * mean + eps)
