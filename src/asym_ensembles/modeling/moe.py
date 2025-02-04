from typing import Dict, List, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.asym_ensembles.modeling.gumbel import GumbelGatingNetwork


class MoE(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            num_experts: int,
            expert_class: Union[Type[nn.Module], List[Type[nn.Module]]],
            expert_params: Union[Dict, List[Dict]],
            gating_type: str = 'standard',  # {'standard', 'gumbel'}
            tau: float = 2.0,  # {'standard', 'softmax'}
            device: str = 'cpu',
            default_num_samples: int = 10,
    ):
        assert gating_type in ['standard', 'gumbel'], f"gating type:{gating_type} is not supported"

        super().__init__()

        self.gating_type = gating_type
        self.default_num_samples = default_num_samples
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_experts = num_experts
        # self.gating_layer = nn.Linear(in_dim, num_experts)
        # self.init_gating_layer()
        if self.gating_type == 'standard':
            self.gate = nn.Sequential(
                nn.Linear(self.in_dim, num_experts),
                nn.Softmax(dim=-1)
            )
        elif self.gating_type == 'gumbel':
            self.gate = GumbelGatingNetwork(self.in_dim, num_experts, tau=tau, device=device)
        else:
            assert False, f'The gating type {self.gating_type} is not supported'

        if isinstance(expert_class, list):
            assert len(expert_class) == num_experts, (
                f"Length of expert_class ({len(expert_class)}) "
                f"should be equal to num_experts ({num_experts})"
            )
            expert_classes = expert_class
        else:
            expert_classes = [expert_class for _ in range(num_experts)]

        if isinstance(expert_params, list):
            assert len(expert_params) == num_experts, (
                f"Length of expert_params ({len(expert_params)}) "
                f"should be equal to num_experts ({num_experts})"
            )
            experts_params_list = expert_params
        else:
            experts_params_list = [expert_params for _ in range(num_experts)]

        self.experts = nn.ModuleList()
        for i in range(num_experts):
            cls_k = expert_classes[i]
            par_k = experts_params_list[i]  # TODO: nicht verstanden, warum machst du es?

            if "in_dim" not in par_k:
                par_k["in_dim"] = in_dim
            if "out_dim" not in par_k:
                par_k["out_dim"] = out_dim

            expert_k = cls_k(**par_k)
            self.experts.append(expert_k)

    def init_gating_layer(self):  # TODO: why do not use a default init?
        """
        Xavier Uniform.
        """
        nn.init.xavier_uniform_(self.gating_layer.weight)
        if self.gating_layer.bias is not None:
            nn.init.constant_(self.gating_layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_dim).
        Output: (batch, out_dim).
        """
        if self.training:
            num_samples = 1
        else:
            num_samples = self.default_num_samples

        alpha = self.gate(x) if self.gating_type == 'standard' else self.gate(x, num_samples)
        if alpha.dim() == 2:
            alpha = alpha.transpose(-1, -2)
        else:
            alpha = alpha.permute(0, 2, 1)
        expert_outs = []
        for k in range(self.num_experts):
            y_k = self.experts[k](x)  # (batch, out_dim)
            expert_outs.append(y_k)
        stack_yo = torch.stack(expert_outs, dim=0)  # (num_experts,batch,out)

        if self.training or num_samples < 2 or self.gating_type == 'standard':
            output = torch.sum(alpha.unsqueeze(-1) * stack_yo, dim=0)
        else:

            # print('alpha shape:')
            # print(alpha.shape)
            # print('check stack_yo shape')
            # print(stack_yo.shape)
            # EVAL MODE (monte carlo)
            weighted_expert_outputs = alpha.unsqueeze(-1) * stack_yo.unsqueeze(0)

            # 4) Sum over experts => [10, batch_size, output_dim]
            weighted_sums = torch.sum(weighted_expert_outputs, dim=1)
            # print('wegithed sums:')
            # print(weighted_sums.shape)

            # [ num_samples, batch_size, output_dim]
            output = weighted_sums.mean(dim=0)
        return output

        # # alpha: (batch, num_experts) => (num_experts, batch)
        # alpha_t = alpha.transpose(0, 1).unsqueeze(-1)  # => (K, batch, 1)
        #
        # # weighted sum => (K, batch, out_dim) * (K, batch, 1)
        # # sum over K => (batch, out_dim)
        # weighted = alpha_t * stack_yo
        # y = weighted.sum(dim=0)  # (batch, out_dim)
        # return y
