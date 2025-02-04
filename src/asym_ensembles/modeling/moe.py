from typing import Dict, List, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoE(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        expert_class: Union[Type[nn.Module], List[Type[nn.Module]]],
        expert_params: Union[Dict, List[Dict]],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.gating_layer = nn.Linear(in_features, num_experts)
        self.init_gating_layer()

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
            par_k = experts_params_list[i]

            if "in_features" not in par_k:
                par_k["in_features"] = in_features
            if "out_features" not in par_k:
                par_k["out_features"] = out_features

            expert_k = cls_k(**par_k)
            self.experts.append(expert_k)

    def init_gating_layer(self):
        """
        Xavier Uniform.
        """
        nn.init.xavier_uniform_(self.gating_layer.weight)
        if self.gating_layer.bias is not None:
            nn.init.constant_(self.gating_layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_features).
        Output: (batch, out_features).
        """
        logits = self.gating_layer(x)  # (batch, num_experts)
        alpha = F.softmax(logits, dim=-1)  # (batch, num_experts)

        expert_outs = []
        for k in range(self.num_experts):
            y_k = self.experts[k](x)  # (batch, out_features)
            expert_outs.append(y_k)

        stack_yo = torch.stack(expert_outs, dim=0)

        # alpha: (batch, num_experts) => (num_experts, batch)
        alpha_t = alpha.transpose(0, 1).unsqueeze(-1)  # => (K, batch, 1)

        # weighted sum => (K, batch, out_features) * (K, batch, 1)
        # sum over K => (batch, out_features)
        weighted = alpha_t * stack_yo
        y = weighted.sum(dim=0)  # (batch, out_features)
        return y
