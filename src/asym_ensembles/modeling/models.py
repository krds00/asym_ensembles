# https://github.com/cptq/asymmetric-networks/blob/main/lmc/models/models_mlp.py
import copy
import itertools
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb


class AsymSwiGLU(nn.Module):
    def __init__(self, dim, scale=1.0, mask_num=0):
        super().__init__()
        g = torch.Generator()
        g.manual_seed(abs(hash(str(mask_num) + str(0))))
        C = torch.randn(dim, dim, generator=g)
        self.register_buffer("C", C)

    def forward(self, x):
        gate = F.sigmoid(F.linear(x, self.C))
        return gate * x


class SigmaMLP(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_dim,
            out_features,
            num_layers,
            norm=None,
            asym_act=True,
    ):
        super().__init__()
        self.lins = nn.ModuleList()
        self.activations = nn.ModuleList()
        if asym_act:
            for i in range(num_layers - 1):
                self.activations.append(AsymSwiGLU(hidden_dim, mask_num=i))
        else:
            for i in range(num_layers - 1):
                self.activations.append(nn.GELU())
        if not norm:
            self.norm = None
        else:
            self.norms = nn.ModuleList()
            if norm == "layer":
                self.norm = nn.LayerNorm
            elif norm == "batch":
                self.norm = nn.BatchNorm1d
            else:
                raise ValueError("Bad norm type. Should be 'layer' or 'batch'")

        if num_layers == 1:
            self.lins.append(nn.Linear(in_features, out_features))

        else:
            if self.norm:
                for _ in range(num_layers - 1):
                    self.norms.append(self.norm(hidden_dim))

            self.lins.append(nn.Linear(in_features, hidden_dim))

            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_dim, hidden_dim))
            self.lins.append(nn.Linear(hidden_dim, out_features))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)

        for idx, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.norm:
                x = self.norms[idx](x)
            x = self.activations[idx](x)
        x = self.lins[-1](x)
        return x

    def count_unused_params(self):
        return 0


class WMLP(nn.Module):
    def __init__(
            self, in_features, hidden_dim, out_features, num_layers, mask_params, norm=None
    ):
        super().__init__()
        self.lins = nn.ModuleList()
        self.mask_params = mask_params
        # Handle norm first
        if not norm:
            self.norm = None
        else:
            self.norms = nn.ModuleList()
            if norm == "layer":
                self.norm = nn.LayerNorm
            elif norm == "batch":
                self.norm = nn.BatchNorm1d
            else:
                raise ValueError("Bad norm type. Should be 'layer' or 'batch'")

        # setup Lins
        if num_layers == 1:
            self.lins.append(
                SparseLinear(in_features, out_features, **mask_params[0], mask_num=0)
            )

        else:
            if self.norm:
                for _ in range(num_layers - 1):
                    self.norms.append(self.norm(hidden_dim))

            self.lins.append(
                SparseLinear(in_features, hidden_dim, **mask_params[0], mask_num=0)
            )
            for i in range(num_layers - 2):
                self.lins.append(
                    SparseLinear(
                        hidden_dim, hidden_dim, **mask_params[i + 1], mask_num=i + 1
                    )
                )
            self.lins.append(
                SparseLinear(
                    hidden_dim,
                    out_features,
                    **mask_params[num_layers - 1],
                    mask_num=num_layers - 1,
                )
            )
        self.activation = nn.GELU()

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)

        for idx, lin in enumerate(self.lins[:-1]):
            prev = x
            x = lin(x)

            if self.norm:
                x = self.norms[idx](x)
            x = self.activation(x)

        x = self.lins[-1](x)
        return x

    def count_unused_params(self):
        return sum(
            lin.count_unused_params() for lin in self.lins if type(lin) != nn.Linear
        )

    def report_masked_ratio(self):
        total_weights = 0
        total_masked = 0
        for lin in self.lins:
            if isinstance(lin, SparseLinear):
                ratio, masked, total = calc_masked_weights_ratio(lin)
                total_weights += total
                total_masked += masked
        ratio = 100.0 * (total_masked / total_weights) if total_weights else 0.0
        return ratio, total_masked


class MLP(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, num_layers, norm=None):
        super().__init__()
        self.lins = nn.ModuleList()
        self.activation = nn.GELU()
        if not norm:
            self.norm = None
        else:
            self.norms = nn.ModuleList()
            if norm == "layer":
                self.norm = nn.LayerNorm
            elif norm == "batch":
                self.norm = nn.BatchNorm1d
            else:
                raise ValueError("Bad norm type. Should be 'layer' or 'batch'")

        if num_layers == 1:
            self.lins.append(nn.Linear(in_features, out_features))

        else:
            if self.norm:
                for _ in range(num_layers - 1):
                    self.norms.append(self.norm(hidden_dim))

            self.lins.append(nn.Linear(in_features, hidden_dim))

            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_dim, hidden_dim))
            self.lins.append(nn.Linear(hidden_dim, out_features))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)

        for idx, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.norm:
                x = self.norms[idx](x)
            x = self.activation(x)
        x = self.lins[-1](x)
        return x

    def count_unused_params(self):
        return 0


class SparseLinear(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            mask_type="densest",
            mask_constant=1,
            mask_num=0,
            num_fixed=6,
            do_normal_mask=True,
    ):

        super().__init__()
        # assert out_features < 2**in_features, "out dim cannot be much higher than in dim" # out_features < С(in_features, num_fixed)
        self.in_features = in_features
        self.out_features = out_features
        mask = make_mask(
            in_features,
            out_features,
            mask_type=mask_type,
            num_fixed=num_fixed,
            mask_num=mask_num,
        )

        self.register_buffer("mask", mask, persistent=True)
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))

        if do_normal_mask:
            self.register_buffer(
                "normal_mask",
                normal_mask(out_features, in_features, mask_num),
                persistent=True,
            )
        else:
            self.register_buffer(
                "normal_mask",
                torch.ones(size=(out_features, in_features)),
                persistent=True,
            )  # torch.ones -> does nothing

        hook = self.weight.register_hook(
            lambda grad: self.mask * grad
        )  # zeros out gradients for masked parts

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.mask_constant = mask_constant
        self.mask_num = mask_num
        self.num_fixed = num_fixed
        self.reset_parameters()

    def forward(self, x):
        self.weight.data = (
                self.weight.data * self.mask
                + (1 - self.mask) * self.mask_constant * self.normal_mask
        )

        return F.linear(x, self.weight, self.bias)
        # return F.linear(x, self.mask * self.weight, self.bias)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # TODO: CHECK THAT
        # self.weight.data = 0 * self.mask
        self.weight.data = (
                self.weight.data * self.mask
                + (1 - self.mask) * self.mask_constant * self.normal_mask
        )  # set entries where mask is zero to the normal mask at that point

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def count_unused_params(self):
        return (1 - self.mask.int()).sum().item()


def get_subset(num_cols, row_idx, num_sample, mask_num):
    g = torch.Generator()
    g.manual_seed(row_idx + abs(hash(str(mask_num))))
    indices = torch.arange(num_cols)
    return indices[torch.randperm(num_cols, generator=g)[:num_sample]]


def normal_mask(out_features, in_features, mask_num):
    g = torch.Generator()
    g.manual_seed(abs(hash(str(mask_num))))
    return torch.randn(size=(out_features, in_features), generator=g)


def make_mask(in_features, out_features, mask_num=0, num_fixed=6, mask_type="densest"):
    # out_features x in_features matrix
    # where each row is unique
    # assert out_features < 2 ** (in_features) # out_features < С(in_features, num_fixed)
    assert in_features > 0 and out_features > 0

    if mask_type == "densest":
        mask = torch.ones(out_features, in_features)
        mask[0, :] = 1  # first row is dense
        row_idx = 1
        if out_features == 1:
            return mask

        for nz in range(1, in_features):
            for zeros_in_row in itertools.combinations(range(in_features), nz):
                mask[row_idx, zeros_in_row] = 0
                row_idx += 1
                if row_idx >= out_features:
                    return mask
    elif mask_type == "bound_zeros":
        # other type of mask based on lower bounding sparsity to break symmetries more
        mask = torch.ones(out_features, in_features)
        least_zeros = num_fixed
        row_idx = 0
        for nz in range(least_zeros, in_features):
            for zeros_in_row in itertools.combinations(range(in_features), nz):
                mask[row_idx, zeros_in_row] = 0
                row_idx += 1
                if row_idx >= out_features:
                    return mask

        raise ValueError(
            "Error in making mask, possibly because out_features is too large for these settings"
        )

    elif mask_type == "random_subsets":
        # other type of mask based on lower bounding sparsity to break symmetries more
        mask = torch.ones(out_features, in_features)
        row_idx = 0
        least_zeros = num_fixed
        for nz in range(least_zeros, in_features):
            while True:

                zeros_in_row = get_subset(in_features, row_idx, least_zeros, mask_num)
                mask[row_idx, zeros_in_row] = 0
                row_idx += 1
                if row_idx >= out_features:
                    return mask

        raise ValueError(
            "Error in making mask, possibly because out_features is too large for these settings"
        )
    else:
        raise ValueError("Invalid mask type")


def calc_masked_weights_ratio(sp_lin: SparseLinear):
    total = sp_lin.mask.numel()
    masked = (1 - sp_lin.mask).sum().item()
    ratio = 100.0 * masked / total if total > 0 else 0.0
    return ratio, masked, total


class InterpolatedModel(nn.Module):
    def __init__(self, models_list):
        super().__init__()
        base = models_list[0]
        if isinstance(base, MLP):
            base_class = MLP
        elif isinstance(base, WMLP):
            base_class = WMLP
        else:
            raise ValueError("Unknown model type")
        in_features = base.lins[0].in_features
        hidden_dim = base.lins[0].out_features
        out_features = base.lins[-1].out_features
        num_layers = len(base.lins)
        if base_class is MLP:
            self.model = MLP(
                in_features, hidden_dim, out_features, num_layers, norm=None
            )
        else:
            mask_params = getattr(base, "mask_params", {}) or {}
            self.model = WMLP(
                in_features,
                hidden_dim,
                out_features,
                num_layers,
                mask_params,
                norm=None,
            )
        self.interpolate_weights(models_list)

    def interpolate_weights(self, models_list):
        with torch.no_grad():
            for param in self.model.parameters():
                param.zero_()
            for m in models_list:
                for p_target, p_source in zip(self.model.parameters(), m.parameters()):
                    p_target.add_(p_source.data)
            for param in self.model.parameters():
                param.div_(len(models_list))

    def forward(self, x):
        return self.model(x)
