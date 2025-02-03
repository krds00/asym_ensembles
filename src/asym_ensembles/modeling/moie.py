import torch
from torch import nn
from src.asym_ensembles.modeling.models import SparseLinear
from src.asym_ensembles.modeling.gumbel import GumbelGatingNetwork


class MoIEBlock(nn.Module):
    """
    One "layer" containing K linear experts of shape (in_dim -> out_dim).
    In forward pass, we:
      1) Pass x through each expert -> Z^{(k)}  [size (B, out_dim)]
      2) Compute alpha * Z^{(k)} and sum across k -> Z_combined
      3) Optionally apply an activation (e.g. ReLU).
    """

    def __init__(self, in_dim, out_dim, num_experts, experts_type_str, mask_params, layer_number, activation=True):
        """

        @type experts_type_str: str in ['MLP', 'WMLP']
        """
        assert experts_type_str in ['imlp', 'iwmlp'], f'experts type: {experts_type_str} is invalid '
        super().__init__()
        self.num_experts = num_experts
        self.activation = activation

        # Create K linear experts as a ModuleList
        self.experts = nn.ModuleList([
            nn.Linear(in_dim, out_dim) if experts_type_str == 'imlp'
            else SparseLinear(in_dim, out_dim, **mask_params[0], mask_num=layer_number)
            for _ in range(num_experts)
        ])
        if self.activation:
            self.relu = nn.ReLU()

    def forward(self, x, alpha):
        """
        Args:
            x: shape (B, in_dim)
            alpha: shape (B, K) gating coefficients (one row per sample).
        Returns:
            combined: shape (B, out_dim)
        """

        # 1) For each expert k, compute Z^{(k)} = x W^{(k)} + b^{(k)}.
        #    We'll stack them to shape (K, B, out_dim) for convenience.
        expert_outputs = []
        for k in range(self.num_experts):
            Z_k = self.experts[k](x)  # shape (B, out_dim)
            expert_outputs.append(Z_k)
        # Stack => shape (K, B, out_dim)
        expert_outputs = torch.stack(expert_outputs, dim=0)

        # 2) Combine with alpha: Z_combined(i,:) = sum_k alpha[i,k] * Z_k(i,:)
        #    We can do this in a batched manner:
        #    Make alpha shape (B, K, 1) => then broadcast multiply with
        #    expert_outputs (K, B, out_dim) after transposing or rearranging.
        #    Easiest is to transpose expert_outputs to (B, K, out_dim) first.
        expert_outputs = expert_outputs.transpose(0, 1)  # => (B, K, out_dim)

        # alpha: (B, K) => alpha.unsqueeze(-1): (B, K, 1)
        alpha_3d = alpha.unsqueeze(-1)  # => (B, K, 1)

        # Multiply elementwise and sum over K => (B, out_dim)
        combined = (expert_outputs * alpha_3d).sum(dim=1)  # (B, out_dim)

        # 3) Optional activation
        if self.activation:
            combined = self.relu(combined)
        return combined


class MoIE(nn.Module):
    def __init__(
            self,
            *,
            in_dim: None | int = None,
            out_dim: None | int = None,
            num_layers: int,
            hidden_dim: int,
            # dropout: float, TODO: let's add?
            activation: str = 'ReLU',
            # activation: str = 'GELU',
            num_experts: int,
            experts_type_str: str,
            mask_params: None | dict,
            gating_type: str = 'standard',  # ['standard' or 'bayesian']
            device: str = 'cpu',
            tau: int = 2.0,
            default_num_samples: int = 10,
    ) -> None:

        assert gating_type in ['standard', 'gumbel']
        assert experts_type_str in ['imlp', 'iwmlp']

        super().__init__()

        self.default_num_samples = default_num_samples
        self.device = device
        self.num_experts = num_experts
        self.gating_type = gating_type
        self.hidden_dim = hidden_dim
        self.mask_params = mask_params
        print(f'gating type: {self.gating_type}')
        print(f'num experts: {self.num_experts}')
        print(f'hidden dim: {self.hidden_dim}')
        print(f'mask params: {self.mask_params}')
        # d_first = hidden_dim // num_experts if in_dim is None else in_dim
        d_first = hidden_dim if in_dim is None else in_dim

        self.stat_alpha_sum = None
        # Gating network
        if self.gating_type == 'standard':
            self.gate = nn.Sequential(
                nn.Linear(d_first, num_experts),
                nn.Softmax(dim=-1)
            )
        elif self.gating_type == 'gumbel':
            self.gate = GumbelGatingNetwork(d_first, num_experts, tau=tau, device=device)
        else:
            assert False, f'The gating type {self.gating_type} is not supported'

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    # MoIEBlock(d_first if i == 0 else hidden_dim // num_experts, hidden_dim // num_experts,
                    MoIEBlock(d_first if i == 0 else hidden_dim, hidden_dim,
                              num_experts,
                              experts_type_str, mask_params, layer_number=i, activation=False),
                    getattr(nn, activation)(),
                    # nn.Dropout(dropout)
                )
                for i in range(num_layers)
            ]
        )

        # elif self.gating_type == 'bayesian':
        #     self.gate = BayesianGatingNetwork(
        #         in_features=d_first,
        #         num_experts=num_experts,
        #         prior_std=gating_prior_std,
        #         device=self.device,
        #     )
        #
        #     self.blocks = nn.ModuleList(
        #         [
        #             nn.Sequential(
        #                 MoIEBlock(d_first if i == 0 else hidden_dim // num_experts, hidden_dim // num_experts, num_experts,
        #                           activation=False),
        #                 getattr(nn, activation)()
        #             )
        #             for i in range(num_layers)
        #         ]
        #     )

        # self.output = None if out_dim is None else MoIEBlock(hidden_dim // num_experts, out_dim, num_experts,
        self.output = None if out_dim is None else MoIEBlock(hidden_dim, out_dim, num_experts,
                                                             experts_type_str, mask_params, num_layers,
                                                             activation=False)
        # print(f'out_dim:{out_dim}')
        # print(self.blocks)
        # print('output:')
        # print(self.output)

        self.device = device
        self.gating_type = gating_type
        self.stat_alpha_sum = None

    def forward(self, x):
        """
        x shape: (B, input_dim)
        alpha shape: (B, K)

        For simplicity, we'll assume we use the *same* alpha at each layer.
        If you want different gating per layer, you'd have multiple gating nets
        or a more advanced design.
        """
        if self.training:
            num_samples = 1
        else:
            num_samples = self.default_num_samples

        alpha = self.gate(x) if self.gating_type == 'standard' else self.gate(x, num_samples)  # shape (B, K)
        # store for later analysis
        if self.training:
            if self.stat_alpha_sum is None:
                self.stat_alpha_sum = alpha.sum(axis=0).detach().cpu().numpy()
            else:
                self.stat_alpha_sum += alpha.sum(axis=0).detach().cpu().numpy()
        # if np.random.random() < 0.01:
        #     print(f'alphas:{self.stat_alpha_mean}')

        if not self.training and self.gating_type == 'gumbel':
            # since it is a linear interpolation, we can do monte carlo here
            alpha = torch.mean(alpha, dim=0)
        # Pass through 1st MoE block
        for block in self.blocks:
            x = block[0](x, alpha)  # Pass both arguments to the first block
            for i in range(1, len(block)):
                x = block[i](x)  # Apply the activation and dropout
        if self.output is not None:
            x = self.output(x, alpha)
        return x
