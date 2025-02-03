from torch import nn
import torch.nn.functional as F


class GumbelGatingNetwork(nn.Module):
    """
    MLP gating with BayesianLinear for each layer.
    We'll provide a method to sum the KL from both layers.
    """

    def __init__(self, in_features=784, num_experts=3, tau=1.0, device='cuda'):
        super().__init__()
        self.lin = nn.Linear(in_features, num_experts)
        self.tau = tau
        self.device = device
        print(f'tau={self.tau}')

    def forward(self, x, num_samples: int, hard: bool = False):
        logits = self.lin(x)  # shape: (batch, num_experts) or (num_samples, batch, num_experts)
        if num_samples < 2:
            # alpha = torch.softmax(logits, dim=-1)  # gating coefficients
            alpha = F.gumbel_softmax(logits, dim=-1, tau=self.tau, hard=hard)  # gating coefficients
        else:
            # Expand logits along the batch dimension for num_samples
            logits_expanded = logits.unsqueeze(0).expand(num_samples, -1, -1)

            # Sample using PyTorch's gumbel_softmax function
            alpha = F.gumbel_softmax(logits_expanded, tau=self.tau, hard=hard, dim=-1)
        return alpha
