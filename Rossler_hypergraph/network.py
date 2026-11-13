import os, sys
import torch
import torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from LV_half_observed.network import ModelMix, RandomFourierFeaturesNN, compute_full_RFF_derivatives_LV as basis_with_derivative   # shared trunk + RFF basis


class EndToEnd(nn.Module):
    """u_{k,i}(t) = Phi(t) @ theta[k,i]: one shared basis, one coefficient block per agent and experiment; holds the
    physical parameters too, so the warm-up can train everything with Adam."""
    def __init__(self, backbone, rff_module, theta, params):
        super().__init__()
        self.backbone, self.rff = backbone, rff_module
        self.theta, self.params = nn.Parameter(theta), nn.Parameter(params)       # (K, N, P, 3), (q,)

    def forward(self, t):
        Phi, _ = self.rff(self.backbone, t)
        return torch.einsum('tp,kipa->ktia', Phi, self.theta)                      # (K, T, N, 3)
