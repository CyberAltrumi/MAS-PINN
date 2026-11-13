import math
import torch
import torch.nn as nn
from torch import func

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelMix(nn.Module):
    """tanh MLP trunk on t (rescaled to [-1, 1]); get_hidden_output feeds the RFF basis."""
    def __init__(self, hidden_size, t_max=3.0):
        super().__init__()
        self.t_max = t_max
        self.hidden_layers = nn.ModuleList()
        in_dim = 1
        for h_dim in hidden_size:
            self.hidden_layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def get_hidden_output(self, t):
        out = 2 * t / self.t_max - 1
        for layer in self.hidden_layers:
            out = torch.tanh(layer(out))
        return out


# RFF mapping: Phi(t) = [1, sqrt(2/K) cos(z W), sqrt(2/K) sin(z W)],  z = trunk output.
# The constant column carries each agent's mean level exactly (zero time derivative).
class RandomFourierFeaturesNN(nn.Module):
    def __init__(self, input_dim, num_features, sigma=1.0, rff_device=None):
        super().__init__()
        self.num_features = num_features
        self.register_buffer('W', torch.randn(input_dim, num_features, device=device if rff_device is None else rff_device) * sigma)

    def forward(self, model, t):
        P = model.get_hidden_output(t) @ self.W
        norm = math.sqrt(2.0 / self.num_features)
        return torch.cat([torch.ones_like(P[:, :1]), norm * torch.cos(P), norm * torch.sin(P)], dim=1), P


def compute_full_RFF_derivatives_LV(model, rff, t):
    """Phi(t) and dPhi/dt, both (T, P).  The input is scalar, so one forward-mode JVP gives all dz/dt."""
    z, z_t = func.jvp(model.get_hidden_output, (t,), (torch.ones_like(t),))
    P, P_t = z @ rff.W, z_t @ rff.W
    norm = math.sqrt(2.0 / rff.num_features)
    Phi = torch.cat([torch.ones_like(P[:, :1]), norm * torch.cos(P), norm * torch.sin(P)], dim=1)
    Phi_t = torch.cat([torch.zeros_like(P[:, :1]), -norm * torch.sin(P) * P_t, norm * torch.cos(P) * P_t], dim=1)
    return Phi, Phi_t


class EndToEndLV(nn.Module):
    """u_{k,i}(t) = Phi(t) @ theta[k,i]: one shared basis, one coefficient block theta_i per agent (and experiment).
    Holds A and w too, so the warm-up can train everything with Adam."""
    def __init__(self, backbone, rff_module, theta, n_edges):
        super().__init__()
        self.backbone, self.rff = backbone, rff_module
        self.theta = nn.Parameter(theta)                                   # (K, N, P, 3)
        self.A = nn.Parameter(torch.zeros(3, 3, device=theta.device, dtype=theta.dtype))
        self.w = nn.Parameter(torch.ones(n_edges, device=theta.device, dtype=theta.dtype))

    def forward(self, t):
        Phi, _ = self.rff(self.backbone, t)
        return torch.einsum('tp,kipa->ktia', Phi, self.theta)             # (K, T, N, 3)
