import numpy as np
import torch
from scipy.integrate import solve_ivp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def l96_rhs_np(x, F):
    """Lorenz-96 on a ring:  x_i' = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F,  cyclic indices, x (..., N)."""
    return (np.roll(x, -1, -1) - np.roll(x, 2, -1)) * np.roll(x, 1, -1) - x + F


def solve_L96(x0, t, F):
    """High-accuracy numerical solution, x0 (N,) -> (T, N)."""
    sol = solve_ivp(lambda _, x: l96_rhs_np(x, F), (t[0], t[-1]), x0, t_eval=t, method='DOP853', rtol=1e-12, atol=1e-12)
    return sol.y.T


def generate_L96_dataset(N=40, F=8.0, t_end=1.0, N_f=1000, n_eval=1001, spinup=20.0, seed=42, device=device):
    """Forward problem: x(0) of every site is given, the trajectory on [0, t_end] is the unknown.
    x(0): x_i = F + 0.01 N(0,1) (the seed's draw), integrated for `spinup` time units onto the attractor;
    spinup=0 keeps the perturbed equilibrium (an easy, slowly growing wave instead of the attractor).
    Returns (data for training, ref for evaluation only)."""
    x0 = F + 0.01 * np.random.default_rng(seed).standard_normal(N)
    if spinup: x0 = solve_L96(x0, np.array([0., spinup]), F)[-1]
    t_f, t_eval = np.linspace(0, t_end, N_f), np.linspace(0, t_end, n_eval)
    T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.get_default_dtype(), device=device)
    data = dict(t_ic=T([[0.]]), x_ic=T(x0), t_f=T(t_f)[:, None], F=F, t_end=t_end)
    ref = dict(t_eval=T(t_eval)[:, None], x_exact=T(solve_L96(x0, t_eval, F)), x0=T(x0))
    return data, ref


def init_theta(data, P):
    """Constant trajectories u_i(t) = x_i(0): basis column 0 is the constant."""
    theta = data['x_ic'].new_zeros(len(data['x_ic']), P)
    theta[:, 0] = data['x_ic']
    return theta


def forward_errors(u_pred, ref):
    """Relative L2 errors in % (evaluation only): whole trajectory and the final time."""
    rel = lambda a, b: 100 * (torch.norm(a - b) / torch.norm(b)).item()
    return dict(traj=rel(u_pred, ref['x_exact']), final=rel(u_pred[-1], ref['x_exact'][-1]))
