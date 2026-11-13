import random
import numpy as np
import torch
from scipy.integrate import solve_ivp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
A_TRUE = [[0., 1., -1.], [-1., 0., 1.], [1., -1., 0.]]   # cyclic (rock-paper-scissors) LV


def small_world_edges(N, seed):
    """Ring plus N//2 random long-range edges."""
    edges = {tuple(sorted((i, (i + 1) % N))) for i in range(N)}
    cand = [(i, j) for i in range(N) for j in range(i + 1, N) if (i, j) not in edges]
    return sorted(edges | set(random.Random(seed).sample(cand, N // 2)))


def neighbour_table(edges, N, device):
    """Padded neighbour lists: nbr[i, s] = s-th neighbour of site i (padded with i), nbr_e[i, s] = its edge id (padded
    with len(edges), which indexes an appended zero weight).  Lets the Laplacian act by gather + sum: O(N d_max) per time
    point, deterministic (no atomic adds)."""
    lists = [[] for _ in range(N)]
    for e, (i, j) in enumerate(edges): lists[i].append((j, e)); lists[j].append((i, e))
    dmax = max(len(l) for l in lists)
    nbr = [[j for j, _ in l] + [i] * (dmax - len(l)) for i, l in enumerate(lists)]
    nbr_e = [[e for _, e in l] + [len(edges)] * (dmax - len(l)) for l in lists]
    return dict(nbr=torch.tensor(nbr, device=device), nbr_e=torch.tensor(nbr_e, device=device))


def solve_LV(x0, t, A, w, B, D, g):
    """Numerical solution of x_i' = x_i * (g + A x_i) - D * (L x)_i,  L = B diag(w) B^T.  x0 (N,3) -> (T,N,3)."""
    L = B @ np.diag(w) @ B.T
    rhs = lambda _, x: (x.reshape(-1, 3) * (g + x.reshape(-1, 3) @ A.T) - (L @ x.reshape(-1, 3)) * D).ravel()
    sol = solve_ivp(rhs, (0, t[-1]), x0.ravel(), t_eval=t, method='DOP853', rtol=1e-11, atol=1e-13)
    return sol.y.T.reshape(len(t), -1, 3)


def generate_LV_dataset(N_obs=21, N_f=160, N_agents=8, K=3, obs_stride=2, ic='observed', noise=0.005,
                        D=0.05, t_end=3.0, n_eval=301, seed=42, device=device, ic_noise=False, x0_half=0.35, x0_dominant=0.0, param_seed=None):
    """Half-observed network LV inverse problem.
    K experiments (different x(0)) share one graph, one interaction matrix A and one set of edge weights w.
    param_seed: if given, the graph and w come from this seed (fixed physical parameters); x(0) and noise still follow `seed`.
    Agents 0, obs_stride, 2*obs_stride, ... are observed at N_obs times in (0, t_end];
    ic = 'observed' supplies x(0) of the observed agents only, 'none' supplies no x(0), 'all' every agent (oracle).
    noise: relative to the std of the clean observations; ic_noise=True perturbs the supplied x(0) with the same noise level.
    Returns (data for training, ref for evaluation only)."""
    pseed = seed if param_seed is None else param_seed
    edges = small_world_edges(N_agents, pseed)
    B = np.zeros((N_agents, len(edges)))
    for e, (i, j) in enumerate(edges): B[i, e], B[j, e] = 1., -1.
    w = np.random.default_rng(pseed + 104729).uniform(0.6, 1.4, len(edges))
    x0 = np.stack([0.85 + np.random.default_rng(seed + 1009 * k).uniform(-x0_half, x0_half, (N_agents, 3)) for k in range(K)])   # x0_half=0.35: paper draws
    if x0_dominant:                                           # site i starts with species i mod 3 in excess (strong LV + dispersal excitation)
        x0[:, np.arange(N_agents), np.arange(N_agents) % 3] += x0_dominant
    A, Dv, g = np.array(A_TRUE), np.full(3, D), np.zeros(3)

    obs = list(range(0, N_agents, obs_stride))
    ic_agents = {'observed': obs, 'none': [], 'all': list(range(N_agents))}[ic]
    t_obs = np.linspace(0, t_end, N_obs + 1)[1:]
    t_f, t_eval = np.linspace(0, t_end, N_f), np.linspace(0, t_end, n_eval)
    x_exact = np.stack([solve_LV(x0[k], t_eval, A, w, B, Dv, g) for k in range(K)])                 # (K,T,N,3)
    clean = torch.as_tensor(np.stack([solve_LV(x0[k], t_obs, A, w, B, Dv, g) for k in range(K)])[:, :, obs])
    gen = torch.Generator().manual_seed(seed)
    sd = noise * clean.std(unbiased=False)
    y_obs = clean + sd * torch.randn(clean.shape, generator=gen, dtype=clean.dtype)      # same draws as before -> y_obs unchanged
    x_ic = torch.as_tensor(x0[:, ic_agents])
    if ic_noise: x_ic = x_ic + sd * torch.randn(x_ic.shape, generator=gen, dtype=x_ic.dtype)

    T = lambda v: torch.as_tensor(np.asarray(v), dtype=torch.get_default_dtype(), device=device)   # noise drawn in float64, then cast
    data = dict(t_obs=T(t_obs)[:, None], y_obs=T(y_obs), obs=torch.tensor(obs, device=device),
                t_ic=T([[0.]]), x_ic=T(x_ic), ic_agents=torch.tensor(ic_agents, dtype=torch.long, device=device),
                t_f=T(t_f)[:, None], B=T(B), D=T(Dv), g=T(g), t_end=t_end,
                edge_idx=(torch.tensor([i for i, j in edges], device=device), torch.tensor([j for i, j in edges], device=device)),
                **neighbour_table(edges, N_agents, device))
    ref = dict(t_eval=T(t_eval)[:, None], x_exact=T(x_exact), A=T(A), w=T(w), x0=T(x0), edges=edges)
    return data, ref
