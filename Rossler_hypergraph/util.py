import numpy as np
import torch
from itertools import combinations
from scipy.integrate import solve_ivp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_AGENTS = 8
A_R, B_R, C_R = 0.2, 0.2, 5.7                                    # Roessler a, b, c
K_EDGE, K_TRI = 0.4, 0.3                                         # pairwise / three-body coupling strengths
EDGES_TRUE = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)]   # generating structure, 0-indexed
TRIS_TRUE = [(0, 1, 2), (1, 3, 4), (4, 5, 6), (5, 6, 7)]


def hypergraph(candidates='all'):
    """Candidate edges / triangles with oriented incidence B (N,E), position membership Tm (3,N,M) and the true weights.
    'all': every pair and triple of the N nodes (topology identification; absent ones have weight 0);
    'true': the generating structure only (weight identification on a known support)."""
    edges = list(combinations(range(N_AGENTS), 2)) if candidates == 'all' else EDGES_TRUE
    tris = list(combinations(range(N_AGENTS), 3)) if candidates == 'all' else TRIS_TRUE
    B, Tm = np.zeros((N_AGENTS, len(edges))), np.zeros((3, N_AGENTS, len(tris)))
    for e, (i, j) in enumerate(edges): B[i, e], B[j, e] = 1., -1.
    for m, (i, j, k) in enumerate(tris): Tm[0, i, m] = Tm[1, j, m] = Tm[2, k, m] = 1.
    w = np.array([K_EDGE if e in EDGES_TRUE else 0. for e in edges])
    v = np.array([K_TRI if t in TRIS_TRUE else 0. for t in tris])
    return edges, tris, B, Tm, w, v


def rossler_rhs_np(u, theta, B, Tm, tri):
    """x' = -y - z - (L x)_i + sum_{(i,j,k)} v (x_j^2 x_k + x_j x_k^2 - 2 x_i^3),  y' = x + a y,  z' = b + z (x - c).
    u (N,3), theta = (a, b, c, w_1..w_E, v_1..v_M), L = B diag(w) B^T.  Numpy reference used to generate the data."""
    E = B.shape[1]
    (a, b, c), w, v = theta[:3], theta[3:3 + E], theta[3 + E:]
    x, y, z = u.T
    xi, xj, xk = x[tri[:, 0]], x[tri[:, 1]], x[tri[:, 2]]
    coup = (-(B * w) @ (B.T @ x) + Tm[0] @ (v * (xj ** 2 * xk + xj * xk ** 2 - 2 * xi ** 3))
            + Tm[1] @ (v * (xi ** 2 * xk + xi * xk ** 2 - 2 * xj ** 3)) + Tm[2] @ (v * (xi ** 2 * xj + xi * xj ** 2 - 2 * xk ** 3)))
    return np.stack([-y - z + coup, x + a * y, b + z * (x - c)], 1)


def solve_rossler(x0, t, theta, B, Tm, tri):
    """High-accuracy numerical solution, x0 (N,3) -> (T,N,3)."""
    rhs = lambda _, s: rossler_rhs_np(s.reshape(-1, 3), theta, B, Tm, tri).ravel()
    sol = solve_ivp(rhs, (0, t[-1]), x0.ravel(), t_eval=t, method='DOP853', rtol=1e-11, atol=1e-13)
    return sol.y.T.reshape(len(t), -1, 3)


def generate_rossler_dataset(N_obs=150, N_f=1600, K=1, obs_stride=1, obs_dims=(0, 1, 2), ic='observed', noise=0.0, t_end=20.0,
                             n_eval=2001, seed=42, candidates='all', fit_local=True, ic_noise=False, device=device):
    """Roessler oscillators on an undirected hypergraph as an affine-parameter inverse problem.
    K experiments (x(0) ~ U(-1,1)) share theta = (a, b, c, w, v).  Agents 0, obs_stride, ... are observed
    in the state components obs_dims at N_obs times in (0, t_end]; ic = 'observed' supplies x(0) of the observed entries,
    'none' nothing, 'all' every entry (oracle).  noise: relative to the std of the clean observations.
    fit_local=False fixes (a, b, c) at the truth, leaving theta = (w, v).
    Returns (data for training, ref for evaluation only)."""
    N = N_AGENTS
    edges, tris, B, Tm, w, v = hypergraph(candidates)
    tri = np.array(tris)
    theta_true = np.r_[A_R, B_R, C_R, w, v]
    x0 = np.stack([np.random.default_rng(seed + 1009 * k).uniform(-1, 1, (N, 3)) for k in range(K)])
    obs = list(range(0, N, obs_stride))
    mask_obs = np.zeros((N, 3), bool); mask_obs[np.ix_(obs, list(obs_dims))] = True
    mask_ic = {'observed': mask_obs, 'none': np.zeros_like(mask_obs), 'all': np.ones_like(mask_obs)}[ic]
    t_obs = np.linspace(0, t_end, N_obs + 1)[1:]
    t_f, t_eval = np.linspace(0, t_end, N_f), np.linspace(0, t_end, n_eval)
    x_exact = np.stack([solve_rossler(x0[k], t_eval, theta_true, B, Tm, tri) for k in range(K)])          # (K,T,N,3)
    clean = torch.as_tensor(np.stack([solve_rossler(x0[k], t_obs, theta_true, B, Tm, tri) for k in range(K)]))
    gen = torch.Generator().manual_seed(seed)
    sd = noise * clean[:, :, torch.as_tensor(mask_obs)].std(unbiased=False)
    y_obs = (clean + sd * torch.randn(clean.shape, generator=gen, dtype=clean.dtype)) * torch.as_tensor(mask_obs)
    x_ic = torch.as_tensor(x0)
    if ic_noise: x_ic = x_ic + sd * torch.randn(x_ic.shape, generator=gen, dtype=x_ic.dtype)
    x_ic = x_ic * torch.as_tensor(mask_ic)
    lo, hi = np.r_[[-10.] * 3, [0.] * (len(edges) + len(tris))], np.r_[[10.] * 3, [1.] * (len(edges) + len(tris))]   # admissible set
    alpha_known = None if fit_local else np.array([A_R, B_R, C_R])
    if not fit_local: theta_true, lo, hi = theta_true[3:], lo[3:], hi[3:]

    T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.get_default_dtype(), device=device)
    data = dict(t_obs=T(t_obs)[:, None], y_obs=T(y_obs), mask_obs=torch.as_tensor(mask_obs, device=device),
                t_ic=T([[0.]]), x_ic=T(x_ic), mask_ic=torch.as_tensor(mask_ic, device=device), t_f=T(t_f)[:, None],
                B=T(B), Tm=T(Tm), tri=torch.as_tensor(tri, dtype=torch.long, device=device),
                alpha_known=None if alpha_known is None else T(alpha_known), lo=T(lo), hi=T(hi),
                obs=obs, obs_dims=list(obs_dims), t_end=t_end)
    ref = dict(t_eval=T(t_eval)[:, None], x_exact=T(x_exact), theta=T(theta_true), x0=T(x0), edges=edges, tris=tris,
               E=len(edges), M=len(tris))
    return data, ref
