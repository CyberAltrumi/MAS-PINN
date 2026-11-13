"""python test_rossler.py -- the torch affine form F0 + G theta == the numpy data model;
matrix-free J v, J^T z and mean diag(H) == autograd of the stacked residual; the parameter LS recovers the true theta."""
import numpy as np
import torch
from util import generate_rossler_dataset, rossler_rhs_np
from physics_residual import rossler_rhs
from LM_optim import _linearise, J_vec, JT_vec, residual_blocks, mean_diag_H
from inverse_QP import solve_theta, init_theta

torch.set_default_dtype(torch.float64)


rng = np.random.default_rng(0)
u = rng.uniform(-5, 5, (8, 3))

for fit_local in (True, False):
    data, ref = generate_rossler_dataset(N_obs=20, N_f=12, K=2, n_eval=51, t_end=2.0, fit_local=fit_local, device='cpu')
    ut = torch.as_tensor(u)
    th_np = np.r_[0.2, 0.2, 5.7, ref['theta'].numpy()] if not fit_local else ref['theta'].numpy()
    assert torch.allclose(rossler_rhs(ut, ref['theta'], data), torch.as_tensor(rossler_rhs_np(u, th_np, data['B'].numpy(), data['Tm'].numpy(), data['tri'].numpy())), atol=1e-12)
    assert torch.allclose(rossler_rhs(ut[None, None].expand(2, 3, 8, 3), ref['theta'], data)[1, 2], rossler_rhs(ut, ref['theta'], data))

    # matrix-free operators == autograd of the stacked residual
    K, N, T, P = 2, 8, 12, 5
    g = torch.Generator().manual_seed(0)
    Phi, Phi_t, Phi_ic, Phi_obs = (torch.randn(n, P, generator=g) for n in (T, T, 1, 20))
    theta = torch.rand(K, N, P, 3, generator=g)
    mats = (Phi_ic, Phi_obs, Phi, Phi_t)
    op = _linearise(theta, mats, ref['theta'], data, 1e-2, 1e-10)
    r_full = lambda th: torch.cat([b.reshape(-1) for b in residual_blocks(th, {**op, 'Phi': Phi}, data)])
    Jd = torch.func.jacrev(r_full)(theta).reshape(-1, K * N * P * 3)
    vv = torch.randn(K, N, P, 3, generator=g)
    Jv = torch.cat([b.reshape(-1) for b in J_vec(vv, op)])
    assert torch.allclose(Jv, Jd @ vv.reshape(-1), atol=1e-10), (Jv - Jd @ vv.reshape(-1)).abs().max()
    zz = [torch.randn_like(b) for b in J_vec(vv, op)]
    JTz = JT_vec(zz, op).reshape(-1)
    assert torch.allclose(JTz, Jd.T @ torch.cat([b.reshape(-1) for b in zz]), atol=1e-10)
    H = Jd.T @ Jd
    per_exp = torch.diagonal(H).reshape(K, -1).mean(1)
    assert torch.allclose(mean_diag_H(op).reshape(-1), per_exp, rtol=1e-10), (mean_diag_H(op).reshape(-1), per_exp)

    # Phi = [I 0], Phi_t = [0 I], theta = [u ; u_t]  ->  exact trajectory and derivative: LS must return the true theta
    x = ref['x_exact'][:, :T]
    eye, zero = torch.eye(T), torch.zeros(T, T)
    th = torch.cat([x, rossler_rhs(x, ref['theta'], data)], 1).permute(0, 2, 1, 3)
    est = solve_theta(th, torch.cat([eye, zero], 1), torch.cat([zero, eye], 1), data)
    assert torch.allclose(est, ref['theta'], atol=1e-6), (est - ref['theta']).abs().max()
    th0 = init_theta(data, 2)                                    # constant trajectories: rank-deficient design must not fail
    assert torch.isfinite(solve_theta(th0, torch.ones(T, 2), torch.zeros(T, 2), data)).all()
print('ok: torch affine form == numpy; matrix-free J v, J^T z, mean diag(H) == autograd; LS recovers theta')
