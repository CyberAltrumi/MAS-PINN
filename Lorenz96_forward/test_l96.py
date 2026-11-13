"""python test_l96.py -- torch rhs == numpy rhs and conserves the advection energy; matrix-free J v, J^T z and
mean diag(H) == autograd of the stacked residual; the RFF loss == the LM residual."""
import numpy as np
import torch
from util import l96_rhs_np, generate_L96_dataset
from physics_residual import l96_rhs, L96_rff_loss
from LM_optim import _linearise, J_vec, JT_vec, residual_blocks, mean_diag_H

torch.set_default_dtype(torch.float64)
g = torch.Generator().manual_seed(0)
for N in (4, 5, 40):
    u = 5 * torch.randn(3, N, generator=g)
    assert torch.allclose(l96_rhs(u, 8.), torch.as_tensor(l96_rhs_np(u.numpy(), 8.)), atol=1e-12)
    assert torch.allclose((u * l96_rhs(u, 0.)).sum(-1), -(u ** 2).sum(-1), atol=1e-10)     # sum_i x_i (x_{i+1}-x_{i-2}) x_{i-1} = 0
    data, ref = generate_L96_dataset(N=N, t_end=0.2, N_f=7, n_eval=11, spinup=1.0, device='cpu')
    T, P, pw, rgl = 7, 5, 0.3, 1e-4
    Phi, Phi_t, Phi_ic = (torch.randn(n, P, generator=g) for n in (T, T, 1))
    mats, theta = (Phi_ic, Phi, Phi_t), torch.randn(N, P, generator=g)
    op = _linearise(theta, mats, data, pw, rgl)
    r_full = lambda th: torch.cat([b.reshape(-1) for b in residual_blocks(th, op, data)])
    Jd = torch.func.jacrev(r_full)(theta).reshape(-1, N * P)
    v = torch.randn(N, P, generator=g)
    Jv = torch.cat([b.reshape(-1) for b in J_vec(v, op)])
    assert torch.allclose(Jv, Jd @ v.reshape(-1), atol=1e-10), (Jv - Jd @ v.reshape(-1)).abs().max()
    z = [torch.randn_like(b) for b in J_vec(v, op)]
    assert torch.allclose(JT_vec(z, op).reshape(-1), Jd.T @ torch.cat([b.reshape(-1) for b in z]), atol=1e-10)
    assert torch.allclose(mean_diag_H(op), (Jd ** 2).sum(0).mean(), rtol=1e-12), (mean_diag_H(op), (Jd ** 2).sum(0).mean())
    assert torch.allclose((r_full(theta) ** 2).sum(), L96_rff_loss(theta, *mats, data, pw, rgl)[0], rtol=1e-12)
print('ok: rhs == numpy, advection conserves energy; matrix-free J v, J^T z, mean diag(H) == autograd; loss == |r|^2')
