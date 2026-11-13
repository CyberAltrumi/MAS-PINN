"""python test_lv.py  -- LM agent Jacobians, gradient and GN matrix == autograd; the matrix-free H v, gradient and
mean diag(H) used by lm_step_LV == the dense ones; parameter QP recovers the true (A, w)."""
import torch
from util import generate_LV_dataset
from physics_residual import lv_rhs
from LM_optim import get_LM_residuals, get_manual_Jacobian, get_H_g, _linearise, J_vec, JT_vec, residual_blocks, mean_diag_H
from inverse_QP import solve_A_w, init_theta

torch.set_default_dtype(torch.float64)          # exactness checks; training may run in float32
data, ref = generate_LV_dataset(N_f=40, device='cpu')
K, N, T, P = 3, 8, 40, 5
g = torch.Generator().manual_seed(0)
Phi, Phi_t, Phi_ic, Phi_obs = (torch.randn(n, P, generator=g) for n in (T, T, 1, 21))
theta = torch.rand(K, N, P, 3, generator=g)
args = (Phi_ic, Phi_obs, Phi, Phi_t, ref['A'], ref['w'], data, 1e-2)
J = get_manual_Jacobian(theta, *args)
J_ad = torch.func.jacrev(lambda th: get_LM_residuals(th, *args))(theta).reshape(K, N, -1, K, N, 3 * P)
blocks = torch.stack([torch.stack([J_ad[k, i, :, k, i] for i in range(N)]) for k in range(K)])
assert torch.allclose(J, blocks, atol=1e-12), (J - blocks).abs().max()
H, gr, loss = get_H_g(theta, *args)
r_full = lambda th: get_LM_residuals(th, *args).reshape(-1)
J_full = torch.func.jacrev(r_full)(theta).reshape(-1, K, N, 3 * P)                 # d r_all / d theta_(k,i)
Jk = J_full.reshape(-1, K, N * 3 * P)                                               # experiments are independent
assert torch.allclose(gr, torch.einsum('r,rkq->kq', r_full(theta), Jk), atol=1e-12)               # exact gradient
assert torch.allclose(H, torch.einsum('rkq,rks->kqs', Jk, Jk), atol=1e-12)                        # exact GN matrix

# matrix-free operators of lm_step_LV == the dense H, g and mean diag(H)
op = _linearise(theta, *args, rgl=1e-10); op.update(A=ref['A'], w=ref['w'])
v = torch.randn(K, N, P, 3, generator=g)
Hv = JT_vec(J_vec(v, op), op).reshape(K, -1)
assert torch.allclose(Hv, (H @ v.reshape(K, -1, 1))[..., 0], atol=1e-10), (Hv - (H @ v.reshape(K, -1, 1))[..., 0]).abs().max()
assert torch.allclose(JT_vec(residual_blocks(theta, op, data), op).reshape(K, -1), gr, atol=1e-12)
assert torch.allclose(mean_diag_H(op).reshape(-1), torch.diagonal(H, dim1=1, dim2=2).mean(-1), rtol=1e-12)

# Phi = [I 0], Phi_t = [0 I] and theta = [u ; u_t]  ->  u and u_t are the exact trajectory and its derivative
u = ref['x_exact'][:, ::(301 - 1) // 50][:, :T]
eye, zero = torch.eye(T), torch.zeros(T, T)
theta = torch.cat([u, lv_rhs(u, ref['A'], ref['w'], data)], 1).permute(0, 2, 1, 3)
A, w = solve_A_w(theta, torch.cat([eye, zero], 1), torch.cat([zero, eye], 1), data)
assert torch.allclose(A, ref['A'], atol=1e-8) and torch.allclose(w, ref['w'], atol=1e-8), ((A - ref['A']).abs().max(), (w - ref['w']).abs().max())
th0 = init_theta(data, N, 2)                                  # constant trajectories -> rank-deficient design: must not fail
A0, w0 = solve_A_w(th0, torch.ones(T, 2), torch.zeros(T, 2), data)
assert torch.isfinite(A0).all() and torch.isfinite(w0).all()
print('ok: LM agent blocks, gradient and GN matrix == autograd; matrix-free H v, gradient, mean diag(H) == dense; QP recovers the true A, w')
