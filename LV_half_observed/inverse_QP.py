import numpy as np
import torch
from scipy.optimize import lsq_linear


def init_theta(data, N, P):
    """Constant trajectories: supplied x(0); every other agent the mean of the supplied ones
    (with no x(0) at all: the first observation).  Uses training data only."""
    known, idx = (data['x_ic'], data['ic_agents']) if data['x_ic'].numel() else (data['y_obs'][:, 0], data['obs'])
    guess = known.mean(1, keepdim=True).expand(-1, N, -1).clone()
    guess[:, idx] = known
    theta = torch.zeros(guess.shape[0], N, P, 3, device=guess.device, dtype=guess.dtype)
    theta[:, :, 0] = guess                                    # basis column 0 is the constant
    return theta


def solve_A_w(theta, Phi, Phi_t, data, A_bound=2.5, w_bounds=(1e-5, 3.0)):
    """Physics residual is linear in (A, w) for a fixed trajectory: box-constrained least squares, one shot.
        u_t - g u  =  u * (A u)  -  D * (B diag(w) B^T u)"""
    u, u_t = torch.einsum('tp,kipa->ktia', Phi, theta), torch.einsum('tp,kipa->ktia', Phi_t, theta)
    I, D, (hi, ti) = torch.eye(3, device=u.device, dtype=u.dtype), data['D'], data['edge_idx']
    E, y = len(hi), u_t - u * data['g']
    # Design matrix (never formed):  M_A[(k,t,i,a),(c,b)] = delta_ac u_a u_b   (column (c,b) = A[c,b]),
    # M_w[(k,t,i,a),e] = -D_a B_ie (B^T u)_{e,a}.  Its normal equations are assembled from edge-local products,
    # O(N_f (N d^3 + |E| d^2 + |E|^2 d)) work and O(|E|^2) memory instead of the (N_f N d) x (9 + |E|) matrix.
    s, ys = u[..., hi, :] - u[..., ti, :], y[..., hi, :] - y[..., ti, :]                          # B^T u, B^T y   (K,T,E,3)
    q = u[..., hi, :, None] * u[..., hi, None, :] - u[..., ti, :, None] * u[..., ti, None, :]       # B^T (u u^T)    (K,T,E,3,3)
    S = torch.einsum('ktic,ktic,ktib,ktid->cbd', u, u, u, u)                                        # sum u_c^2 u_b u_d
    G_AA = torch.einsum('cbd,ce->cbed', S, I).reshape(9, 9)
    G_Aw = -(D[:, None, None] * torch.einsum('ktec,ktecb->cbe', s, q)).reshape(9, E)
    G_ww = (data['B'].T @ data['B']) * torch.einsum('a,ktea,ktfa->ef', D ** 2, s, s)
    G = torch.cat([torch.cat([G_AA, G_Aw], 1), torch.cat([G_Aw.T, G_ww], 1)], 0)
    My = torch.cat([torch.einsum('ktic,ktib,ktic->cb', u, u, y).reshape(9), -torch.einsum('a,ktea,ktea->e', D, s, ys)])
    # only the (9+E) x (9+E) normal system leaves the GPU:  |M x - y|^2 = |R x - R^-T M^T y|^2 + const,  M^T M = R^T R
    R, info = torch.linalg.cholesky_ex(G)
    if info.item():                                          # rank-deficient design, e.g. two hidden agents with identical
        R = torch.linalg.cholesky(G + 1e-10 * torch.diagonal(G).mean() * torch.eye(len(G), device=G.device, dtype=G.dtype))
    R = R.mT                                                 # (constant) initial trajectories: an edge column is zero
    c = torch.linalg.solve_triangular(R.mT, My[:, None], upper=False)
    R, c = R.cpu().numpy(), c[:, 0].cpu().numpy()
    lo, hi = np.array([-A_bound] * 9 + [w_bounds[0]] * E), np.array([A_bound] * 9 + [w_bounds[1]] * E)
    x = np.linalg.solve(R, c)
    if (x < lo).any() or (x > hi).any():                     # bounds active: exact bounded solve
        x = lsq_linear(R, c, bounds=(lo, hi), method='bvls').x
    x = torch.as_tensor(x, device=u.device, dtype=u.dtype)
    return x[:9].reshape(3, 3), x[9:]


def inverse_errors(u_pred, A, w, ref, data):
    """Relative L2 errors in % (evaluation only): A, w, all trajectories, hidden-agent trajectories and x(0)."""
    rel = lambda a, b: 100 * (torch.norm(a - b) / torch.norm(b)).item()
    hidden = [i for i in range(u_pred.shape[2]) if i not in data['obs'].tolist()]
    return dict(A=rel(A, ref['A']), w=rel(w, ref['w']), traj=rel(u_pred, ref['x_exact']),
                hidden=rel(u_pred[:, :, hidden], ref['x_exact'][:, :, hidden]),
                x0_hidden=rel(u_pred[:, 0, hidden], ref['x_exact'][:, 0, hidden]))
