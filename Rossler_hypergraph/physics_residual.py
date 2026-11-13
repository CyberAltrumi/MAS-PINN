import torch


def features(u, data):
    """Affine structure of the Roessler hypergraph dynamics:  rhs(u) = F0(u) + G(u) theta,  u (..., N, 3).
    theta = (a, b, c, w_1..w_E, v_1..v_M); with data['alpha_known'] set, (a, b, c) move into F0 and theta = (w, v).
    Only the x-equation carries coupling: column e of G is x_j - x_i at node i (x_i - x_j at j) for edge (i, j), and
    column m is x_j^2 x_k + x_j x_k^2 - 2 x_i^3 at node i (cyclically at j, k) for triangle (i, j, k)."""
    x, y, z = u.unbind(-1)
    B, Tm, (ti, tj, tk) = data['B'], data['Tm'], data['tri'].unbind(1)
    xi, xj, xk = x[..., ti], x[..., tj], x[..., tk]
    edge_cols = -B * (x @ B)[..., None, :]                                                             # (..., N, E)
    tri_cols = (Tm[0] * (xj ** 2 * xk + xj * xk ** 2 - 2 * xi ** 3)[..., None, :]
                + Tm[1] * (xi ** 2 * xk + xi * xk ** 2 - 2 * xj ** 3)[..., None, :]
                + Tm[2] * (xi ** 2 * xj + xi * xj ** 2 - 2 * xk ** 3)[..., None, :])                     # (..., N, M)
    Gx = torch.cat([edge_cols, tri_cols], -1)
    zero, one = torch.zeros_like(x), torch.ones_like(x)
    F0 = torch.stack([-y - z, x, z * x], -1)
    if data['alpha_known'] is None:
        Z = lambda n: zero[..., None].expand(*x.shape, n)
        G = torch.stack([torch.cat([Z(3), Gx], -1),
                         torch.cat([y[..., None], Z(2 + Gx.shape[-1])], -1),
                         torch.cat([Z(1), one[..., None], -z[..., None], Z(Gx.shape[-1])], -1)], -2)    # (..., N, 3, q)
    else:
        a, b, c = data['alpha_known']
        F0 = F0 + torch.stack([zero, a * y, b - c * z], -1)
        G = torch.stack([Gx, torch.zeros_like(Gx), torch.zeros_like(Gx)], -2)
    return F0, G


def rossler_rhs(u, theta, data):
    F0, G = features(u, data)
    return F0 + G @ theta


def rff_loss(theta, Phi_ic, Phi_obs, Phi, Phi_t, params, data, physics_weight=1e-2, rgl=0):
    """FALM loss with u = Phi @ theta: IC (supplied entries) + observations (observed entries) + physics + rgl*|theta|^2."""
    u_ic = torch.einsum('p,kipa->kia', Phi_ic[0], theta)
    u_obs = torch.einsum('tp,kipa->ktia', Phi_obs, theta)
    u = torch.einsum('tp,kipa->ktia', Phi, theta)
    u_t = torch.einsum('tp,kipa->ktia', Phi_t, theta)
    n_ic = u_ic.shape[0] * data['mask_ic'].sum()
    term_ic = ((data['mask_ic'] * (u_ic - data['x_ic'])) ** 2).sum() / n_ic if n_ic else u.new_zeros(())
    term_obs = ((data['mask_obs'] * (u_obs - data['y_obs'])) ** 2).sum() / (u_obs.shape[0] * u_obs.shape[1] * data['mask_obs'].sum())
    term_phys = ((u_t - rossler_rhs(u, params, data)) ** 2).mean()
    return term_ic + term_obs + physics_weight * term_phys + rgl * (theta ** 2).sum(), term_ic, term_obs, term_phys
