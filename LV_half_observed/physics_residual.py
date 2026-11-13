import torch


def laplacian_apply(u, w, data):
    """(B diag(w) B^T) u = sum_{e=(i,j)} w_e (u_i - u_j) through the padded neighbour table (util.neighbour_table):
    O(N d_max) per time point instead of the dense N x N product, and deterministic."""
    nbr, nbr_e = data['nbr'], data['nbr_e']
    wp = torch.cat([w, w.new_zeros(1)])[nbr_e]                                    # (N, d_max), zero on padding
    return (wp[:, :, None] * (u[..., :, None, :] - u[..., nbr, :])).sum(-2)


def lv_rhs(u, A, w, data):
    """x_i' = x_i * (g + A x_i) - D * (L x)_i,  L = B diag(w) B^T;  u (..., N, 3)."""
    return u * (data['g'] + u @ A.T) - laplacian_apply(u, w, data) * data['D']


def LV_rff_loss(theta, Phi_ic, Phi_obs, Phi, Phi_t, A, w, data, physics_weight=1e-2, rgl=0):
    """FALM loss with u = Phi @ theta:  IC (supplied agents) + observations (observed agents) + physics + rgl*|theta|^2.
    rgl keeps theta bounded along the near-null directions of a large RFF basis (cond(Phi) ~ 1e15 at P = 65)."""
    u_ic = torch.einsum('tp,kipa->ktia', Phi_ic, theta)[:, 0, data['ic_agents']]
    u_obs = torch.einsum('tp,kipa->ktia', Phi_obs, theta)[:, :, data['obs']]
    u = torch.einsum('tp,kipa->ktia', Phi, theta)
    u_t = torch.einsum('tp,kipa->ktia', Phi_t, theta)
    term_ic = ((u_ic - data['x_ic']) ** 2).mean() if u_ic.numel() else u.new_zeros(())
    term_obs = ((u_obs - data['y_obs']) ** 2).mean()
    term_phys = ((u_t - lv_rhs(u, A, w, data)) ** 2).mean()
    return term_ic + term_obs + physics_weight * term_phys + rgl * (theta ** 2).sum(), term_ic, term_obs, term_phys
