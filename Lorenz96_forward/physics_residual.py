import torch


def l96_rhs(u, F):
    """x_i' = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F,  u (..., N), cyclic indices (torch.roll(u, s)[i] = u[i - s])."""
    return (u.roll(-1, -1) - u.roll(2, -1)) * u.roll(1, -1) - u + F


def L96_rff_loss(theta, Phi_ic, Phi, Phi_t, data, physics_weight=1.0, rgl=0):
    """MAS-PINN loss with u_i = Phi @ theta_i (theta (N, P)):  IC + physics + rgl*|theta|^2."""
    u, u_t = Phi @ theta.T, Phi_t @ theta.T                                           # (T, N)
    term_ic = ((Phi_ic @ theta.T)[0] - data['x_ic']).pow(2).mean()
    term_phys = (u_t - l96_rhs(u, data['F'])).pow(2).mean()
    return term_ic + physics_weight * term_phys + rgl * (theta ** 2).sum(), term_ic, term_phys
