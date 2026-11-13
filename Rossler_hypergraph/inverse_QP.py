import numpy as np
import torch
from scipy.optimize import lsq_linear
from physics_residual import features


def init_theta(data, P):
    """Constant trajectories from training data only: supplied x(0) where given, otherwise the per-component mean of the
    supplied entries (with no x(0) at all: the first observation); components nobody supplies start at 0."""
    known, mask = (data['x_ic'], data['mask_ic']) if data['mask_ic'].any() else (data['y_obs'][:, 0], data['mask_obs'])
    cnt = mask.sum(0).clamp(min=1)                                                 # (3,)
    fill = (known * mask).sum(1, keepdim=True) / cnt                               # (K, 1, 3)
    guess = torch.where(mask, known, fill.expand_as(known))
    theta = torch.zeros(guess.shape[0], guess.shape[1], P, 3, device=guess.device, dtype=guess.dtype)
    theta[:, :, 0] = guess                                                         # basis column 0 is the constant
    return theta


def init_params(data):
    """Admissible starting point for Adam-trained parameters: 0 for (a, b, c), 0.1 for every coupling weight."""
    p = torch.full_like(data['lo'], 0.1)
    if data['alpha_known'] is None: p[:3] = 0.
    return p


def solve_theta(theta, Phi, Phi_t, data):
    """Physics residual is linear in theta for a fixed trajectory: box-constrained least squares, one shot.
        u_t - F0(u) = G(u) theta"""
    u, u_t = torch.einsum('tp,kipa->ktia', Phi, theta), torch.einsum('tp,kipa->ktia', Phi_t, theta)
    F0, G = features(u, data)
    M, y = G.reshape(-1, G.shape[-1]).double(), (u_t - F0).reshape(-1).double()   # float64: M^T M squares cond(M)
    # only the q x q normal system leaves the GPU:  |M x - y|^2 = |R x - R^-T M^T y|^2 + const,  M^T M = R^T R
    Gr = M.T @ M
    R, info = torch.linalg.cholesky_ex(Gr)
    if info.item():                                                                # rank-deficient design (e.g. constant init)
        R = torch.linalg.cholesky(Gr + 1e-10 * torch.diagonal(Gr).mean() * torch.eye(len(Gr), device=Gr.device, dtype=Gr.dtype))
    R = R.mT
    c = torch.linalg.solve_triangular(R.mT, (M.T @ y)[:, None], upper=False)
    R, c = R.cpu().numpy(), c[:, 0].cpu().numpy()
    lo, hi = data['lo'].cpu().numpy(), data['hi'].cpu().numpy()
    x = np.linalg.solve(R, c)
    if (x < lo).any() or (x > hi).any():                                           # bounds active: exact bounded solve
        x = lsq_linear(R, c, bounds=(lo, hi), method='bvls').x
    return torch.as_tensor(x, device=u.device, dtype=u.dtype)


def auc(score, positive):
    """Area under the ROC curve from ranks (ties get the mean rank); nan if one class is empty."""
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if n_pos == 0 or n_neg == 0: return float('nan')
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    for s in np.unique(score): ranks[score == s] = ranks[score == s].mean()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def inverse_errors(u_pred, params, ref, data):
    """Relative L2 errors in % (evaluation only): local (a,b,c), edge weights w, triangle weights v, all trajectories,
    unobserved entries; plus the ROC AUC of ranking the candidate edges / triangles by their estimated weight."""
    rel = lambda a, b: 100 * (torch.norm(a - b) / torch.norm(b)).item()
    E, M, q0 = ref['E'], ref['M'], 0 if data['alpha_known'] is not None else 3
    w, v, w_true, v_true = params[q0:q0 + E], params[q0 + E:], ref['theta'][q0:q0 + E], ref['theta'][q0 + E:]
    hid = ~data['mask_obs']
    out = dict(alpha=rel(params[:3], ref['theta'][:3]) if q0 else float('nan'), w=rel(w, w_true), v=rel(v, v_true),
               traj=rel(u_pred, ref['x_exact']),
               hidden=rel(u_pred[:, :, hid], ref['x_exact'][:, :, hid]) if hid.any() else float('nan'),
               auc2=auc(w.cpu().numpy(), w_true.cpu().numpy() > 0), auc3=auc(v.cpu().numpy(), v_true.cpu().numpy() > 0))
    return out
