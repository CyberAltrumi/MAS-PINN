import math
import torch
from physics_residual import rossler_rhs

# MAS structure: agent i owns theta_i (P x 3 per experiment).  The physics Jacobian d r_f / d theta is
# s_f (Phi_t - J_rhs(t) Phi) with J_rhs(t) the (3N x 3N) Jacobian of the right-hand side at every collocation time: the
# 3x3 node-local blocks on the diagonal and, in the x-rows, the edge Laplacian plus the state-dependent three-body coupling.
# J_rhs is obtained by autodiff once per linearisation (vmap over the K*T points); the LM system is never assembled:
# CG only needs H v = J^T (J v), which are einsums with Phi, Phi_t and J_rhs.  Experiments are independent and solved as a batch.
# note: dense (3N)^2 Jacobian per collocation point; switch to node-local + sparse-coupling products if N d grows.


def _scales(data, physics_weight, K, T):
    n_ic, n_obs = K * data['mask_ic'].sum(), K * data['t_obs'].shape[0] * data['mask_obs'].sum()
    s_ic = data['mask_ic'] * (math.sqrt(1 / n_ic) if n_ic else 0.)
    s_obs = data['mask_obs'] * math.sqrt(1 / n_obs)
    return s_ic, s_obs, math.sqrt(physics_weight / (K * T * data['mask_obs'].numel()))


def _linearise(theta, mats, params, data, physics_weight, rgl):
    """Everything J depends on: row scales and the rhs Jacobian at every (k, t)."""
    Phi_ic, Phi_obs, Phi, Phi_t = mats
    K, N, P, _ = theta.shape
    T = Phi.shape[0]
    s_ic, s_obs, s_f = _scales(data, physics_weight, K, T)
    u = torch.einsum('tp,kipa->ktia', Phi, theta)
    f = lambda up: rossler_rhs(up, params, data)
    Jr = torch.func.vmap(torch.func.jacrev(f))(u.reshape(K * T, N, 3)).reshape(K, T, N * 3, N * 3)
    return dict(Phi_ic=Phi_ic, Phi_obs=Phi_obs, Phi=Phi, Phi_t=Phi_t, Jr=Jr, params=params, rgl=math.sqrt(rgl),
                s_ic=s_ic, s_obs=s_obs, s_f=s_f)


def J_vec(v, op):
    """J v for v (K, N, P, 3), as the blocks (IC, observations, physics, rgl)."""
    K, N = v.shape[:2]
    du = torch.einsum('tp,kipa->ktia', op['Phi'], v)
    Jdu = torch.einsum('ktrs,kts->ktr', op['Jr'], du.reshape(K, du.shape[1], -1)).reshape(du.shape)
    return (op['s_ic'] * torch.einsum('p,kipa->kia', op['Phi_ic'][0], v),
            op['s_obs'] * torch.einsum('tp,kipa->ktia', op['Phi_obs'], v),
            op['s_f'] * (torch.einsum('tp,kipa->ktia', op['Phi_t'], v) - Jdu),
            op['rgl'] * v)


def JT_vec(z, op):
    """J^T z, the adjoint of J_vec."""
    z_ic, z_obs, z_f, z_r = z
    z_f = op['s_f'] * z_f
    state = -torch.einsum('ktrs,ktr->kts', op['Jr'], z_f.reshape(*z_f.shape[:2], -1)).reshape(z_f.shape)
    return (torch.einsum('p,kia->kipa', op['Phi_ic'][0], op['s_ic'] * z_ic)
            + torch.einsum('tp,ktia->kipa', op['Phi_obs'], op['s_obs'] * z_obs)
            + torch.einsum('tp,ktia->kipa', op['Phi_t'], z_f) + torch.einsum('tp,ktia->kipa', op['Phi'], state) + op['rgl'] * z_r)


def residual_blocks(theta, op, data):
    """The residual in the block layout of J_vec (absent IC / observation rows are zero through the scales)."""
    u = torch.einsum('tp,kipa->ktia', op['Phi'], theta)
    return (op['s_ic'] * (torch.einsum('p,kipa->kia', op['Phi_ic'][0], theta) - data['x_ic']),
            op['s_obs'] * (torch.einsum('tp,kipa->ktia', op['Phi_obs'], theta) - data['y_obs']),
            op['s_f'] * (torch.einsum('tp,kipa->ktia', op['Phi_t'], theta) - rossler_rhs(u, op['params'], data)),
            op['rgl'] * theta)


def mean_diag_H(op):
    """Exact mean of diag(J^T J) per experiment, (K, 1, 1, 1).  Column (i, p, a) collects the physics rows
    (Phi_t - Jr[., (i,a), (i,a)] Phi on its own row, -Jr[., r, (i,a)] Phi on every other row r) and the IC / observation / rgl rows."""
    Phi, Phi_t, Jr = op['Phi'], op['Phi_t'], op['Jr']
    K, T = Jr.shape[:2]
    N = op['s_obs'].shape[0]
    own = torch.diagonal(Jr, dim1=-2, dim2=-1).reshape(K, T, N, 3)
    colsq = (Jr ** 2).sum(-2).reshape(K, T, N, 3)
    phys = ((Phi_t ** 2).sum(0)[None, None, :, None] - 2 * torch.einsum('ktia,tp->kipa', own, Phi_t * Phi)
            + torch.einsum('ktia,tp->kipa', colsq, Phi ** 2))
    diag = (op['s_f'] ** 2 * phys + (op['s_ic'] ** 2)[None, :, None, :] * (op['Phi_ic'][0] ** 2)[None, None, :, None]
            + (op['s_obs'] ** 2)[None, :, None, :] * (op['Phi_obs'] ** 2).sum(0)[None, None, :, None] + op['rgl'] ** 2)
    return diag.flatten(1).mean(1).view(-1, 1, 1, 1)


CG_ITERS = 100   # truncated-CG steps per LM solve


def cg(apply, b, iters=None, tol=1e-5):
    """Batched conjugate gradient for apply(x) = b, one system per experiment (leading dim), truncated after `iters`."""
    iters = iters or CG_ITERS
    dot = lambda a, c: (a * c).flatten(1).sum(1).view(-1, *[1] * (a.dim() - 1))
    x, r = torch.zeros_like(b), b.clone()
    p, rr = r.clone(), dot(r, r)
    stop = tol ** 2 * dot(b, b)
    for it in range(iters):
        live = rr > stop
        Mp = apply(p)
        a = torch.where(live, rr / dot(p, Mp), torch.zeros_like(rr))
        x, r = x + a * p, r - a * Mp
        rr_new = dot(r, r)
        if (it + 1) % 10 == 0 and not (rr_new > stop).any(): break
        p, rr = r + torch.where(live, rr_new / rr, torch.zeros_like(rr)) * p, rr_new
    return x


def lm_step(theta, mats, params, data, physics_weight=1e-2, damping=1e-3, max_iter=1, rgl=1e-10):
    """Levenberg-Marquardt on all agents' theta, matrix-free.  damping is relative to mean diag(H); a step is kept
    when the actual loss reduction is at least 1e-3 of the predicted one.  Returns (theta, damping, loss)."""
    squares = lambda blocks: sum((b ** 2).sum() for b in blocks).item()
    for _ in range(max_iter):
        op = _linearise(theta, mats, params, data, physics_weight, rgl)
        r = residual_blocks(theta, op, data)
        loss, g = squares(r), JT_vec(r, op)
        mu = mean_diag_H(op)
        for _ in range(8):
            d = cg(lambda v: JT_vec(J_vec(v, op), op) + damping * mu * v, -g)
            cand = theta + d
            loss_new = squares(residual_blocks(cand, op, data))
            predicted = -(2 * (g * d).sum().item() + squares(J_vec(d, op)))
            if predicted > 0 and (loss - loss_new) / predicted > 1e-3:
                theta, loss, damping = cand, loss_new, max(damping / 3, 1e-8)
                break
            damping *= 2
    return theta, damping, loss
