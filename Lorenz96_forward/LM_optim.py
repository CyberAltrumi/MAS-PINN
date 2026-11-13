import math
import torch
from physics_residual import l96_rhs

# MAS structure: site i owns theta_i (P coefficients, d = 1).  The physics Jacobian d r_f / d theta is
# s_f (Phi_t - J_rhs(t) Phi), and J_rhs is the cyclic 4-banded ring stencil of Lorenz-96:
#   d f_i / d x_{i+1} = x_{i-1} = -d f_i / d x_{i-2},   d f_i / d x_{i-1} = x_{i+1} - x_{i-2},   d f_i / d x_i = -1,
# so J v and J^T z are a few rolls and two products with Phi / Phi_t: O(T N P) per CG step, J and H are never formed.


def _scales(N, T, physics_weight):
    return math.sqrt(1 / N), math.sqrt(physics_weight / (T * N))


def _linearise(theta, mats, data, physics_weight, rgl):
    """Everything J depends on: row scales and the two stencil coefficients a = x_{i-1}, b = x_{i+1} - x_{i-2} at every t."""
    Phi_ic, Phi, Phi_t = mats
    u = Phi @ theta.T                                                                   # (T, N)
    s_ic, s_f = _scales(theta.shape[0], Phi.shape[0], physics_weight)
    return dict(Phi_ic=Phi_ic, Phi=Phi, Phi_t=Phi_t, a=u.roll(1, -1), b=u.roll(-1, -1) - u.roll(2, -1),
                s_ic=s_ic, s_f=s_f, rgl=math.sqrt(rgl), F=data['F'])


def jac_rhs(du, op):
    """(d rhs / d u) du, du (T, N)."""
    return op['a'] * (du.roll(-1, -1) - du.roll(2, -1)) + op['b'] * du.roll(1, -1) - du


def jac_rhs_T(z, op):
    """(d rhs / d u)^T z, the adjoint of jac_rhs."""
    za, zb = z * op['a'], z * op['b']
    return za.roll(1, -1) - za.roll(-2, -1) + zb.roll(-1, -1) - z


def J_vec(v, op):
    """J v for v (N, P), as the blocks (IC, physics, rgl)."""
    return (op['s_ic'] * (op['Phi_ic'] @ v.T)[0],
            op['s_f'] * (op['Phi_t'] @ v.T - jac_rhs(op['Phi'] @ v.T, op)),
            op['rgl'] * v)


def JT_vec(z, op):
    """J^T z, the adjoint of J_vec."""
    z_ic, z_f, z_r = z
    z_f = op['s_f'] * z_f
    return (op['s_ic'] * z_ic[:, None] * op['Phi_ic'] + z_f.T @ op['Phi_t'] - jac_rhs_T(z_f, op).T @ op['Phi'] + op['rgl'] * z_r)


def residual_blocks(theta, op, data):
    """The residual in the block layout of J_vec."""
    u = op['Phi'] @ theta.T
    return (op['s_ic'] * ((op['Phi_ic'] @ theta.T)[0] - data['x_ic']),
            op['s_f'] * (op['Phi_t'] @ theta.T - l96_rhs(u, op['F'])),
            op['rgl'] * theta)


def mean_diag_H(op):
    """Exact mean of diag(J^T J).  Column (i, p) has the own physics row Phi_t + Phi (d f_i / d x_i = -1), the rows of
    sites i-1, i+2, i+1 with -x_{i-2}, x_{i+1}, -(x_{i+2} - x_{i-1}) times Phi (distinct rows for N >= 4), the IC and rgl rows."""
    Phi, Phi_t, a, b = op['Phi'], op['Phi_t'], op['a'], op['b']
    off = a.roll(1, -1) ** 2 + a.roll(-2, -1) ** 2 + b.roll(-1, -1) ** 2                 # (T, N)
    phys = ((Phi_t + Phi) ** 2).sum(0)[None, :] + off.T @ Phi ** 2                      # (N, P)
    return (op['s_f'] ** 2 * phys + op['s_ic'] ** 2 * op['Phi_ic'] ** 2 + op['rgl'] ** 2).mean()


CG_ITERS = 100   # truncated-CG steps per LM solve


def cg(apply, b, iters=None, tol=1e-5):
    """Conjugate gradient for apply(x) = b, truncated after `iters`; the convergence test syncs GPU and CPU, so every 10 steps."""
    iters = iters or CG_ITERS
    x, r = torch.zeros_like(b), b.clone()
    p, rr = r.clone(), (r * r).sum()
    stop = tol ** 2 * rr
    for it in range(iters):
        Mp = apply(p)
        a = rr / (p * Mp).sum()
        x, r = x + a * p, r - a * Mp
        rr_new = (r * r).sum()
        if (it + 1) % 10 == 0 and rr_new <= stop: break
        p, rr = r + rr_new / rr * p, rr_new
    return x


def lm_step_L96(theta, mats, data, physics_weight=1.0, damping=1e-3, max_iter=1, rgl=1e-10):
    """Levenberg-Marquardt on all sites' theta, matrix-free.  damping is relative to mean diag(H); a step is kept
    when the actual loss reduction is at least 1e-3 of the predicted one.  Returns (theta, damping, loss)."""
    squares = lambda blocks: sum((b ** 2).sum() for b in blocks).item()
    for _ in range(max_iter):
        op = _linearise(theta, mats, data, physics_weight, rgl)
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
