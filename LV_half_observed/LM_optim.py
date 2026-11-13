import math
import torch
from physics_residual import lv_rhs, laplacian_apply

# MAS structure: agent i owns theta_i (P x 3 per experiment), and the Gauss-Newton system is assembled from
# per-agent blocks: d r_i / d theta_i (own IC, observation and physics rows) plus the coupling blocks
# d r_phys_j / d theta_i = s_f D L_ji Phi, through which the observed agents' equations reach the hidden ones.
# Experiments are independent, so the K systems (N*3P unknowns each) are solved as one batch.
# Solving each agent on its own (block Jacobi) was tried and fails at D = 0.5: it drops exactly those coupling
# blocks, and the hidden agents have no data of their own.  The linear solve is a truncated CG (30 steps), which
# never moves along the near-null directions of a large RFF basis (cond(Phi) ~ 1e15); an exact solve does.
# CG only needs H v = J^T (J v), and J v / J^T z are a few einsums with Phi, Phi_t, the 3x3 local Jacobians and L,
# so lm_step_LV never forms J or H: O(K T N P) per CG step instead of the dense O(K T N^3 P^2) assembly.
# get_manual_Jacobian / get_H_g build the same matrices densely; they are kept as the reference for test_lv.py.


def _scales(data, physics_weight, K, T, N):
    s_ic = torch.zeros(N, device=data['B'].device, dtype=data['B'].dtype)
    s_obs = torch.zeros_like(s_ic)
    if data['x_ic'].numel(): s_ic[data['ic_agents']] = math.sqrt(1 / data['x_ic'].numel())
    s_obs[data['obs']] = math.sqrt(1 / data['y_obs'].numel())
    return s_ic, s_obs, math.sqrt(physics_weight / (K * T * N * 3))


def get_LM_residuals(theta, Phi_ic, Phi_obs, Phi, Phi_t, A, w, data, physics_weight, rgl=1e-10):
    """Per-agent residual vectors (K, N, 3 + 3*T_obs + 3*T + 3P); rows of absent IC / observations are zero."""
    K, N = theta.shape[:2]
    s_ic, s_obs, s_f = _scales(data, physics_weight, K, Phi.shape[0], N)
    x0 = torch.zeros(K, N, 3, device=theta.device, dtype=theta.dtype); x0[:, data['ic_agents']] = data['x_ic']
    y = torch.zeros(K, Phi_obs.shape[0], N, 3, device=theta.device, dtype=theta.dtype); y[:, :, data['obs']] = data['y_obs']
    u = torch.einsum('tp,kipa->ktia', Phi, theta)
    r_ic = s_ic[:, None] * (torch.einsum('p,kipa->kia', Phi_ic[0], theta) - x0)
    r_obs = s_obs[:, None] * (torch.einsum('tp,kipa->ktia', Phi_obs, theta) - y)
    r_f = s_f * (torch.einsum('tp,kipa->ktia', Phi_t, theta) - lv_rhs(u, A, w, data))
    per_agent = lambda r: r.permute(0, 2, 1, 3).reshape(K, N, -1)
    return torch.cat([r_ic, per_agent(r_obs), per_agent(r_f), math.sqrt(rgl) * theta.reshape(K, N, -1)], dim=2)


def get_manual_Jacobian(theta, Phi_ic, Phi_obs, Phi, Phi_t, A, w, data, physics_weight, rgl=1e-10):
    """Block Jacobians d r_i / d theta_i, shape (K, N, rows, 3P)."""
    K, N, P, _ = theta.shape
    T, To = Phi.shape[0], Phi_obs.shape[0]
    s_ic, s_obs, s_f = _scales(data, physics_weight, K, T, N)
    I = torch.eye(3, device=theta.device, dtype=theta.dtype)
    u = torch.einsum('tp,kipa->ktia', Phi, theta)
    L_ii = torch.diagonal(data['B'] @ torch.diag(w) @ data['B'].T)
    # d/dx_b of x_a (g_a + (A x)_a) = delta_ab (g_a + (A x)_a) + x_a A_ab
    J_loc = torch.einsum('ab,ktia->ktiab', I, data['g'] + u @ A.T) + u[..., :, None] * A
    J_ic = s_ic[:, None, None, None] * torch.einsum('p,ab->apb', Phi_ic[0], I)                       # (N,3,P,3)
    J_obs = s_obs[:, None, None, None, None] * torch.einsum('tp,ab->tapb', Phi_obs, I)              # (N,To,3,P,3)
    J_f = s_f * (torch.einsum('tp,ab->tapb', Phi_t, I)[None, :, None]
                 + torch.einsum('a,i,tp,ab->tiapb', data['D'], L_ii, Phi, I)[None]
                 - torch.einsum('ktiab,tp->ktiapb', J_loc, Phi))                                     # (K,T,N,3,P,3)
    J_ic = J_ic.reshape(1, N, 3, 3 * P).expand(K, -1, -1, -1)
    J_obs = J_obs.reshape(1, N, To * 3, 3 * P).expand(K, -1, -1, -1)
    J_f = J_f.permute(0, 2, 1, 3, 4, 5).reshape(K, N, T * 3, 3 * P)
    J_rgl = math.sqrt(rgl) * torch.eye(3 * P, device=theta.device, dtype=theta.dtype).expand(K, N, -1, -1)
    return torch.cat([J_ic, J_obs, J_f, J_rgl], dim=2)


def get_H_g(theta, Phi_ic, Phi_obs, Phi, Phi_t, A, w, data, physics_weight, rgl=1e-10):
    """Full Gauss-Newton system per experiment, built from the agent blocks: H (K,N*3P,N*3P), g (K,N*3P), loss."""
    K, N, P, _ = theta.shape
    T, To = Phi.shape[0], Phi_obs.shape[0]
    r = get_LM_residuals(theta, Phi_ic, Phi_obs, Phi, Phi_t, A, w, data, physics_weight, rgl)
    J_own = get_manual_Jacobian(theta, Phi_ic, Phi_obs, Phi, Phi_t, A, w, data, physics_weight, rgl)
    J = torch.zeros(K, N, J_own.shape[2], N, 3 * P, device=theta.device, dtype=theta.dtype)
    idx = torch.arange(N)
    J[:, idx, :, idx] = J_own.transpose(0, 1)                                         # agent blocks
    s_f = _scales(data, physics_weight, K, T, N)[2]
    L = data['B'] @ torch.diag(w) @ data['B'].T
    L_off = L - torch.diag(torch.diagonal(L))
    I3 = torch.eye(3, device=theta.device, dtype=theta.dtype)
    J[:, :, 3 + 3 * To: 3 + 3 * To + 3 * T] += s_f * torch.einsum(                     # neighbour coupling blocks
        'a,ji,tp,ab->jtaipb', data['D'], L_off, Phi, I3).reshape(N, 3 * T, N, 3 * P)
    J = J.reshape(K, -1, N * 3 * P)
    return J.transpose(1, 2) @ J, (J.transpose(1, 2) @ r.reshape(K, -1, 1))[..., 0], (r ** 2).sum().item()


def _linearise(theta, Phi_ic, Phi_obs, Phi, Phi_t, A, w, data, physics_weight, rgl):
    """Everything J depends on: row scales, the 3x3 local Jacobians at every (k, t, i), and L."""
    K, N = theta.shape[:2]
    s_ic, s_obs, s_f = _scales(data, physics_weight, K, Phi.shape[0], N)
    u = torch.einsum('tp,kipa->ktia', Phi, theta)
    I = torch.eye(3, device=theta.device, dtype=theta.dtype)
    J_loc = torch.einsum('ab,ktia->ktiab', I, data['g'] + u @ A.T) + u[..., :, None] * A
    return dict(Phi_ic=Phi_ic, Phi_obs=Phi_obs, Phi=Phi, Phi_t=Phi_t, J_loc=J_loc, D=data['D'], rgl=math.sqrt(rgl), w=w, nbr=data['nbr'], nbr_e=data['nbr_e'],
                L=data['B'] @ torch.diag(w) @ data['B'].T, s_ic=s_ic[:, None], s_obs=s_obs[:, None], s_f=s_f)   # dense L: mean_diag_H only


def J_vec(v, op):
    """J v for v (K, N, P, 3), as the blocks (IC, observations, physics, rgl)."""
    du = torch.einsum('tp,kipa->ktia', op['Phi'], v)
    return (op['s_ic'] * torch.einsum('p,kipa->kia', op['Phi_ic'][0], v),
            op['s_obs'] * torch.einsum('tp,kipa->ktia', op['Phi_obs'], v),
            op['s_f'] * (torch.einsum('tp,kipa->ktia', op['Phi_t'], v) - torch.einsum('ktiab,ktib->ktia', op['J_loc'], du)
                         + laplacian_apply(du, op['w'], op) * op['D']),
            op['rgl'] * v)


def JT_vec(z, op):
    """J^T z, the adjoint of J_vec (L is symmetric)."""
    z_ic, z_obs, z_f, z_r = z
    z_f = op['s_f'] * z_f
    state = -torch.einsum('ktiab,ktia->ktib', op['J_loc'], z_f) + laplacian_apply(z_f, op['w'], op) * op['D']
    return (torch.einsum('p,kia->kipa', op['Phi_ic'][0], op['s_ic'] * z_ic)
            + torch.einsum('tp,ktia->kipa', op['Phi_obs'], op['s_obs'] * z_obs)
            + torch.einsum('tp,ktia->kipa', op['Phi_t'], z_f) + torch.einsum('tp,ktia->kipa', op['Phi'], state) + op['rgl'] * z_r)


def residual_blocks(theta, op, data):
    """The residual in the block layout of J_vec (absent IC / observation rows are zero)."""
    K, N = theta.shape[:2]
    x0 = torch.zeros(K, N, 3, device=theta.device, dtype=theta.dtype); x0[:, data['ic_agents']] = data['x_ic']
    y = torch.zeros(K, op['Phi_obs'].shape[0], N, 3, device=theta.device, dtype=theta.dtype); y[:, :, data['obs']] = data['y_obs']
    u = torch.einsum('tp,kipa->ktia', op['Phi'], theta)
    rhs = lv_rhs(u, op['A'], op['w'], data)
    return (op['s_ic'] * (torch.einsum('p,kipa->kia', op['Phi_ic'][0], theta) - x0),
            op['s_obs'] * (torch.einsum('tp,kipa->ktia', op['Phi_obs'], theta) - y),
            op['s_f'] * (torch.einsum('tp,kipa->ktia', op['Phi_t'], theta) - rhs), op['rgl'] * theta)


def mean_diag_H(op):
    """Exact mean of diag(J^T J) per experiment, (K, 1, 1, 1).  Column (i, p, a) collects the own physics row
    (Phi_t + (D_a L_ii - J_loc[i,a,a]) Phi), the other species' rows J_loc[i,c,a] Phi, the neighbours' D_a L_ji Phi,
    and the IC / observation / rgl rows."""
    Phi, Phi_t, J, L, D = op['Phi'], op['Phi_t'], op['J_loc'], op['L'], op['D']
    own = torch.diagonal(J, dim1=-2, dim2=-1)                                           # J_loc[..., a, a]
    c = D * torch.diagonal(L)[:, None] - own                                            # (K, T, N, 3)
    q = (J ** 2).sum(-2) - own ** 2 + D ** 2 * ((L ** 2).sum(0) - torch.diagonal(L) ** 2)[:, None]
    phys = ((Phi_t ** 2).sum(0)[None, None, :, None] + 2 * torch.einsum('ktia,tp->kipa', c, Phi_t * Phi)
            + torch.einsum('ktia,tp->kipa', c ** 2 + q, Phi ** 2))
    diag = (op['s_f'] ** 2 * phys + (op['s_ic'] ** 2)[None, :, :, None] * op['Phi_ic'][0][None, None, :, None] ** 2
            + (op['s_obs'] ** 2)[None, :, :, None] * (op['Phi_obs'] ** 2).sum(0)[None, None, :, None] + op['rgl'] ** 2)
    return diag.flatten(1).mean(1).view(-1, 1, 1, 1)


CG_ITERS = 30   # truncated-CG steps per LM solve (sweep knob; 30 = paper setting)


def cg(apply, b, iters=None, tol=1e-5):
    iters = iters or CG_ITERS
    """Batched conjugate gradient for apply(x) = b, one system per experiment (leading dim), truncated after `iters`.
    The convergence test syncs GPU and CPU, so it runs every 10 steps; converged systems are frozen in between."""
    dot = lambda a, c: (a * c).flatten(1).sum(1).view(-1, *[1] * (a.dim() - 1))
    x, r = torch.zeros_like(b), b.clone()
    p, rr = r.clone(), dot(r, r)
    stop = tol ** 2 * dot(b, b)
    for it in range(iters):
        live = rr > stop                                   # converged systems take no further step
        Mp = apply(p)
        a = torch.where(live, rr / dot(p, Mp), torch.zeros_like(rr))
        x, r = x + a * p, r - a * Mp
        rr_new = dot(r, r)
        if (it + 1) % 10 == 0 and not (rr_new > stop).any(): break
        p, rr = r + torch.where(live, rr_new / rr, torch.zeros_like(rr)) * p, rr_new
    return x


def lm_step_LV(theta, Phi_ic, Phi_obs, Phi, Phi_t, A, w, data, physics_weight=1e-2, damping=1e-3, max_iter=1, rgl=1e-10):
    """Levenberg-Marquardt on all agents' theta, matrix-free.  damping is relative to mean diag(H); a step is kept
    when the actual loss reduction is at least 1e-3 of the predicted one.  Returns (theta, damping, loss)."""
    squares = lambda blocks: sum((b ** 2).sum() for b in blocks).item()
    for _ in range(max_iter):
        op = _linearise(theta, Phi_ic, Phi_obs, Phi, Phi_t, A, w, data, physics_weight, rgl)
        op.update(A=A, w=w)
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
