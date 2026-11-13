"""GPU wall-clock and peak-memory benchmark on the half-observed network LV (final configuration, seed 2026, 0.5% noise).
Per 100 outer iterations:  MAS-PINN matrix-free LM with 30 / 60 / 100 CG steps;  the same loop with a dense LM step
(explicit J, normal matrix, Cholesky; LM_optim.get_H_g).
    python bench_time.py [--epochs 100] [--seed 2026]        -> prints a table, writes results/timing.json"""
import argparse, json, math, time, os
import torch
torch.set_default_dtype(torch.float64)
import sweep_falm as sf, run_paper as rp, LM_optim
from LM_optim import get_H_g, get_LM_residuals

dev = torch.device('cuda')
sync = torch.cuda.synchronize


def lm_step_dense(theta, Phi_ic, Phi_obs, Phi, Phi_t, A, w, data, physics_weight, damping=1e-3, max_iter=1, rgl=1e-10):
    """lm_step_LV with the linear solve replaced by explicit assembly of J^T J (get_H_g) and a Cholesky solve."""
    for _ in range(max_iter):
        H, g, loss = get_H_g(theta, Phi_ic, Phi_obs, Phi, Phi_t, A, w, data, physics_weight, rgl)
        mu = torch.diagonal(H, dim1=1, dim2=2).mean(1).view(-1, 1, 1)
        I = torch.eye(H.shape[1], device=H.device, dtype=H.dtype)
        for _ in range(8):
            d = -torch.cholesky_solve(g[..., None], torch.linalg.cholesky(H + damping * mu * I))[..., 0]
            cand = theta + d.view_as(theta)
            loss_new = (get_LM_residuals(cand, Phi_ic, Phi_obs, Phi, Phi_t, A, w, data, physics_weight, rgl) ** 2).sum().item()
            predicted = -(2 * (g * d).sum().item() + (d[:, None, :] @ H @ d[..., None]).sum().item())
            if predicted > 0 and (loss - loss_new) / predicted > 1e-3:
                theta, loss, damping = cand, loss_new, max(damping / 3, 1e-8)
                break
            damping *= 2
    return theta, damping, loss


def dense_step_floored(theta, *rest, floor=1e-2):
    """lm_step_dense with a relative damping floor: the exact solve is unstable along the near-null directions of the
    RFF basis without it (diverges within 100 outer iterations); the floor leaves time and memory unchanged."""
    return lm_step_dense(*((theta,) + rest[:-2] + (max(rest[-2], floor), rest[-1])))


def falm_loop(state, cfg, epochs, lm_step, cg_iters=None):
    """`epochs` outer iterations of the sweep_falm.run alternating phase from a copied state; returns per-part seconds + peak memory."""
    LM_optim.CG_ITERS = cg_iters or 30
    backbone, rff, data, theta, A, w, damping = state['backbone'], state['rff'], state['data'], state['theta'].clone(), state['A'].clone(), state['w'].clone(), state['damping']
    backbone.load_state_dict(state['backbone_sd'])
    mats = lambda: (rff(backbone, data['t_ic'])[0], rff(backbone, data['t_obs'])[0], *sf.compute_full_RFF_derivatives_LV(backbone, rff, data['t_f']))
    opt = torch.optim.Adam(backbone.parameters(), lr=cfg['lr_falm'])
    pw, WB = cfg['physics_weight'], tuple(cfg['w_bounds'])
    parts = dict(adam=0., basis=0., lm=0., params=0., loss=0.); lm_peak = peak = 0
    for epoch in range(epochs + 5):                       # 5 untimed warm-up iterations
        timed = epoch >= 5
        if epoch == 5: sync(); torch.cuda.reset_peak_memory_stats(); t_all = time.time()
        if timed: peak = max(peak, torch.cuda.max_memory_allocated()); torch.cuda.reset_peak_memory_stats()
        t = time.time(); opt.zero_grad(); sf.LV_rff_loss(theta, *mats(), A, w, data, pw)[0].backward(); opt.step(); sync()
        if timed: parts['adam'] += time.time() - t
        with torch.no_grad():
            t = time.time(); m = mats(); sync(); parts['basis'] += (time.time() - t) * timed
            t = time.time(); torch.cuda.reset_peak_memory_stats(); base_mem = torch.cuda.memory_allocated()
            theta, damping, _ = lm_step(theta, *m, A, w, data, pw, damping, 1); sync(); parts['lm'] += (time.time() - t) * timed
            lm_peak = max(lm_peak, torch.cuda.max_memory_allocated() - base_mem) if timed else 0; peak = max(peak, torch.cuda.max_memory_allocated())
            t = time.time(); A, w = sf.solve_A_w(theta, m[2], m[3], data, w_bounds=WB); sync(); parts['params'] += (time.time() - t) * timed
            t = time.time(); loss = sf.LV_rff_loss(theta, *m, A, w, data, pw)[0].item(); parts['loss'] += (time.time() - t) * timed
    sync(); peak = max(peak, torch.cuda.max_memory_allocated())
    return dict(total=time.time() - t_all, **parts, peak_mem_GB=peak / 1e9, lm_peak_GB=lm_peak / 1e9, final_loss=loss, epochs=epochs)


def build_state(cfg, seed=2026, noise=0.005, N_agents=None, N_f=None):
    """Data, trunk, RFF basis and the initial coefficient solve of sweep_falm.run; returns the common starting state."""
    sf.PROBLEM.update(rp.PROBLEM, noise=noise)
    if N_agents: sf.PROBLEM['N_agents'] = N_agents
    if N_f: sf.PROBLEM['N_f'] = N_f
    data, ref = sf.generate_LV_dataset(**sf.PROBLEM, ic_noise=True, seed=seed, device=dev)
    torch.manual_seed(seed)
    backbone = sf.ModelMix(cfg['hidden'], t_max=sf.PROBLEM['t_end']).to(dev)
    rff = sf.RandomFourierFeaturesNN(cfg['hidden'][-1], cfg['K_rff'], sigma=cfg['sigma'])
    P = 1 + 2 * cfg['K_rff']
    model = sf.EndToEndLV(backbone, rff, sf.init_theta(data, sf.PROBLEM['N_agents'], P), len(ref['edges'])).to(dev)
    theta, A, w = model.theta.detach().clone(), model.A.detach().clone(), model.w.detach().clone()
    mats = lambda: (rff(backbone, data['t_ic'])[0], rff(backbone, data['t_obs'])[0], *sf.compute_full_RFF_derivatives_LV(backbone, rff, data['t_f']))
    LM_optim.CG_ITERS = cfg['cg_iters']
    with torch.no_grad():                                 # the initial coefficient solve of sweep_falm.run (common starting state)
        m = mats()
        theta, damping, _ = sf.lm_step_LV(theta, *m, A, w, data, cfg['physics_weight'], damping=1e-3, max_iter=20)
        A, w = sf.solve_A_w(theta, m[2], m[3], data, w_bounds=tuple(cfg['w_bounds']))
        theta, damping, _ = sf.lm_step_LV(theta, *m, A, w, data, cfg['physics_weight'], damping, max_iter=5)
    return dict(backbone=backbone, rff=rff, data=data, ref=ref, theta=theta, A=A, w=w, damping=damping, mats=m, edges=len(ref['edges']),
                backbone_sd={k: v.clone() for k, v in backbone.state_dict().items()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--epochs', type=int, default=100); ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--noise', type=float, default=0.005); a = ap.parse_args()
    cfg = {**sf.BASE, **rp.FALM, 'physics_weight': 1e-2}
    state = build_state(cfg, a.seed, a.noise); theta, A, w, m, data, ref = state['theta'], state['A'], state['w'], state['mats'], state['data'], state['ref']
    n = theta.numel(); rows = sum(r.numel() for r in LM_optim.residual_blocks(theta, {**LM_optim._linearise(theta, *m, A, w, data, cfg['physics_weight'], 1e-10), 'A': A, 'w': w}, data))
    print(f"seed {a.seed}, noise {a.noise:g}: coefficient dimension PNd = {n}, residual rows = {rows}; {a.epochs} timed outer iterations each", flush=True)
    out = {}
    for k in (30, 60, 100):
        out[f'falm_cg{k}'] = r = falm_loop(state, cfg, a.epochs, sf.lm_step_LV, cg_iters=k)
        print(f"MAS-PINN  CG {k:3d}: {r['total']:7.2f} s /{a.epochs} it  (adam {r['adam']:.2f}, basis {r['basis']:.2f}, LM {r['lm']:.2f}, params {r['params']:.2f}, loss {r['loss']:.2f})  peak {r['peak_mem_GB']:.2f} GB  loss {r['final_loss']:.3e}", flush=True)
    out['dense'] = r = falm_loop(state, cfg, a.epochs, dense_step_floored)
    print(f"dense LM (J^T J + Cholesky): {r['total']:7.2f} s /{a.epochs} it  (adam {r['adam']:.2f}, basis {r['basis']:.2f}, LM {r['lm']:.2f}, params {r['params']:.2f}, loss {r['loss']:.2f})  peak {r['peak_mem_GB']:.2f} GB  loss {r['final_loss']:.3e}", flush=True)
    out.update(seed=a.seed, noise=a.noise, epochs=a.epochs, gpu=torch.cuda.get_device_name(0), PNd=n, rows=rows)
    os.makedirs('results', exist_ok=True); json.dump(out, open('results/timing.json', 'w'), indent=1)
    print("written results/timing.json")
