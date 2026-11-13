"""MAS-PINN hyper-parameter sweep on the half-observed network LV.
Usage: python sweep_falm.py --out sweep.csv --seeds 42 7 1 --device cuda
Each run: data from the seed (graph, w, x0, noise), warm-up, alternating phase, best checkpoint on observable loss."""
import argparse, copy, csv, itertools, json, os, time
import numpy as np, torch
torch.set_default_dtype(torch.float64)
from util import generate_LV_dataset
from network import ModelMix, RandomFourierFeaturesNN, EndToEndLV, compute_full_RFF_derivatives_LV
from physics_residual import LV_rff_loss
from LM_optim import lm_step_LV
from inverse_QP import init_theta, solve_A_w, inverse_errors

BASE = dict(hidden=[64, 64, 64], K_rff=32, sigma=3 * 2 * np.pi, n_warmup=1000, lr_warmup=1e-3, n_falm=3000, lr_falm=3e-4, lm_iter=1,
            sched_step=0, sched_gamma=0.97, lr_min=1e-5)   # sched_step=0: constant lr
GRID = [{}] + [{k: v} for k, vs in dict(K_rff=[16, 64], sigma=[1 * 2 * np.pi, 6 * 2 * np.pi], hidden=[[64, 64], [128, 128, 128]],
                                         n_warmup=[0], lm_iter=[2], n_falm=[6000], lr_falm=[1e-4, 1e-3]).items() for v in vs]
import os
PROBLEM = dict(N_obs=int(os.environ.get('N_OBS', 161)), N_f=1600, N_agents=8, K=1, obs_stride=2, ic='observed', noise=0.005, D=0.5, t_end=3.0, x0_half=float(os.environ.get('X0_HALF', 0.35)), x0_dominant=float(os.environ.get('X0_DOM', 0.0)))
physics_weight, lr_param = 1e-2, 3e-3


def run(cfg, seed, device):
    physics_weight = cfg.get('physics_weight', 1e-2)
    WB = tuple(cfg.get('w_bounds', (1e-5, 3.0)))
    import LM_optim; LM_optim.CG_ITERS = cfg.get('cg_iters', 30)
    data_full, ref = generate_LV_dataset(**PROBLEM, ic_noise=True, seed=seed, device=device)
    data = data_full
    torch.manual_seed(seed + cfg.get('net_seed_offset', 0))      # restart test: same data, different network init
    backbone = ModelMix(cfg['hidden'], t_max=PROBLEM['t_end']).to(device)
    rff = RandomFourierFeaturesNN(cfg['hidden'][-1], cfg['K_rff'], sigma=cfg['sigma'])
    P = 1 + 2 * cfg['K_rff']
    model = EndToEndLV(backbone, rff, init_theta(data, PROBLEM['N_agents'], P), len(ref['edges'])).to(device)
    mats = lambda: (rff(backbone, data['t_ic'])[0], rff(backbone, data['t_obs'])[0], *compute_full_RFF_derivatives_LV(backbone, rff, data['t_f']))
    t0 = time.time()
    if cfg['n_warmup']:
        opt = torch.optim.Adam([{'params': backbone.parameters(), 'lr': cfg['lr_warmup']}, {'params': [model.theta], 'lr': cfg['lr_warmup']},
                                {'params': [model.A, model.w], 'lr': lr_param}])
        for _ in range(cfg['n_warmup']):
            opt.zero_grad(); LV_rff_loss(model.theta, *mats(), model.A, model.w, data, physics_weight)[0].backward(); opt.step()
            with torch.no_grad(): model.A.clamp_(-2.5, 2.5); model.w.clamp_(1e-5, 3.0)
    theta, A, w = model.theta.detach().clone(), model.A.detach().clone(), model.w.detach().clone()
    best = dict(loss=float('inf')); hist, best_tr = [], dict(traj=float('inf'))
    pw0, pw1 = physics_weight, cfg.get('pw_final', physics_weight)        # geometric ramp of the physics weight (pw_final = pw0: constant)
    stages = cfg.get('t_stages') or [PROBLEM['t_end']]                     # time curriculum: fit [0, t_cut] first, then extend
    for si, t_cut in enumerate(stages):
        final = si == len(stages) - 1
        if t_cut < PROBLEM['t_end']:
            mo, mf = data_full['t_obs'][:, 0] <= t_cut, data_full['t_f'][:, 0] <= t_cut
            data = {**data_full, 't_obs': data_full['t_obs'][mo], 'y_obs': data_full['y_obs'][:, mo], 't_f': data_full['t_f'][mf]}
        else:
            data = data_full
        n_ep = cfg['n_falm'] if final else cfg.get('n_stage', 2000)
        opt = torch.optim.Adam(backbone.parameters(), lr=cfg['lr_falm'])
        sched = torch.optim.lr_scheduler.StepLR(opt, cfg['sched_step'], cfg['sched_gamma']) if cfg['sched_step'] else None
        with torch.no_grad():
            m = mats()
            theta, damping, _ = lm_step_LV(theta, *m, A, w, data, physics_weight, damping=1e-3, max_iter=20)
            A, w = solve_A_w(theta, m[2], m[3], data, w_bounds=WB)
            theta, damping, _ = lm_step_LV(theta, *m, A, w, data, physics_weight, damping, max_iter=5)
        for epoch in range(1, n_ep + 1):
            physics_weight = pw0 * (pw1 / pw0) ** (epoch / n_ep)
            opt.zero_grad(); LV_rff_loss(theta, *mats(), A, w, data, physics_weight)[0].backward(); opt.step()
            if sched and opt.param_groups[0]['lr'] > cfg['lr_min']: sched.step()
            with torch.no_grad():
                m = mats()
                for _ in range(cfg['lm_iter']):
                    theta, damping, _ = lm_step_LV(theta, *m, A, w, data, physics_weight, damping, 1)
                A, w = solve_A_w(theta, m[2], m[3], data, w_bounds=WB)
                loss = LV_rff_loss(theta, *m, A, w, data, pw1)[0].item()            # checkpoint on the final-weight loss
                if final and loss < best['loss']:
                    best = dict(loss=loss, epoch=epoch, theta=theta.clone(), A=A.clone(), w=w.clone(), basis=copy.deepcopy(backbone.state_dict()))
                ev = cfg.get('eval_every', 0)                                        # truth-based error history (evaluation only)
                if final and ev and (epoch % ev == 0 or epoch == n_ep):
                    u_now = torch.einsum('tp,kipa->ktia', rff(backbone, ref['t_eval'])[0], theta)
                    e_now = inverse_errors(u_now, A, w, ref, data); hist.append(dict(epoch=epoch, **e_now))
                    if e_now['traj'] < best_tr['traj']:
                        best_tr = dict(e_now, epoch=epoch, u=u_now[0].cpu().numpy(), A=A.cpu().numpy(), w=w.cpu().numpy())
    backbone.load_state_dict(best['basis'])
    if cfg.get('n_final_lm'):                                               # extra LM / (A,w) sweeps on the fixed best basis
        theta, A, w = best['theta'], best['A'], best['w']
        with torch.no_grad():
            m = mats(); damping = 1e-3
            for _ in range(cfg['n_final_lm']):
                theta, damping, _ = lm_step_LV(theta, *m, A, w, data, pw1, damping, 1)
                A, w = solve_A_w(theta, m[2], m[3], data, w_bounds=WB)
        best.update(theta=theta, A=A, w=w)
    with torch.no_grad():
        u = torch.einsum('tp,kipa->ktia', rff(backbone, ref['t_eval'])[0], best['theta'])
        e = inverse_errors(u, best['A'], best['w'], ref, data)
    npy = lambda v: v.detach().cpu().numpy()
    est = dict(A=npy(best['A']), w=npy(best['w']), u=npy(u[0]), A_true=npy(ref['A']), w_true=npy(ref['w']), x0_true=npy(ref['x0'][0]),
               edges=np.array(ref['edges']), t_eval=npy(ref['t_eval'])[:, 0])
    return dict(**e, best_epoch=best['epoch'], loss=best['loss'], seconds=time.time() - t0, est=est, hist=hist, est_min=best_tr)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='sweep.csv'); ap.add_argument('--seeds', type=int, nargs='+', default=[42, 7, 1])
    ap.add_argument('--device', default='cuda'); ap.add_argument('--configs', type=int, nargs='*', default=None); ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--grid', default=None, help='JSON list of override dicts replacing the built-in GRID')
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    if a.grid: GRID = json.loads(a.grid)
    done = set()
    if os.path.exists(a.out):
        done = {(r['config'], int(r['seed'])) for r in csv.DictReader(open(a.out))}
    for ci, over in enumerate(GRID):
        if a.configs is not None and ci not in a.configs: continue
        cfg = {**BASE, **over}; name = json.dumps(over, default=float) if over else 'base'
        for seed in a.seeds:
            if (name, seed) in done: continue
            r = run(cfg, seed, a.device); r.pop('hist'); r.pop('est_min'); np.savez(f"{a.out[:-4]}_{ci}_{seed}.npz", **r.pop('est'))
            row = dict(config=name, seed=seed, **{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
            new = not os.path.exists(a.out)
            with open(a.out, 'a', newline='') as f:
                wr = csv.DictWriter(f, fieldnames=list(row)); (wr.writeheader() if new else None); wr.writerow(row)
            print(f"{name:40s} seed {seed:4d} | A {r['A']:.2f}% w {r['w']:.2f}% traj {r['traj']:.2f}% hidden {r['hidden']:.2f}% | best@{r['best_epoch']} {r['seconds']:.0f}s", flush=True)
