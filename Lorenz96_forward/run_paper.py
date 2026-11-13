"""Lorenz-96 forward problem with MAS-PINN (method key `falm`).
Each trial is saved to <out>/falm_N<N>_F<F>_T<t_end>_s<seed><tag>.npz with the prediction u (t_eval, N), the reference
solution, the relative errors (%) and the wall time.
    python run_paper.py                                      (N = 50)
    python run_paper.py --problem '{"N": 100}'
    python run_paper.py --problem '{"F": 4, "t_end": 2, "N_f": 2000, "n_eval": 2001}' --cfg '{"K_rff": 128}' --tag _K128"""
import argparse, copy, json, os, time
import numpy as np, torch
torch.set_default_dtype(torch.float64)
from util import generate_L96_dataset, init_theta, forward_errors
from network import ModelMix, RandomFourierFeaturesNN, basis_with_derivative
from physics_residual import L96_rff_loss
import LM_optim
from LM_optim import lm_step_L96

PROBLEM = dict(N=50, F=12.0, t_end=1.0, N_f=1000, n_eval=1001, spinup=20.0)   # paper setting (N = 50; N = 100 via --problem)
# sigma=None: 2*pi per time unit (the trunk normalises t to [-1, 1]); physics_weight 1.0 leaves the IC unfitted on half the runs
FALM = dict(hidden=[64, 64, 64], K_rff=64, sigma=None, n_falm=10000, lr_falm=1e-3, sched_step=100, sched_gamma=0.97, lr_min=1e-5,
            lm_iter=1, cg_iters=100, physics_weight=1e-2)
SEEDS = {50: [1, 4, 9, 0, 5], 100: [5, 1, 14, 2, 6]}   # default trials per network size


def run_falm(data, ref, seed, device, cfg=FALM, log_every=500):
    """Shared basis Phi(t) (trunk + RFF), u_i = Phi theta_i.  Outer iteration: one Adam step on the trunk (theta fixed),
    one matrix-free LM step on all theta_i (basis fixed); checkpoint on the training loss (IC + physics)."""
    LM_optim.CG_ITERS = cfg['cg_iters']
    pw = cfg['physics_weight']
    torch.manual_seed(seed)
    backbone = ModelMix(cfg['hidden'], t_max=data['t_end']).to(device)
    rff = RandomFourierFeaturesNN(cfg['hidden'][-1], cfg['K_rff'], sigma=cfg['sigma'] or 2 * np.pi * data['t_end'], rff_device=device)
    mats = lambda: (rff(backbone, data['t_ic'])[0], *basis_with_derivative(backbone, rff, data['t_f']))
    predict = lambda th: rff(backbone, ref['t_eval'])[0] @ th.T                                  # (T_eval, N)
    opt = torch.optim.Adam(backbone.parameters(), lr=cfg['lr_falm'])
    sched = torch.optim.lr_scheduler.StepLR(opt, cfg['sched_step'], cfg['sched_gamma']) if cfg['sched_step'] else None
    with torch.no_grad():                                                                        # fit theta to the initial basis
        theta, damping, _ = lm_step_L96(init_theta(data, 1 + 2 * cfg['K_rff']), mats(), data, pw, 1e-3, max_iter=20)
    best, hist, t0 = dict(loss=float('inf')), [], time.time()
    for epoch in range(1, cfg['n_falm'] + 1):
        opt.zero_grad(); L96_rff_loss(theta, *mats(), data, pw)[0].backward(); opt.step()
        if sched and opt.param_groups[0]['lr'] > cfg['lr_min']: sched.step()
        with torch.no_grad():
            m = mats()
            for _ in range(cfg['lm_iter']):
                theta, damping, _ = lm_step_L96(theta, m, data, pw, damping, 1)
            loss, l_ic, l_phys = (v.item() for v in L96_rff_loss(theta, *m, data, pw))
            if loss < best['loss']:
                best = dict(loss=loss, epoch=epoch, theta=theta.clone(), basis=copy.deepcopy(backbone.state_dict()))
            if log_every and (epoch % log_every == 0 or epoch == cfg['n_falm']):                  # truth: evaluation only
                e = forward_errors(predict(theta), ref); hist.append(dict(epoch=epoch, **e))
                print(f"MAS-PINN {epoch:6d} | loss {loss:.3e} ic {l_ic:.1e} phys {l_phys:.1e} damping {damping:.1e} | "
                      f"traj {e['traj']:.3f}% final {e['final']:.3f}% | {time.time() - t0:.0f}s", flush=True)
    backbone.load_state_dict(best['basis'])
    with torch.no_grad():
        return predict(best['theta']), hist, best['epoch']


def save(path, method, seed, problem, cfg, u, ref, seconds, hist, best_epoch):
    e = forward_errors(u, ref)
    npy = lambda v: v.detach().cpu().numpy()
    np.savez(path, method=method, seed=seed, seconds=seconds, u=npy(u), x_true=npy(ref['x_exact']), t_eval=npy(ref['t_eval'])[:, 0],
             best_epoch=best_epoch, problem=json.dumps(problem), cfg=json.dumps(cfg, default=float),
             hist=np.array([[h['epoch'], h['traj'], h['final']] for h in hist]), **{f'err_{k}': v for k, v in e.items()})
    print(f"{method:8s} N {problem['N']} F {problem['F']:g} T {problem['t_end']:g} seed {seed:5d} | traj {e['traj']:.3f}% "
          f"final {e['final']:.3f}% | best@{best_epoch} {seconds:.0f}s -> {path}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=None, help='trials: initial state (default: SEEDS for the network size)')
    ap.add_argument('--problem', default='{}', help='JSON overrides of PROBLEM')
    ap.add_argument('--cfg', default='{}', help='JSON overrides of FALM')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='results')
    ap.add_argument('--tag', default='', help='file-name suffix (sweeps)')
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args(); torch.set_num_threads(a.threads); os.makedirs(a.out, exist_ok=True)
    problem = {**PROBLEM, **json.loads(a.problem)}
    cfg = {**FALM, **json.loads(a.cfg)}
    device = torch.device(a.device)
    for seed in a.seeds or SEEDS.get(problem['N'], SEEDS[50]):
        data, ref = generate_L96_dataset(**problem, seed=seed, device=device)
        t0 = time.time()
        u, hist, best_epoch = run_falm(data, ref, seed, device, cfg)
        save(os.path.join(a.out, f"falm_N{problem['N']}_F{problem['F']:g}_T{problem['t_end']:g}_s{seed}{a.tag}.npz"),
             'falm', seed, problem, cfg, u, ref, time.time() - t0, hist, best_epoch)
