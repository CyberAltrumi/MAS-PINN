"""Paper runs of MAS-PINN (method key `falm`) on the half-observed network LV.
Each trial is saved to <out>/falm_s<seed>_n<noise>.npz with the reconstruction u (t_eval, N, 3), A, w, the truths, the
observations, the relative errors (%) and the wall time.
    python run_paper.py --noise 0.005 --device cuda          (noise levels of the paper: 0.005, 0.01, 0.03)
    python run_paper.py --seeds 0 --import sweep_0_0.npz      (reuse a sweep_falm.py run)
Then: python ../plotting/plot_nature.py --results results --out ../figures --error-units percent"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import argparse, time
import numpy as np, torch
torch.set_default_dtype(torch.float64)
from util import generate_LV_dataset
from inverse_QP import inverse_errors

PROBLEM = dict(N_obs=161, N_f=1600, N_agents=8, K=1, obs_stride=2, ic='observed', D=0.5, t_end=3.0)
W_BOUNDS = (0.3, 3.0)                                   # admissible set of the edge weights
FALM = dict(n_warmup=0, K_rff=64, n_falm=10000, sched_step=100, lr_falm=1e-3, cg_iters=100, w_bounds=list(W_BOUNDS))
SEEDS = {0.005: [2026, 2025, 100, 8, 42], 0.01: [2026, 8, 100, 0, 5], 0.03: [2026, 100, 8, 0, 3]}   # default trials per noise level


def run_falm(data, ref, seed, device, noise):
    """sweep_falm.run with the final configuration."""
    import sweep_falm
    sweep_falm.PROBLEM.update(PROBLEM, noise=noise)
    r = sweep_falm.run({**sweep_falm.BASE, **FALM, 'eval_every': 100}, seed, device)
    T = lambda a: torch.as_tensor(a, device=device)
    return T(r['est']['u'])[None], T(r['est']['A']), T(r['est']['w']), r['hist'], r['est_min']


KEYS = ('A', 'w', 'traj', 'hidden', 'x0_hidden')


def save(path, method, seed, noise, u, A, w, ref, data, seconds, hist, snap):
    e = inverse_errors(u, A, w, ref, data)                                     # final / loss-selected iterate
    mins = {k: min([h[k] for h in hist] + [e[k]]) for k in KEYS}                # smallest value reached during training
    npy = lambda v: v.detach().cpu().numpy()
    snap = snap or dict(epoch=-1, u=npy(u[0]), A=npy(A), w=npy(w))
    np.savez(path, method=method, seed=seed, noise=noise, seconds=seconds, u=npy(u[0]), A=npy(A), w=npy(w),
             u_min=snap['u'], A_min=snap['A'], w_min=snap['w'], epoch_min=snap['epoch'],
             hist=np.array([[h['epoch']] + [h[k] for k in KEYS] for h in hist]) if hist else np.zeros((0, 1 + len(KEYS))),
             x_true=npy(ref['x_exact'][0]), A_true=npy(ref['A']), w_true=npy(ref['w']), t_eval=npy(ref['t_eval'])[:, 0],
             t_obs=npy(data['t_obs'])[:, 0], y_obs=npy(data['y_obs'][0]), obs=npy(data['obs']), edges=np.array(ref['edges']),
             **{f'err_{k}': v for k, v in e.items()}, **{f'min_{k}': v for k, v in mins.items()})
    print(f"{method:8s} seed {seed:5d} noise {noise:g} | final A {e['A']:.2f}% w {e['w']:.2f}% hidden {e['hidden']:.2f}% | "
          f"min A {mins['A']:.2f}% w {mins['w']:.2f}% hidden {mins['hidden']:.2f}% | {seconds:.0f}s -> {path}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=None, help='trials: x(0) and noise realisation (default: SEEDS for --noise)')
    ap.add_argument('--noise', type=float, default=0.005, help='relative measurement noise; the paper uses 0.005, 0.01 and 0.03')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='results')
    ap.add_argument('--import', dest='imports', nargs='*', default=[], help='sweep_falm.py npz files (…_<seed>.npz) to reuse instead of training')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=None, help='override the training length (smoke tests)')
    ap.add_argument('--param-seed', type=int, default=2026, help='the graph and the edge weights w of every trial are those of this seed; negative: each trial draws its own')
    ap.add_argument('--sigma', type=float, default=None, help='RFF bandwidth sigma (default sweep_falm.BASE, 6*pi)')
    a = ap.parse_args(); torch.set_num_threads(a.threads); os.makedirs(a.out, exist_ok=True)
    if a.epochs: FALM['n_falm'] = a.epochs
    if a.sigma is not None: FALM['sigma'] = a.sigma
    if a.param_seed >= 0: PROBLEM['param_seed'] = a.param_seed              # propagates to sweep_falm.PROBLEM via run_falm
    device = torch.device(a.device)
    for seed in a.seeds or SEEDS.get(a.noise, SEEDS[0.005]):
        data, ref = generate_LV_dataset(**PROBLEM, noise=a.noise, ic_noise=True, seed=seed, device=device)
        t0 = time.time()
        if a.imports:
            f = next(p for p in a.imports if p.endswith(f'_{seed}.npz')); z = np.load(f)
            T = lambda k: torch.as_tensor(z[k], device=device)
            u, A, w, seconds, hist, snap = T('u')[None], T('A'), T('w'), float('nan'), [], None
        else:
            u, A, w, hist, snap = run_falm(data, ref, seed, device, a.noise); seconds = time.time() - t0
        save(os.path.join(a.out, f'falm_s{seed}_n{a.noise:g}.npz'), 'falm', seed, a.noise, u, A, w, ref, data, seconds, hist, snap)
