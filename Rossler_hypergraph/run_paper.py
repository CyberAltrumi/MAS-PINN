"""MAS-PINN (method key `falm`) on Roessler oscillators coupled through an undirected hypergraph.
Each trial is saved to <out>/falm_s<seed>_n<noise>.npz with the reconstruction u (t_eval, N, 3), theta, the truths, the
observations, the relative errors (%), the AUCs and the wall time.
    python run_paper.py --noise 0.005 --device cuda          (noise levels of the paper: 0.005, 0.01, 0.03)
run_benchmark.py runs the experiment of the paper."""
import argparse, copy, os, time
import numpy as np, torch
torch.set_default_dtype(torch.float64)
from util import generate_rossler_dataset
from network import ModelMix, RandomFourierFeaturesNN, EndToEnd, basis_with_derivative
from physics_residual import rff_loss
from inverse_QP import init_theta, init_params, solve_theta, inverse_errors
import LM_optim
from LM_optim import lm_step

PROBLEM = dict(N_obs=150, N_f=1600, K=1, obs_stride=1, obs_dims=(0, 1, 2), ic='observed', t_end=20.0, candidates='all', fit_local=True)
FALM = dict(hidden=[64, 64, 64], K_rff=64, sigma=3 * 2 * np.pi, n_warmup=0, lr_warmup=1e-3, lr_param=3e-3, n_falm=3000, lr_falm=1e-3,
            sched_step=100, sched_gamma=0.97, lr_min=1e-5, lm_iter=1, cg_iters=100, physics_weight=1e-2)
KEYS = ('alpha', 'w', 'v', 'traj', 'hidden', 'auc2', 'auc3')


def run_falm(data, ref, seed, device, cfg=FALM, log_every=500, return_state=False):
    """MAS-PINN: shared trunk + RFF basis, per-agent coefficients; (optional Adam warm-up), then per outer iteration
    trunk Adam step -> coupled matrix-free LM on the coefficients -> bounded LS for theta; checkpoint on the observable loss."""
    LM_optim.CG_ITERS = cfg['cg_iters']
    pw, K, N = cfg['physics_weight'], PROBLEM['K'], ref['x_exact'].shape[2]
    torch.manual_seed(seed)
    backbone = ModelMix(list(cfg['hidden']), t_max=PROBLEM['t_end']).to(device)
    rff = RandomFourierFeaturesNN(cfg['hidden'][-1], cfg['K_rff'], sigma=cfg['sigma'], rff_device=device)
    model = EndToEnd(backbone, rff, init_theta(data, 1 + 2 * cfg['K_rff']), init_params(data)).to(device)
    mats = lambda: (rff(backbone, data['t_ic'])[0], rff(backbone, data['t_obs'])[0], *basis_with_derivative(backbone, rff, data['t_f']))
    track = Tracker(ref, data, log_every, cfg['n_warmup'] + cfg['n_falm'], 'MAS-PINN')
    if cfg['n_warmup']:
        opt = torch.optim.Adam([{'params': list(backbone.parameters()) + [model.theta], 'lr': cfg['lr_warmup']}, {'params': [model.params], 'lr': cfg['lr_param']}])
        for epoch in range(1, cfg['n_warmup'] + 1):
            opt.zero_grad(); loss = rff_loss(model.theta, *mats(), model.params, data, pw); loss[0].backward(); opt.step()
            with torch.no_grad(): model.params.clamp_(data['lo'], data['hi'])
            track(epoch, lambda: model(ref['t_eval']), model.params, f"warm-up obs {loss[2].item():.2e} phys {loss[3].item():.2e}")
    theta, params = model.theta.detach().clone(), model.params.detach().clone()
    opt = torch.optim.Adam(backbone.parameters(), lr=cfg['lr_falm'])
    sched = torch.optim.lr_scheduler.StepLR(opt, cfg['sched_step'], cfg['sched_gamma']) if cfg['sched_step'] else None
    with torch.no_grad():                                        # fit the coefficients to the initial basis first
        m = mats()
        theta, damping, _ = lm_step(theta, m, params, data, pw, damping=1e-3, max_iter=20)
        params = solve_theta(theta, m[2], m[3], data)
        theta, damping, _ = lm_step(theta, m, params, data, pw, damping, max_iter=5)
    best, loss_hist = dict(loss=float('inf')), []
    for epoch in range(1, cfg['n_falm'] + 1):
        opt.zero_grad(); rff_loss(theta, *mats(), params, data, pw)[0].backward(); opt.step()
        if sched and opt.param_groups[0]['lr'] > cfg['lr_min']: sched.step()
        with torch.no_grad():
            m = mats()
            for _ in range(cfg['lm_iter']):
                theta, damping, _ = lm_step(theta, m, params, data, pw, damping, 1)
            params = solve_theta(theta, m[2], m[3], data)
            loss = rff_loss(theta, *m, params, data, pw)
            if return_state and (epoch == 1 or epoch == cfg['n_falm'] or (log_every and epoch % log_every == 0)):
                loss_hist.append([epoch] + [v.item() for v in loss] + [damping, opt.param_groups[0]['lr']])
            if loss[0].item() < best['loss']:                    # observable loss only: the truth never selects anything
                best = dict(loss=loss[0].item(), epoch=epoch, theta=theta.clone(), params=params.clone(), basis=copy.deepcopy(backbone.state_dict()))
            track(cfg['n_warmup'] + epoch, lambda: torch.einsum('tp,kipa->ktia', rff(backbone, ref['t_eval'])[0], theta), params,
                  f"loss {loss[0].item():.3e} obs {loss[2].item():.2e} phys {loss[3].item():.2e} damping {damping:.1e}")
    backbone.load_state_dict(best['basis'])
    track.best.update(ckpt_epoch=best['epoch'], ckpt_loss=best['loss'])
    if return_state:
        track.best['loss_history'] = np.array(loss_hist)
        track.best['model_state'] = dict(backbone={k: v.detach().cpu() for k, v in backbone.state_dict().items()},
                                         rff_W=rff.W.detach().cpu(), coefficients=best['theta'].detach().cpu())
    print(f"best checkpoint (observable loss) at epoch {best['epoch']}")
    with torch.no_grad():
        return torch.einsum('tp,kipa->ktia', rff(backbone, ref['t_eval'])[0], best['theta']), best['params'], track.hist, track.best


class Tracker:
    """Truth-based error history every `every` epochs (evaluation only) and the snapshot with the smallest trajectory error."""
    def __init__(self, ref, data, every, epochs, name):
        self.ref, self.data, self.every, self.epochs, self.name = ref, data, every, epochs, name
        self.hist, self.best, self.t0 = [], dict(traj=float('inf')), time.time()

    def __call__(self, epoch, state, params, msg=''):
        if self.every and (epoch % self.every == 0 or epoch == self.epochs or epoch == 1):
            with torch.no_grad():
                u = state(); e = inverse_errors(u, params, self.ref, self.data); self.hist.append(dict(epoch=epoch, **e))
                if e['traj'] < self.best['traj']:
                    self.best = dict(e, epoch=epoch, u=u[0].cpu().numpy(), params=params.detach().cpu().numpy())
            print(f"{self.name:8s} {epoch:6d} | {msg} | alpha {e['alpha']:.2f}% w {e['w']:.2f}% v {e['v']:.2f}% traj {e['traj']:.2f}% "
                  f"auc {e['auc2']:.3f}/{e['auc3']:.3f} | {time.time() - self.t0:.0f}s", flush=True)


def save(path, method, seed, noise, u, params, ref, data, seconds, hist, snap):
    e = inverse_errors(u, params, ref, data)
    npy = lambda v: v.detach().cpu().numpy()
    snap = snap or dict(epoch=-1, u=npy(u[0]), params=npy(params))
    np.savez(path, method=method, seed=seed, noise=noise, seconds=seconds, u=npy(u[0]), theta=npy(params),
             u_min=snap['u'], theta_min=snap['params'], epoch_min=snap['epoch'],
             hist=np.array([[h['epoch']] + [h[k] for k in KEYS] for h in hist]) if hist else np.zeros((0, 1 + len(KEYS))),
             x_true=npy(ref['x_exact'][0]), theta_true=npy(ref['theta']), t_eval=npy(ref['t_eval'])[:, 0],
             t_obs=npy(data['t_obs'])[:, 0], y_obs=npy(data['y_obs'][0]), mask_obs=npy(data['mask_obs']),
             edges=np.array(ref['edges']), tris=np.array(ref['tris']), **{f'err_{k}': v for k, v in e.items()})
    print(f"{method:8s} seed {seed:5d} noise {noise:g} | alpha {e['alpha']:.2f}% w {e['w']:.2f}% v {e['v']:.2f}% traj {e['traj']:.2f}% "
          f"auc {e['auc2']:.3f}/{e['auc3']:.3f} | {seconds:.0f}s -> {path}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 2, 3, 4, 2026])
    ap.add_argument('--noise', type=float, default=0.005, help='relative measurement noise; the paper uses 0.005, 0.01 and 0.03')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='results')
    ap.add_argument('--epochs', type=int, default=None, help='override the training length (smoke tests)')
    ap.add_argument('--log-every', type=int, default=500)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.epochs: FALM['n_falm'] = a.epochs
    device = torch.device(a.device)
    for seed in a.seeds:
        data, ref = generate_rossler_dataset(**PROBLEM, noise=a.noise, ic_noise=a.noise > 0, seed=seed, device=device)
        t0 = time.time()
        u, params, hist, snap = run_falm(data, ref, seed, device, log_every=a.log_every)
        save(os.path.join(a.out, f'falm_s{seed}_n{a.noise:g}.npz'), 'falm', seed, a.noise, u, params, ref, data, time.time() - t0, hist, snap)
