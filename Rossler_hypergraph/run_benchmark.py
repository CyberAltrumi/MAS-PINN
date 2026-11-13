"""Rössler hypergraph experiment of the paper with MAS-PINN (method key `falm`).

python run_benchmark.py --prepare                    datasets with relative noise 0.5%, 1%, 3%
python run_benchmark.py --method falm --device cpu   one run per seed and noise level
Data: 8 oscillators, x(0) ~ U(-1, 1), RK45 solution on [0, 20] sampled at 151 times; for each seed the same Gaussian draw
is scaled to every noise level: sd = noise * std(clean observations).  All 28 pair and 56 triple candidates are
identified with a = b = 0.2, c = 5.7 fixed; no clean initial condition is supplied.
Completed trials are validated and skipped; new outputs are written atomically.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support
import torch
import run_paper
from util import hypergraph, EDGES_TRUE, TRIS_TRUE, A_R, B_R, C_R, K_EDGE, K_TRI

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / 'results' / 'benchmark'
SEEDS = [0, 2, 3, 4, 2026]
NOISES = [.005, .01, .03]
FALM = dict(run_paper.FALM, K_rff=128, sched_step=100, sched_gamma=.97, lr_min=1e-5, n_falm=10000)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_npz(path, **arrays):
    tmp = Path(str(path) + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def source_hashes():
    paths = [HERE / n for n in ['run_benchmark.py',
             'run_paper.py', 'network.py', 'physics_residual.py', 'LM_optim.py', 'inverse_QP.py', 'util.py']]
    paths += [HERE.parent / 'LV_half_observed' / 'network.py']
    return {str(p.relative_to(HERE.parents[1])): digest(p) for p in paths}


def data_rhs(_, s):
    """Right-hand side of the generating system on the state layout [x(8), y(8), z(8)].  The couplings are accumulated edge
    by edge and triangle by triangle: this fixes the floating-point result, hence the datasets, bit for bit."""
    x, y, z = s[:8], s[8:16], s[16:]
    pair, triple = np.zeros(8), np.zeros(8)
    for i, j in EDGES_TRUE:
        pair[i] += x[j] - x[i]; pair[j] += x[i] - x[j]
    for i, j, k in TRIS_TRUE:
        triple[i] += x[j]**2 * x[k] - x[i]**3 + x[j] * x[k]**2 - x[i]**3
        triple[j] += x[i]**2 * x[k] - x[j]**3 + x[i] * x[k]**2 - x[j]**3
        triple[k] += x[i]**2 * x[j] - x[k]**3 + x[i] * x[j]**2 - x[k]**3
    return np.concatenate((-y - z + K_EDGE * pair + K_TRI * triple, x + A_R * y, B_R + z * (x - C_R)))


def prepare(root, seeds=SEEDS, noises=NOISES):
    root.mkdir(parents=True, exist_ok=True)
    for name in ['datasets', 'runs', 'checkpoints', 'logs', 'failures']:
        (root/name).mkdir(exist_ok=True)
    path = root/'protocol.json'
    protocol = dict(seeds=list(seeds), noises=list(noises), methods=['falm'],
        observations=151, physics_points=151, t_end=20., evaluation_points=601, precision='float64 throughout; no float32 rounding',
        fixed_local=[.2, .2, 5.7], edge_bounds=[0., .4], triangle_bounds=[0., .3],
        noise_definition='noise * std(all 151x24 clean observations, ddof=0); same Gaussian draw across noise levels',
        noisy_initial_observation=True, extra_ic_loss=False, solver='RK45 rtol=1e-3 atol=1e-6',
        falm=FALM,
        falm_selection='minimum IC+data+0.01*physics training loss; no truth-based selection',
        example_selection=dict(seed=2026, nodes_zero_based=[1, 2, 3]), source_hashes=source_hashes())
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'Protocol/source changed: use a new output directory'
    else:
        path.write_text(json.dumps(protocol, indent=2)+'\n')
    for seed in seeds:
        needed = [root/'datasets'/f's{seed}_n{noise:g}.npz' for noise in noises]
        if all(p.exists() for p in needed):
            continue
        rng = np.random.RandomState(seed)
        x0, gaussian = rng.uniform(-1, 1, 24), rng.randn(151, 24)                       # state stored as [x(8), y(8), z(8)]
        sol = solve_ivp(data_rhs, (0, 20), x0, t_eval=np.linspace(0, 20, 151), dense_output=True)   # RK45, rtol 1e-3, atol 1e-6
        assert sol.success, sol.message
        clean, te = sol.y.T, np.linspace(0., 20., 601)
        t_obs = torch.linspace(0, 20, 151, dtype=torch.float64)[:, None].numpy()
        states = lambda a: a.reshape(-1,3,8).transpose(0,2,1).copy()                    # (T, 24) -> (T, 8 nodes, 3 components)
        for noise, dest in zip(noises, needed):
            if dest.exists():
                continue
            sd = noise*clean.std()
            noisy = clean+sd*gaussian
            atomic_npz(dest, seed=seed, noise=noise, noise_sd=sd, x0_blocked=x0, t_obs=t_obs, y_obs=states(noisy),
                y_clean=states(clean), gaussian_blocked=gaussian, y_train_blocked=noisy, t_eval=te[:,None],
                x_true=states(sol.sol(te).T))
    return protocol


def load_condition(path, device):
    with np.load(path) as src:
        a = {k: src[k] for k in src.files}
    assert all(v.dtype == np.float64 for v in a.values() if np.issubdtype(v.dtype, np.floating))
    tensor = lambda v: torch.as_tensor(v, dtype=torch.float64, device=device)
    edges, tris, B, Tm, w, v = hypergraph()
    times, obs = tensor(a['t_obs']), tensor(a['y_obs'])[None]
    data = dict(t_obs=times, y_obs=obs, t_f=times, t_ic=times[:1], x_ic=obs[:,0],
        mask_obs=torch.ones(8,3,dtype=torch.bool,device=device), mask_ic=torch.zeros(8,3,dtype=torch.bool,device=device),
        B=tensor(B), Tm=tensor(Tm), tri=torch.tensor(tris,dtype=torch.long,device=device),
        alpha_known=tensor([.2,.2,5.7]), lo=tensor(np.zeros(84)), hi=tensor(np.r_[np.full(28,.4),np.full(56,.3)]),
        obs=list(range(8)), obs_dims=[0,1,2], t_end=20.)
    ref = dict(t_eval=tensor(a['t_eval']), x_exact=tensor(a['x_true'])[None], theta=tensor(np.r_[w,v]),
        edges=edges, tris=tris, E=28, M=56)
    return data, ref, a


def run_falm(data, ref, seed, device, epochs):
    from network import ModelMix, RandomFourierFeaturesNN
    cfg = dict(FALM, n_falm=epochs)
    u, params, history, snap = run_paper.run_falm(data,ref,seed,device,cfg=cfg,log_every=500,return_state=True)
    saved = snap['model_state']
    backbone = ModelMix(cfg['hidden'], t_max=20.).to(device)
    backbone.load_state_dict(saved['backbone'])
    rff = RandomFourierFeaturesNN(cfg['hidden'][-1], cfg['K_rff'], cfg['sigma'], rff_device=device).to(device)
    rff.W.copy_(saved['rff_W'])
    with torch.no_grad():
        u_obs = torch.einsum('tp,kipa->ktia', rff(backbone,data['t_obs'])[0],saved['coefficients'].to(device))
    errors = np.array([[h['epoch']]+[h[k] for k in run_paper.KEYS] for h in history])
    return u,u_obs,params,snap['loss_history'],saved,dict(error_history=errors,
        ckpt_epoch=snap['ckpt_epoch'],ckpt_loss=snap['ckpt_loss'])


def execute(root, method, seed, noise, device, epochs=None):
    stem = f'{method}_s{seed}_n{noise:g}'
    dest, data_path = root/'runs'/f'{stem}.npz', root/'datasets'/f's{seed}_n{noise:g}.npz'
    protocol_hash, dataset_hash = digest(root/'protocol.json'), digest(data_path)
    intended_epochs = epochs or 10000
    if dest.exists():
        with np.load(dest) as saved:
            assert str(saved['protocol_sha256'])==protocol_hash and str(saved['dataset_sha256'])==dataset_hash
            assert int(saved['epochs'])==intended_epochs, 'Existing run has a different training length'
        print('SKIP completed '+stem,flush=True)
        return
    data,ref,raw = load_condition(data_path,device)
    t0=time.time()
    u,u_obs,params,history,model,extra=run_falm(data,ref,seed,device,intended_epochs)
    arr=lambda t:t.detach().cpu().numpy()
    pred,pred_obs,p,truth=arr(u)[0],arr(u_obs)[0],arr(params),arr(ref['theta'])
    assert all(v.dtype == np.float64 for v in [pred,pred_obs,p,truth,history])
    if not (np.isfinite(pred).all() and np.isfinite(p).all()):
        raise FloatingPointError('Nonfinite saved result')
    rel=lambda a,b:float(100*np.linalg.norm(a-b)/np.linalg.norm(b))
    metrics=dict(w=rel(p[:28],truth[:28]),v=rel(p[28:],truth[28:]),theta=rel(p,truth),
        traj=rel(pred,raw['x_true']),traj_obs=rel(pred_obs,raw['y_clean']),
        obs_fit=rel(pred_obs,raw['y_obs']))
    scores=p/np.r_[np.full(28,.4),np.full(56,.3)]
    roc={}
    for label,sl in [('2',slice(0,28)),('3',slice(28,None)),('all',slice(None))]:
        target=truth[sl]>0
        metrics['auc'+label]=float(roc_auc_score(target,scores[sl]))
        fpr,tpr,thresholds=roc_curve(target,scores[sl])
        roc.update({f'roc{label}_fpr':fpr,f'roc{label}_tpr':tpr,f'roc{label}_thresholds':thresholds})
    precision,recall,f1,_=precision_recall_fscore_support(truth>0,scores>=.5,average='binary',zero_division=0)
    metrics.update(precision=float(precision),recall=float(recall),f1=float(f1))
    seconds=time.time()-t0
    runtime=dict(device=str(device),precision='float64',torch_version=torch.__version__,numpy_version=np.__version__,
        cpu_affinity=sorted(os.sched_getaffinity(0)),torch_threads=torch.get_num_threads(),
        rff_initialization_device=str(device))
    checkpoint=root/'checkpoints'/f'{stem}.pt'
    if model:
        tmp=Path(str(checkpoint)+'.tmp')
        torch.save(dict(method=method,seed=seed,noise=noise,epochs=intended_epochs,params=params.detach().cpu(),
                        model=model,protocol_sha256=protocol_hash,dataset_sha256=dataset_hash),tmp)
        tmp.replace(checkpoint)
    atomic_npz(dest, method=method,seed=seed,noise=noise,seconds=seconds,epochs=intended_epochs,
        protocol_sha256=protocol_hash,dataset_sha256=dataset_hash,u=pred,u_obs=pred_obs,params=p,
        runtime_json=np.array(json.dumps(runtime)),
        w=p[:28],v=p[28:],theta_true=truth,scores=scores,labels=truth>0,t_eval=raw['t_eval'][:,0],
        x_true=raw['x_true'],t_obs=raw['t_obs'][:,0],y_obs=raw['y_obs'],y_clean=raw['y_clean'],
        noise_sd=raw['noise_sd'],edges=np.array(ref['edges']),tris=np.array(ref['tris']),
        loss_history=history,**{f'err_{k}':value for k,value in metrics.items()},**roc,**extra)
    (root/'runs'/f'{stem}.json').write_text(json.dumps(dict(method=method,seed=seed,noise=noise,
        seconds=seconds,epochs=intended_epochs,**metrics),indent=2)+'\n')
    print('FINAL '+stem+' '+json.dumps(dict(seconds=seconds,**metrics)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--method',choices=['falm'])
    parser.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    parser.add_argument('--noises',type=float,nargs='+',default=NOISES)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--epochs',type=int)
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.prepare:
        prepare(args.out, args.seeds,args.noises)
    if args.method:
        protocol=json.loads((args.out/'protocol.json').read_text())
        assert protocol['source_hashes']==source_hashes(),'Sources differ from frozen protocol'
        for seed in args.seeds:
            for noise in args.noises:
                assert seed in protocol['seeds'] and noise in protocol['noises']
                try:
                    execute(args.out,args.method,seed,noise,args.device,args.epochs)
                except Exception as exc:
                    (args.out/'failures'/f'{args.method}_s{seed}_n{noise:g}.json').write_text(json.dumps(
                        dict(error=repr(exc),method=args.method,seed=seed,noise=noise),indent=2)+'\n')
                    raise
