"""One-factor-at-a-time scaling study for Proposition 1 (cost of the matrix-free LM operator) on the network LV.
Axes: N (agents), N_f (collocation points), P (basis size, via K_rff), CG steps; the others stay at the paper setting
(N=8, N_f=1600, P=129, CG 100).  For each setting: (i) one operator application v -> J^T(Jv) + mu v, time over 200 calls
and the size of the cached quantities; (ii) 100 outer iterations of MAS-PINN: LM-solve time and peak memory, whole-iteration
time and peak memory; (iii) 20 outer iterations with the dense LM step (explicit J, J^T J, Cholesky; damping floor 1e-2).
    python bench_scaling.py            -> prints a table, writes results/scaling.json"""
import json, math, os, time, traceback
import torch
torch.set_default_dtype(torch.float64)
import bench_time as bt, sweep_falm as sf, run_paper as rp, LM_optim
from LM_optim import _linearise, residual_blocks, J_vec, JT_vec, mean_diag_H

MED = dict(N=8, N_f=1600, K_rff=64, cg=100)
AXES = dict(N=[4, 8, 16, 32, 64], N_f=[800, 1600, 6400], K_rff=[32, 64, 256], cg=[30, 60, 100])
DENSE_MAX_N = 16                                           # N = 32 needs ~35 GB for the explicit Jacobian
d = 3


def operator_bench(state, pw, calls=200):
    """Time of one v -> J^T(Jv) + mu v (median of `calls`), and the bytes of the cached operator + CG work vectors."""
    theta, A, w, data, m = state['theta'], state['A'], state['w'], state['data'], state['mats']
    with torch.no_grad():
        op = _linearise(theta, *m, A, w, data, pw, 1e-10); op.update(A=A, w=w)
        mu = mean_diag_H(op); v = torch.randn_like(theta)
        apply = lambda v: JT_vec(J_vec(v, op), op) + 1e-3 * mu * v
        for _ in range(20): apply(v)
        bt.sync(); ts = []
        for _ in range(calls):
            t = time.time(); apply(v); bt.sync(); ts.append(time.time() - t)
    cache = sum(x.numel() * 8 for x in op.values() if torch.is_tensor(x)) + 5 * theta.numel() * 8
    return dict(op_ms=1e3 * sorted(ts)[len(ts) // 2], cache_MB=cache / 1e6)


def predicted_cost(N, N_f, P, E, N_obs, nnzC):
    """Proposition 1 operation count (no constants)."""
    return N_f * (N * P * d + (N + E) * d * d) + N_obs * (N * P * d + nnzC)


def run_setting(N, N_f, K_rff, cg, dense=True):
    cfg = {**sf.BASE, **rp.FALM, 'physics_weight': 1e-2, 'K_rff': K_rff, 'cg_iters': cg}
    state = bt.build_state(cfg, 2026, 0.005, N_agents=N, N_f=N_f)
    P, E = 1 + 2 * K_rff, 2 * state['edges']                       # directed edge count (both directions of every undirected edge)
    N_obs, nnzC = state['data']['y_obs'].shape[1], state['data']['y_obs'].numel() // state['data']['y_obs'].shape[1]
    r = dict(N=N, N_f=N_f, P=P, cg=cg, E=E, N_obs=N_obs, PNd=P * N * d, pred_ops=predicted_cost(N, N_f, P, E, N_obs, nnzC),
             pred_store_MB=8e-6 * (N_f * (N + E) * d * d + (N_f + N_obs) * P + P * N * d))
    r.update(operator_bench(state, cfg['physics_weight']))
    f = bt.falm_loop(state, cfg, 100, sf.lm_step_LV, cg_iters=cg)
    r.update(mf_total_s=f['total'], mf_lm_s=f['lm'], mf_peak_GB=f['peak_mem_GB'], mf_lm_peak_GB=f['lm_peak_GB'], mf_loss=f['final_loss'])
    if dense:
        try:
            g = bt.falm_loop(state, cfg, 20, bt.dense_step_floored)
            r.update(dense_lm_s_per100=5 * g['lm'], dense_peak_GB=g['peak_mem_GB'], dense_lm_peak_GB=g['lm_peak_GB'], dense_loss=g['final_loss'])
        except torch.cuda.OutOfMemoryError:
            r.update(dense_lm_s_per100=float('nan'), dense_peak_GB=float('nan'), dense_lm_peak_GB=float('nan'), dense_loss=float('nan'), dense_oom=True)
            torch.cuda.empty_cache()
    print(f"N={N:3d} N_f={N_f:5d} P={P:4d} CG={cg:3d} | op {r['op_ms']:7.3f} ms, cache {r['cache_MB']:7.1f} MB (pred {r['pred_store_MB']:7.1f}) | "
          f"MF /100 it: LM {r['mf_lm_s']:7.2f} s, LM peak {1e3*r['mf_lm_peak_GB']:8.1f} MB, total {r['mf_total_s']:7.2f} s, peak {1e3*r['mf_peak_GB']:8.1f} MB | "
          + (f"dense /100 it: LM {r['dense_lm_s_per100']:8.1f} s, LM peak {1e3*r['dense_lm_peak_GB']:9.1f} MB" + (' OOM' if r.get('dense_oom') else '') if dense else 'dense: skipped'), flush=True)
    del state; torch.cuda.empty_cache()
    return r


if __name__ == '__main__':
    out = []; done = {}
    for axis, values in AXES.items():
        for v in values:
            p = {**MED, axis: v}; key = tuple(sorted(p.items()))
            if key in done: out.append({**done[key], 'axis': axis}); continue
            try:
                r = run_setting(p['N'], p['N_f'], p['K_rff'], p['cg'], dense=p['N'] <= DENSE_MAX_N)
            except Exception as e:
                print(f"FAILED {p}: {e}"); traceback.print_exc(); continue
            done[key] = r; out.append({**r, 'axis': axis})
            json.dump(out, open('results/scaling.json', 'w'), indent=1)
    print("written results/scaling.json; DONE_SCALING")
