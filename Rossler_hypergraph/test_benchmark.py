"""python test_benchmark.py -- the benchmark datasets: y = clean + sd * Z with one Gaussian draw Z per seed shared by all
noise levels, float64 throughout; the generating right-hand side == the model that is identified."""
from pathlib import Path
import tempfile
import numpy as np
import torch
from run_benchmark import NOISES, prepare, load_condition, data_rhs
from util import hypergraph, rossler_rhs_np, A_R, B_R, C_R

torch.set_num_threads(1)
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    prepare(root, seeds=[42], noises=NOISES)
    data, ref, raw = load_condition(root/'datasets'/f's42_n{NOISES[0]:g}.npz', 'cpu')
    assert data['y_obs'].shape == (1, 151, 8, 3) and ref['x_exact'].shape == (1, 601, 8, 3)
    previous = None
    for noise in NOISES:
        with np.load(root/'datasets'/f's42_n{noise:g}.npz') as d:
            clean = d['y_clean'].transpose(0, 2, 1).reshape(151, 24)
            assert np.isclose(float(d['noise_sd']), noise * clean.std(), rtol=1e-14)
            assert np.array_equal(d['y_train_blocked'], clean + float(d['noise_sd']) * d['gaussian_blocked'])
            assert all(d[k].dtype == np.float64 for k in ['t_obs', 'y_obs', 'y_clean', 'y_train_blocked', 'x_true'])
            assert previous is None or np.array_equal(previous, d['gaussian_blocked'])
            previous = d['gaussian_blocked'].copy()
_, tris, B, Tm, w, v = hypergraph('all')
s = np.random.default_rng(0).uniform(-5, 5, 24)                   # state layout [x(8), y(8), z(8)]
model = rossler_rhs_np(s.reshape(3, 8).T, np.r_[A_R, B_R, C_R, w, v], B, Tm, np.array(tris)).T.ravel()
assert np.allclose(model, data_rhs(0, s), rtol=0, atol=1e-11)
print('ok: generating right-hand side == identified model; paired noisy data (same Gaussian draw across noise levels, sd = noise * std of the clean observations, float64)')
