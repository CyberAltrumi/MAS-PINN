"""Lorenz-96 inference: reference field and log-scale absolute-error field |x_tilde - x| of MAS-PINN, plot_nature's style.

python plot_lorenz.py            # N = 100, trial 1 of ../Lorenz96_forward/results -> ../figures/lorenz_errors.{pdf,png}
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import plot_nature as style   # rcParams and display name

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument('--results', type=Path, default=ROOT / 'Lorenz96_forward' / 'results')
ap.add_argument('--N', type=int, default=100); ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--cmap', default='viridis'); ap.add_argument('--vmin', type=float, default=1e-5); ap.add_argument('--vmax', type=float, default=1e1)
ap.add_argument('--width-mm', type=float, default=180); ap.add_argument('--out', type=Path, default=ROOT / 'figures')
a = ap.parse_args()

z = np.load(a.results / f'falm_N{a.N}_F12_T1_s{a.seed}.npz')
t, x, name = z['t_eval'], z['x_true'], style.METHODS['falm'][0]
sites = np.arange(1, a.N + 1)

fig, axes = plt.subplots(1, 2, figsize=(a.width_mm / 25.4, 0.31 * a.width_mm / 25.4), sharex=True, sharey=True, layout='constrained')
panels = [('Reference', x.T, None), (name + ' Abs Err', np.abs(z['u'] - x).T, LogNorm(a.vmin, a.vmax))]
for ax, (title, field, norm) in zip(axes, panels):
    im = ax.pcolormesh(t, sites, field, cmap=a.cmap, norm=norm, shading='nearest', rasterized=True)
    cb = fig.colorbar(im, ax=ax, pad=0.02, aspect=18)   # values outside [vmin, vmax] are clipped to the end colours
    cb.outline.set_linewidth(0.5); cb.ax.tick_params(width=0.5, length=2)
    ax.set_title(title, pad=3); ax.set_xlabel('Time $t$')
axes[0].set_ylabel('Site $i$')
axes[0].set_yticks([1, a.N // 4, a.N // 2, 3 * a.N // 4, a.N])
a.out.mkdir(exist_ok=True)
for suffix in ('.pdf', '.png'):
    fig.savefig((a.out / 'lorenz_errors').with_suffix(suffix), dpi=600 if suffix == '.png' else 300)
e = np.abs(z['u'] - x); print(f"{name} rel L2 {np.linalg.norm(z['u'] - x) / np.linalg.norm(x):.2e}  abs err median {np.median(e):.1e} max {e.max():.1e}")
