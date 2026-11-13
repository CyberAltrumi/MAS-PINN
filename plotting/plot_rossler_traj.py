"""Trajectory figure of the Rössler experiment: one row, the three state components of the selected nodes, MAS-PINN vs ground truth.

python plot_rossler_traj.py --experiment ../Rossler_hypergraph/results/benchmark      -> <experiment>/figures_nature/rossler_trajectories.*
Trial and nodes default to the example of the experiment's protocol.json; the noise level defaults to the largest one."""
import argparse, json
from pathlib import Path
import numpy as np
import plot_nature as style
from matplotlib import ticker
from matplotlib.lines import Line2D

ap = argparse.ArgumentParser()
ap.add_argument('--experiment', type=Path, required=True)
ap.add_argument('--seed', type=int); ap.add_argument('--noise', type=float); ap.add_argument('--nodes', type=int, nargs='+', help='zero-based')
ap.add_argument('--out', type=Path, help='output path without suffix')
a = ap.parse_args()
protocol = json.loads((a.experiment / 'protocol.json').read_text()); example = protocol['example_selection']
seed, noise = a.seed if a.seed is not None else example['seed'], a.noise or max(protocol['noises'])
nodes = a.nodes or example['nodes_zero_based']
methods = [m for m in style.METHODS if (a.experiment / 'runs' / f'{m}_s{seed}_n{noise:g}.npz').exists()]
runs = {m: np.load(a.experiment / 'runs' / f'{m}_s{seed}_n{noise:g}.npz') for m in methods}
t, truth = runs[methods[0]]['t_eval'], runs[methods[0]]['x_true']
fig, axes = style.make_grid(1, 180)
for comp, ax in enumerate(axes[0]):
    for node in nodes:
        for m in methods:
            _, color, ls, _ = style.METHODS[m]
            ax.plot(t, runs[m]['u'][:, node, comp], color=color, ls=ls, lw=1.2, alpha=.95, zorder=2)
        ax.plot(t, truth[:, node, comp], color=style.INK, ls=(0, (3, 2)), lw=.8, zorder=3)
    ax.set_xlim(t[0], t[-1]); ax.margins(y=.06)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4)); ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    style.panel_header(ax, 'abc'[comp], f'Component {comp+1}')
    ax.set_xlabel(r'Time $t$'); ax.set_ylabel(f'$x_{{i,{comp+1}}}$')
handles = [Line2D([], [], color=style.INK, ls=(0, (3, 2)), lw=.9, label='Ground truth')]
handles += [Line2D([], [], color=style.METHODS[m][1], ls=style.METHODS[m][2], lw=1.4, label=style.METHODS[m][0]) for m in methods]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .995), ncol=len(handles), handlelength=2.3, columnspacing=1.4)
base = a.out or a.experiment / 'figures_nature' / 'rossler_trajectories'; base.parent.mkdir(parents=True, exist_ok=True)
style.save_geometry(fig, axes, base.with_suffix('.alignment.json'))
for suf in ['.pdf', '.svg', '.png']: fig.savefig(base.with_suffix(suf), dpi=600 if suf == '.png' else None)
print('saved', base)
