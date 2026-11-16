"""Restyle network Lotka--Volterra figures from run_paper.py NPZ outputs.

Requires numpy and matplotlib. No simulation or training data are modified.
Display name MAS-PINN retains the original on-disk method key 'falm'.
Error input units must be declared; no automatic unit inference is performed.
--site uses ZERO-BASED array indices; all selected sites are overlaid in one row of species panels.
The original minimum-error selection is retained, but explicitly disclosed.
"""
import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from audit_panel_alignment import require_matplotlib_panel_alignment

# method key -> (display name, colour, line style, marker); add an entry to plot the runs of another method.
METHODS = {
    'falm': ('MAS-PINN', '#ff4040', '-', 'o'),
}
INK, GRID = '#30343B', '#E3E6EA'
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7.5,
    'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': INK, 'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK, 'axes.linewidth': 0.65,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.width': 0.65, 'ytick.major.width': 0.65,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'legend.frameon': False, 'axes.axisbelow': True,
})


def load_runs(folder, seeds, noises):
    runs = {}
    for path in sorted(Path(folder).glob('*.npz')):
        with np.load(path, allow_pickle=False) as src:
            z = {key: src[key].copy() for key in src.files}
        method = str(z['method'].item())
        seed, noise = int(z['seed'].item()), float(z['noise'].item())
        if seeds is not None and seed not in seeds:
            continue
        if noises is not None and not any(np.isclose(noise, n, atol=1e-12, rtol=0) for n in noises):
            continue
        if method not in METHODS:
            raise ValueError(f'Unknown method {method!r} in {path}; add its display style.')
        if not np.isfinite(noise) or noise < 0:
            raise ValueError(f'Invalid noise in {path}')
        key = (method, noise)
        if seed in runs.get(key, {}):
            raise ValueError(f'Duplicate method/noise/seed: {method}, {noise}, {seed}')
        runs.setdefault(key, {})[seed] = z
    if not runs:
        raise ValueError('No matching NPZ runs found. Check --results, --seeds and --noise.')
    return runs


def error_value(z, metric, selection, units):
    key = ('min_' if selection == 'min' else 'err_') + metric
    if key not in z:
        raise ValueError(f'Missing {key!r}; refusing to mix minimum and final errors.')
    value = float(z[key]) / (100 if units == 'percent' else 1)   # plotted as a ratio
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{key}={value}: log-scale errors must be finite and positive; no values are clipped.')
    return value


def make_grid(nrows, width_mm):
    """Fixed physical gutters give identical data-rectangle sizes in each row."""
    height_mm = 18 + 26 * nrows   # 26 mm data rectangles: flat panels to save page height
    fig, axes = plt.subplots(nrows, 3, squeeze=False,
                             figsize=(width_mm / 25.4, height_mm / 25.4))
    left, right, bottom, top = 17, 4, 12, 6
    gutter_x, gutter_y = 12, 15
    panel_w = (width_mm - left - right - 2 * gutter_x) / 3
    panel_h = (height_mm - top - bottom - (nrows - 1) * gutter_y) / nrows
    for row in range(nrows):
        for col in range(3):
            x = left + col * (panel_w + gutter_x)
            y = height_mm - top - (row + 1) * panel_h - row * gutter_y
            axes[row, col].set_position([x / width_mm, y / height_mm,
                                       panel_w / width_mm, panel_h / height_mm])
    return fig, axes


def save_geometry(fig, axes, path):
    """Block export if rendered plot-area alignment is inconsistent."""
    require_matplotlib_panel_alignment(fig, json_out=str(path),
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)


def panel_header(ax, letter, title):
    """Panel letters and titles are intentionally omitted; they are given in the caption."""
    ax.grid(axis='y', color=GRID, linewidth=0.5)


def fig_errors(runs, args, metrics=('A', 'w', 'hidden'), name='lv_errors', values_csv='lv_error_values.csv'):
    """Defaults give the LV figure; plot_rossler.py passes its own metrics and file names to share this style."""
    noises = sorted({noise for _, noise in runs})
    methods = [m for m in args.methods if any((m, n) in runs for n in noises)]
    fig, axes = make_grid(1, args.width_mm)
    gap = 0.78 / len(methods)
    box_width = 0.66 * gap
    rows = []
    for ax, metric in zip(axes[0], metrics):
        for group, noise in enumerate(noises):
            for j, method in enumerate(methods):
                trials = sorted(runs.get((method, noise), {}).items())
                if not trials:
                    warnings.warn(f'Missing group: {method}, noise={noise}; leaving an empty slot.')
                    continue
                values = [error_value(z, metric, args.select, args.error_units) for _, z in trials]
                x = group + (j - (len(methods) - 1) / 2) * gap
                _, color, _, marker = METHODS[method]
                ax.boxplot([values], positions=[x], widths=box_width, whis=1.5,
                           showfliers=False, patch_artist=True, manage_ticks=False,
                           boxprops={'facecolor': to_rgba(color, 0.18), 'edgecolor': color, 'linewidth': 0.8},
                           medianprops={'color': color, 'linewidth': 1.3},
                           whiskerprops={'color': color, 'linewidth': 0.75},
                           capprops={'color': color, 'linewidth': 0.75}, zorder=2)
                # Repeatable offsets; same seed order has the same offset in each metric.
                offsets = np.linspace(-box_width * 0.3, box_width * 0.3, len(values)) if len(values) > 1 else [0]
                ax.scatter(x + np.asarray(offsets), values, s=13, marker=marker,
                           facecolor=color, edgecolor='white', linewidth=0.35, zorder=3)
                rows.extend({'method': METHODS[method][0], 'noise_fraction': noise,
                             'seed': seed, 'metric': metric, 'error_ratio': value,
                             'selection': args.select}
                            for (seed, _), value in zip(trials, values))
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1, 3), numticks=8))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.set_xticks(range(len(noises)), [f'{100*n:g}%' for n in noises])
        ax.set_xlim(-0.5, len(noises) - 0.5)
        ax.set_xlabel('Measurement noise')
        ax.tick_params(axis='x', length=0, pad=5)
        ax.margins(y=0.14)
        panel_header(ax, '', '')
    axes[0, 0].set_ylabel('Relative $L^2$ error')
    handles = [Patch(facecolor=to_rgba(METHODS[m][1], 0.2), edgecolor=METHODS[m][1],
                     label=METHODS[m][0]) for m in methods]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.53, 0.995),
               ncol=len(methods), handlelength=1.2, columnspacing=1.5)
    base = args.out / name
    save_geometry(fig, axes, base.with_suffix('.alignment.json'))
    fig.savefig(base.with_suffix('.pdf'))
    fig.savefig(base.with_suffix('.svg'))
    fig.savefig(base.with_suffix('.png'), dpi=600)
    plt.close(fig)
    with (args.out / values_csv).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def choose_trajectory(runs, args):
    noises = sorted(n for m, n in runs if m == 'falm')
    if not noises:
        raise ValueError("Trajectory reference requires at least one 'falm' run.")
    noise = noises[0] if args.traj_noise is None else next(
        (n for n in noises if np.isclose(n, args.traj_noise, atol=1e-12, rtol=0)), None)
    if noise is None:
        raise ValueError('--traj-noise is absent from the filtered runs.')
    candidates = runs['falm', noise]
    seed = args.seed
    if seed is None:
        seed = min(candidates, key=lambda s: error_value(candidates[s], 'w', args.select, args.error_units))
        warnings.warn('Trajectory seed selected by lowest MAS-PINN w error, as in the original script. '
                      'For a prespecified example, pass --seed explicitly.')
    if seed not in candidates:
        raise ValueError(f'Seed {seed} is absent at noise={noise}.')
    return seed, noise, candidates[seed]


def fig_trajectories(runs, args):
    seed, noise, ref = choose_trajectory(runs, args)
    t, truth = np.asarray(ref['t_eval']).reshape(-1), ref['x_true']
    if truth.ndim != 3 or truth.shape[0] != len(t) or truth.shape[2] != 3:
        raise ValueError('Expected x_true shape (time, sites, 3).')
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
        raise ValueError('Truth must be finite and evaluation times strictly increasing.')
    sites = args.site if args.site is not None else list(range(truth.shape[1]))
    if len(set(sites)) != len(sites) or any(i < 0 or i >= truth.shape[1] for i in sites):
        raise ValueError('--site must contain distinct, valid zero-based indices.')
    ukey = 'u_min' if args.select == 'min' else 'u'
    predictions = {}
    for method in args.traj_methods:
        z = runs.get((method, noise), {}).get(seed)
        if z is None:
            raise ValueError(f'Missing trajectory run for {method}, seed={seed}, noise={noise}.')
        if ukey not in z or z[ukey].shape != truth.shape:
            raise ValueError(f'{method}: missing {ukey} or incompatible trajectory dimensions.')
        if not np.all(np.isfinite(z[ukey])):
            raise ValueError(f'{method}: nonfinite trajectory values.')
        if not np.array_equal(np.asarray(z['t_eval']).reshape(-1), t) or not np.allclose(z['x_true'], truth):
            raise ValueError(f'{method}: evaluation grid or ground truth differs from reference.')
        predictions[method] = z[ukey]
    fig, axes = make_grid(1, args.width_mm)
    for species, ax in enumerate(axes[0]):
        for site in sites:
            for method, prediction in predictions.items():
                _, color, style, _ = METHODS[method]
                ax.plot(t, prediction[:, site, species], color=color, ls=style,
                        lw=1.2 if method == 'falm' else 1.0, alpha=0.95, zorder=2)
            ax.plot(t, truth[:, site, species], color=INK, ls=(0, (3, 2)), lw=0.8, zorder=3)
        ax.set_xlim(t[0], t[-1])
        ax.margins(y=0.06)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        panel_header(ax, 'abc'[species], f'Species {species+1}')
        ax.set_xlabel(r'Time $t$')
    axes[0, 0].set_ylabel('Population density')
    handles = [Line2D([], [], color=INK, ls=(0, (3, 2)), lw=0.9, label='Ground truth')]
    handles += [Line2D([], [], color=METHODS[m][1], ls=METHODS[m][2], lw=1.4,
                       label=METHODS[m][0]) for m in args.traj_methods]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.53, 0.995),
               ncol=len(handles), handlelength=2.3, columnspacing=1.4)
    base = args.out / 'lv_trajectories'
    save_geometry(fig, axes, base.with_suffix('.alignment.json'))
    fig.savefig(base.with_suffix('.pdf'))
    fig.savefig(base.with_suffix('.svg'))
    fig.savefig(base.with_suffix('.png'), dpi=600)
    plt.close(fig)
    return {'seed': seed, 'noise_fraction': noise, 'site_indices_zero_based': sites,
            'selection': args.select, 'prediction_key': ukey,
            'seed_rule': 'explicit' if args.seed is not None else 'best MAS-PINN w error'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, default=Path('results'))
    ap.add_argument('--out', type=Path, default=Path('figures_nature'))
    ap.add_argument('--seeds', type=int, nargs='+')
    ap.add_argument('--noise', type=float, nargs='+')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--traj-noise', type=float)
    ap.add_argument('--site', type=int, nargs='+', help='ZERO-BASED site indices; default all sites')
    ap.add_argument('--methods', nargs='+', choices=list(METHODS), default=list(METHODS), help='methods in the error figure')
    ap.add_argument('--traj-methods', nargs='+', choices=list(METHODS), default=list(METHODS))
    ap.add_argument('--select', choices=['min', 'final'], default='min',
                    help='min: oracle minimum errors; final: supplied err_* and u fields')
    ap.add_argument('--error-units', choices=['percent', 'ratio'], required=True,
                    help='percent if stored 1 means 1%%; ratio if stored 0.01 means 1%%')
    ap.add_argument('--width-mm', type=float, default=180)
    args = ap.parse_args()
    if args.width_mm < 170:
        ap.error('Use --width-mm >= 170 for readable three-column panels.')
    if args.select == 'min':
        warnings.warn('min_* may be per-metric oracle minima at different iterations; '
                      'they need not correspond to u_min. Use --select final for supplied final/loss-selected estimates.')
    runs = load_runs(args.results, args.seeds, args.noise)
    args.out.mkdir(parents=True, exist_ok=True)
    fig_errors(runs, args)
    selection = fig_trajectories(runs, args)
    selection.update({'input_error_units': args.error_units, 'output_error_units': 'percent',
                      'boxplot': 'Q1 to Q3; median; whiskers 1.5 IQR; all trials overlaid',
                      'groups': {f'{m}, noise={n:g}': sorted(d) for (m, n), d in runs.items()}})
    (args.out / 'plot_manifest.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    print(f'Figures, plotted error values, and selection metadata written to {args.out.resolve()}')


if __name__ == '__main__':
    main()
