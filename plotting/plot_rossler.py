"""Rössler boxplots, ROC curves and trajectories using plot_nature's style.

python plot_rossler.py --experiment ../Rossler_hypergraph/results/benchmark
Only supplied final/loss-selected estimates are plotted; no oracle minima.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
import plot_nature as style

METHODS = list(style.METHODS)
NAMES = {m: style.METHODS[m][0] for m in METHODS}


def load(root, allow_partial=False):
    protocol=json.loads((root/'protocol.json').read_text())
    runs={}
    for method in METHODS:
        for noise in protocol['noises']:
            for seed in protocol['seeds']:
                path=root/'runs'/f'{method}_s{seed}_n{noise:g}.npz'
                if not path.exists():
                    if allow_partial:
                        continue
                    raise FileNotFoundError(f'Incomplete experiment: {path}')
                with np.load(path) as src:
                    runs[method,noise,seed]={k:src[k].copy() for k in src.files}
    return protocol,runs


def export(fig, axes, out, name):
    base=out/name
    style.save_geometry(fig,axes,base.with_suffix('.alignment.json'))
    for suffix in ['.pdf','.svg','.png']:
        fig.savefig(base.with_suffix(suffix),dpi=600 if suffix=='.png' else None)
    plt.close(fig)


def legend(fig):
    handles=[Line2D([],[],color=style.METHODS[m][1],marker=style.METHODS[m][3],
                    linewidth=1,label=NAMES[m],markersize=3) for m in METHODS]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,1.0),ncol=len(METHODS),frameon=False)


def boxes(protocol,runs,out,auc=False):
    if not auc:   # error figure: same code and style as the LV one (plot_nature.fig_errors): ratios, no percent sign
        grouped={}
        for (method,noise,seed),z in runs.items():
            grouped.setdefault((method,noise),{})[seed]=z
        style.fig_errors(grouped,argparse.Namespace(methods=METHODS,width_mm=180,select='final',
            error_units='percent',out=out),metrics=('w','v','traj'),name='rossler_error_boxplots',
            values_csv='rossler_error_boxplots_values.csv')
        return
    fig,axes=style.make_grid(1,180)
    rows=[]
    for col,(metric,title) in enumerate(zip(['auc2','auc3','aucall'],['Pairwise topology','Third-order topology','All hyperedges'])):
        ax=axes[0,col]
        style.panel_header(ax,'abc'[col],title)
        for group,noise in enumerate(protocol['noises']):
            for j,method in enumerate(METHODS):
                trials=[(s,runs[method,noise,s]) for s in protocol['seeds'] if (method,noise,s) in runs]
                if not trials:
                    continue
                vals=[float(z['err_'+metric]) for s,z in trials]
                if not np.isfinite(vals).all():
                    raise ValueError(f'Invalid plotted values: {method}, {noise}, {metric}')
                color=style.METHODS[method][1]
                pos=group+(j-(len(METHODS)-1)/2)*.25
                ax.boxplot([vals],positions=[pos],widths=.19,whis=1.5,showfliers=False,manage_ticks=False,patch_artist=True,
                    boxprops=dict(facecolor=to_rgba(color,.18),edgecolor=color,linewidth=.8),
                    medianprops=dict(color=color,linewidth=1.3),whiskerprops=dict(color=color,linewidth=.75),
                    capprops=dict(color=color,linewidth=.75))
                offsets=np.linspace(-.055,.055,len(vals)) if len(vals)>1 else [0.]
                ax.scatter(pos+np.asarray(offsets),vals,s=12,c=color,marker=style.METHODS[method][3],edgecolors='white',linewidths=.3,zorder=3)
                rows += [dict(metric=metric,method=method,seed=s,noise=noise,value=v,units='AUC') for (s,_),v in zip(trials,vals)]
        ax.set_xticks(range(len(protocol['noises'])),[f'{n*100:g}%' for n in protocol['noises']])
        ax.set(xlabel='Observation noise',ylabel='ROC AUC')
        ax.set_ylim(-.025,1.04)
    legend(fig)
    export(fig,axes,out,'rossler_auc_boxplots')
    with (out/'rossler_auc_boxplots_values.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def rocs(protocol,runs,out):
    fig,axes=style.make_grid(1,180)
    for col,noise in enumerate(protocol['noises']):
        ax=axes[0,col]
        style.panel_header(ax,'abc'[col],f'{noise*100:g}% observation noise')
        for method in METHODS:
            trials=[runs[method,noise,s] for s in protocol['seeds'] if (method,noise,s) in runs]
            if not trials:
                continue
            color=style.METHODS[method][1]
            grid=np.linspace(0.,1.,201)
            curves=np.array([np.interp(grid,z['rocall_fpr'],z['rocall_tpr']) for z in trials])
            for z in trials:
                ax.plot(z['rocall_fpr'],z['rocall_tpr'],color=color,alpha=.18,linewidth=.5)
            ax.plot(grid,curves.mean(0),color=color,linewidth=1.2,
                label=f"{NAMES[method]} ({np.mean([z['err_aucall'] for z in trials]):.2f})")
        ax.plot([0,1],[0,1],':',color='.65',linewidth=.7)
        ax.set(xlim=(0,1),ylim=(0,1.025),xlabel='False positive rate',ylabel='True positive rate')
        ax.legend(fontsize=6,loc='lower right',title='Mean AUC',title_fontsize=6)
    export(fig,axes,out,'rossler_roc')


def trajectories(protocol,runs,out,seed,nodes):
    for node in nodes:
        fig,axes=style.make_grid(len(protocol['noises']),180)
        for row,noise in enumerate(protocol['noises']):
            available=[m for m in METHODS if (m,noise,seed) in runs]
            if not available:
                continue
            base=runs[available[0],noise,seed]
            for col in range(3):
                ax=axes[row,col]
                style.panel_header(ax,chr(97+row*3+col),f"{'xyz'[col]}, noise {noise*100:g}%")
                ax.plot(base['t_eval'],base['x_true'][:,node,col],color='.15',linewidth=1.05,label='Clean reference',zorder=2)
                ax.scatter(base['t_obs'],base['y_obs'][:,node,col],s=2,color='.65',alpha=.45,edgecolors='none',label='Observed',zorder=1)
                for method in available:
                    z=runs[method,noise,seed]
                    color,line=style.METHODS[method][1:3]
                    ax.plot(z['t_eval'],z['u'][:,node,col],color=color,linestyle=line,linewidth=.9,label=NAMES[method],zorder=3)
                ax.set(xlabel='Time',ylabel=f"{'xyz'[col]} (node {node+1})")
        # One scale per state component across the noise rows; no data clipping.
        for col in range(3):
            low=min(ax.get_ylim()[0] for ax in axes[:,col]);high=max(ax.get_ylim()[1] for ax in axes[:,col])
            for ax in axes[:,col]:
                ax.set_ylim(low,high)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.54,1.),ncol=5,fontsize=6.5)
        export(fig,axes,out,f'rossler_trajectories_s{seed}_node{node+1}')


def summarize(root,protocol,runs,out):
    rows=[]
    for (method,noise,seed),z in sorted(runs.items()):
        row=dict(method=method,seed=seed,noise=noise,seconds=float(z['seconds']))
        row.update({k[4:]:float(v) for k,v in z.items() if k.startswith('err_')})
        rows.append(row)
    with (out/'all_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    meta=dict(completed=len(runs),expected=len(METHODS)*len(protocol['seeds'])*len(protocol['noises']),
        groups={f'{m}, noise={n:g}':[s for s in protocol['seeds'] if (m,n,s) in runs] for m in METHODS for n in protocol['noises']},
        selection='supplied final/loss-selected estimates; no oracle minima',trajectory_example=protocol['example_selection'],
        trajectory_reference='original solver dense interpolation; reconstruction/interpolation, not forecasting',
        units='relative errors in percent, AUC unitless',boxplot='Q1-Q3, median, 1.5 IQR whiskers and every trial point')
    (out/'plot_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--seed',type=int,help='trial of the trajectory figures (default: the protocol\'s example)')
    parser.add_argument('--nodes',type=int,nargs='+',help='zero-based nodes of the trajectory figures (default: the protocol\'s example)')
    args=parser.parse_args()
    out=args.out or args.experiment/'figures_nature'
    out.mkdir(parents=True,exist_ok=True)
    protocol,runs=load(args.experiment,args.allow_partial)
    if not runs:
        raise ValueError('No completed runs')
    summarize(args.experiment,protocol,runs,out)
    boxes(protocol,runs,out)
    boxes(protocol,runs,out,auc=True)
    rocs(protocol,runs,out)
    example=protocol['example_selection']
    trajectories(protocol,runs,out,args.seed if args.seed is not None else example['seed'],args.nodes or example['nodes_zero_based'])
