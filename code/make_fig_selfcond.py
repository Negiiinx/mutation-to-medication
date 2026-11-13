"""Paired-slope figure for the self-conditioning arm and its false-attribution control.

Left: endorsement of resistance cases at baseline and with the model's own conclusion quoted.
Right: the same cases under a false attribution, which is where the interpretation turns.
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from anlib import LABEL, VENDOR, ORDER

BLUE, ORANGE, AQUA, PURPLE = '#2a78d6', '#eb6834', '#1baf7a', '#6a4fc9'
INK, INK2, GRID = '#1a1a19', '#52514e', '#e8e7e3'
VCOL = {'OpenAI': BLUE, 'Anthropic': ORANGE, 'Google': AQUA, 'open': PURPLE}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
                     'axes.edgecolor': INK2, 'axes.linewidth': 0.6,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.labelcolor': INK, 'figure.facecolor': 'white',
                     'axes.facecolor': 'white', 'savefig.facecolor': 'white'})

C = pd.read_csv('results/controls.csv').set_index('model')
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.05), sharey=True)

panels = [('Own conclusion quoted', 'base_yes', 'self_yes', ('Baseline', 'Own'), False),
          ('Opposite conclusion attributed', 'base_yes', 'false_yes', ('Baseline', 'False'), True)]
for ax, (title, c0, c1, labs, do_label) in zip(axes, panels):
    for m in ORDER:
        if m not in C.index:
            continue
        r = C.loc[m]
        col = VCOL.get(VENDOR[m], INK2)
        ax.plot([0, 1], [r[c0], r[c1]], color=col, lw=1.6, marker='o', ms=5,
                mec='white', mew=0.8, alpha=0.92, zorder=3)
        if do_label:
            _off = {'gpt-4o': 3.0, 'gpt-4o-mini': -3.0, 'o4-mini': -3.2,
                    'claude-opus-5': 3.2}.get(m, 0.0)
            ax.annotate(LABEL[m], (1.04, r[c1] + _off), fontsize=6.6, color=col,
                        va='center', ha='left', annotation_clip=False)
    ax.set_xlim(-0.12, 1.62 if do_label else 1.12)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylim(-3, 100)
    ax.set_title(title, fontsize=8.5, loc='left', pad=6, color=INK)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)

axes[0].set_ylabel('Endorsement of resistance cases (%)', fontsize=8)
axes[0].set_yticks([0, 25, 50, 75, 100])
fig.legend(handles=[Line2D([], [], color=VCOL[v], lw=1.6, marker='o', ms=5,
                           mec='white', mew=0.8) for v in ['OpenAI', 'Anthropic', 'Google']],
           labels=['OpenAI', 'Anthropic', 'Google'], loc='lower center',
           bbox_to_anchor=(0.5, -0.10), frameon=False, fontsize=7.4, ncol=3)
fig.savefig('paper/figures/fig_selfcond.pdf', bbox_inches='tight', dpi=300)
plt.close(fig)
print('fig_selfcond: 2 panels,', len(C), 'models')
