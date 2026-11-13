"""Figure: matched discrimination by curated evidence level, per model, per knowledge base.

Replaces the earlier version, which truncated the model list to the first six of ORDER
(silently dropping Gemini and both open-weight models) while still drawing a legend that
advertised them. All models with data are plotted, and each panel legends only the models
it actually contains.
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
# within a vendor, separate models by line style and marker
STYLE = {'o4-mini':        ('-',  'o'), 'gpt-4.1':     ('--', 's'),
         'gpt-4o':         ('-.', '^'), 'gpt-4o-mini': (':',  'D'),
         'claude-opus-5':  ('-',  'o'), 'claude-haiku-4-5': ('--', 's'),
         'gemini-2.5-flash': ('-', 'o'),
         'qwen2.5:7b':     ('-',  'o'), 'medgemma:4b': ('--', 's')}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
                     'axes.edgecolor': INK2, 'axes.linewidth': 0.6,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.labelcolor': INK, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'figure.facecolor': 'white',
                     'axes.facecolor': 'white', 'savefig.facecolor': 'white'})

bl = pd.read_csv('results/bylevel.csv')
kbs = [k for k in ['CIViC', 'JAX-CKB', 'CGI'] if k in set(bl.kb)]
fig, axes = plt.subplots(1, len(kbs), figsize=(2.55 * len(kbs) + 1.15, 2.55), sharey=True)
if len(kbs) == 1:
    axes = [axes]

for ax, kb in zip(axes, kbs):
    sub = bl[bl.kb == kb]
    levels = [L for L in ['A', 'B', 'C', 'D'] if L in set(sub.level)]
    models = [m for m in ORDER if m in set(sub.model)]
    for m in models:
        s = sub[sub.model == m].set_index('level').reindex(levels)
        if s.delta.notna().sum() < 2:
            continue
        ls, mk = STYLE.get(m, ('-', 'o'))
        ax.plot(range(len(levels)), s.delta.values, color=VCOL.get(VENDOR[m], INK2),
                lw=1.5, ls=ls, marker=mk, ms=4.2, mec='white', mew=0.7,
                alpha=0.92, zorder=3, label=LABEL[m])
    ax.axhline(0, color=INK2, lw=0.8, zorder=2)
    n = dict(sub.groupby('level').n.first())
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"{L}\nn={n[L]}" for L in levels], fontsize=8)
    ax.set_xlim(-0.25, len(levels) - 0.75)
    ax.set_ylim(-0.16, 1.04)
    ax.set_title(f"{kb}  ({len(models)} models)", fontsize=8.5, loc='left', pad=5, color=INK)
    ax.set_xlabel('Curated resistance evidence level', fontsize=8, labelpad=3)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)

axes[0].set_ylabel('Per-level difference $\\delta_\\ell$', fontsize=8)
axes[0].set_yticks([0, .25, .5, .75, 1.0])
axes[0].set_yticklabels(['0', '25%', '50%', '75%', '100%'])

# one legend for the whole figure, naming every model actually drawn
seen, handles = [], []
for m in ORDER:
    if m in set(bl.model) and m not in seen:
        ls, mk = STYLE.get(m, ('-', 'o'))
        handles.append(Line2D([], [], color=VCOL.get(VENDOR[m], INK2), lw=1.5, ls=ls,
                              marker=mk, ms=4.2, mec='white', mew=0.7, label=LABEL[m]))
        seen.append(m)
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.30),
           frameon=False, fontsize=7.2, ncol=5, columnspacing=1.5, handlelength=2.4)
fig.savefig('paper/figures/fig_bylevel.pdf', bbox_inches='tight', dpi=300)
plt.close(fig)
print(f"fig_bylevel: {len(kbs)} panels, {len(seen)} models")
