"""Tables for the control, cross-source and robustness arms.

Writes the six new tables the revision \\input's, into tables/.  Every number is
derived here from the raw response files; nothing is transcribed by hand.
"""
import warnings; warnings.filterwarnings('ignore')
import json, glob, os
import numpy as np, pandas as pd

TAIL = '\n\\bottomrule\n\\end{tabular}\n'
from scipy.stats import spearmanr, rankdata
from anlib import load, matched, ORDER, LABEL, VENDOR, LEVS, REC

CO = pd.read_csv('data/cohort_direction.csv')
def w(name, s):
    open(f'tables/{name}.tex', 'w').write(s); print(f'  tables/{name}.tex')

def arm(tag, cohort=None):
    """Successful judgments for an arm, joined to the cohort."""
    co = CO if cohort is None else pd.read_csv(cohort)
    rows = []
    for f in glob.glob(f'raw_responses/{tag}_*.jsonl'):
        for L in open(f):
            try: r = json.loads(L)
            except Exception: continue
            q = r.get('resp', {})
            if 'recommendation' not in q: continue
            rows.append({'uid': r['uid'], 'model': r['model'], 'rep': int(r.get('rep', 0)),
                         'rec': q['recommendation']})
    if not rows: return None
    d = pd.DataFrame(rows).drop_duplicates(['uid', 'model', 'rep'])
    d = d.merge(co[['uid', 'label', 'gene', 'best_level']], on='uid')
    d['endorse'] = (d.rec.map(REC) >= 3).astype(int)
    d['strongrec'] = (d.rec == 'Strongly recommend').astype(int)
    return d

MIN_CELL = 2   # the estimator's cell rule, matching reviewer_analyses.MIN_CELL

def _cells(d):
    """(gene, weight, delta) per (gene, level) cell holding MIN_CELL cases in both arms."""
    out = []
    for (gene, _), s in d.groupby(['gene', 'best_level']):
        a = s[s.label == 'SENSITIVE']; b = s[s.label == 'RESISTANT']
        if len(a) >= MIN_CELL and len(b) >= MIN_CELL:
            out.append((gene, len(b), a.endorse.mean() - b.endorse.mean()))
    return out

def dgene(d):
    """Stratum-weighted risk difference over qualifying (gene, level) cells."""
    c = _cells(d)
    return np.average([x[2] for x in c], weights=[x[1] for x in c]) if c else np.nan

BASE, SELF, NOI, IO = arm('dir'), arm('self'), arm('noinstr'), arm('instronly')
DIS, RAG = arm('distract'), arm('rag')

# ---------------------------------------------------------------- the 2x2
YES = 'Yes, confers resistance'
know = {}
for f in glob.glob('raw_responses/know_*.jsonl'):
    for L in open(f):
        try: r = json.loads(L)
        except Exception: continue
        if 'resistance_known' in r.get('resp', {}):
            know.setdefault((r['model'], r['uid']), r['resp']['resistance_known'])

rows, eff = [], []
for m in ORDER:
    keep = {u for (mm, u), v in know.items() if mm == m and v == YES}
    if len(keep) < 30: continue
    def pct(D):
        g = D[(D.model == m) & (D.label == 'RESISTANT') & (D.uid.isin(keep))]
        return (100 * g.endorse.mean(), len(g)) if len(g) > 20 else (np.nan, len(g))
    (b, nb), (i, _), (c, _), (s_, _) = pct(BASE), pct(IO), pct(NOI), pct(SELF)
    f = lambda x: '--' if pd.isna(x) else f'{x:.1f}'
    rows.append(f'{LABEL[m]} & {nb} & {f(b)} & {f(i)} & {f(c)} & {f(s_)} \\\\')
    if not any(pd.isna(x) for x in (b, i, c, s_)): eff.append((i - b, c - b, s_ - b))
E = np.array(eff)
w('twobytwo', '\\begin{tabular}{lrrrrr}\n\\toprule\n'
  '& & \\multicolumn{4}{c}{Endorsement of resistance cases (\\%)} \\\\\n\\cmidrule(lr){3-6}\n'
  '& & Neither & Instruction & Conclusion & Both \\\\\n'
  'Model & $n$ & (baseline) & only & only & (self) \\\\\n\\midrule\n'
  + '\n'.join(rows) + '\n\\midrule\n'
  + f'\\emph{{Mean shift from baseline}} & & -- & ${E[:,0].mean():+.1f}$ & ${E[:,1].mean():+.1f}$ '
    f'& ${E[:,2].mean():+.1f}$ \\\\' + TAIL)

# ---------------------------------------------------------------- distractor
rows = []
for m in ORDER:
    g = lambda D: D[(D.model == m) & (D.label == 'RESISTANT')]
    b, d, r = g(BASE), g(DIS), g(RAG)
    if not len(d): continue
    rows.append(f'{LABEL[m]} & {VENDOR[m].replace("open","Open-weight")} & {100*b.endorse.mean():.1f} & '
                f'{100*d.endorse.mean():.1f} & {100*r.endorse.mean():.1f} \\\\')
w('distractor', '\\begin{tabular}{llrrr}\n\\toprule\n'
  '& & \\multicolumn{3}{c}{Endorsement of resistance cases (\\%)} \\\\\n\\cmidrule(lr){3-5}\n'
  '& & No block & Distractor & Scored entry \\\\\n'
  'Model & Vendor & (baseline) & (sensitising) & (oracle) \\\\\n\\midrule\n'
  + '\n'.join(rows) + TAIL)

# ---------------------------------------------------------------- oracle split
rows = []
for m in ORDER:
    g = RAG[(RAG.model == m) & (RAG.label == 'RESISTANT')]
    if not len(g): continue
    e = 100 * g.endorse.mean(); st = 100 * g.strongrec.mean()
    rows.append(f'{LABEL[m]} & {len(g)} & {e:.1f} & {st:.1f} & {e-st:.1f} & {100*st/e if e else 0:.0f} \\\\')
w('oraclesplit', '\\begin{tabular}{lrrrrr}\n\\toprule\n'
  '& & \\multicolumn{3}{c}{Under oracle evidence (\\% of cases)} & Strong as \\\\\n\\cmidrule(lr){3-5}\n'
  'Model & $n$ & Endorsed & Strongly rec. & Consider & \\% of endorsed \\\\\n\\midrule\n'
  + '\n'.join(rows) + TAIL)

# ---------------------------------------------------------------- temperature
def tload(m):
    rows = []
    for suf in ('', '_sens'):
        p = f'raw_responses/temp{suf}_{m}.jsonl'
        if not os.path.exists(p): continue
        for L in open(p):
            try: r = json.loads(L)
            except Exception: continue
            if 'recommendation' not in r.get('resp', {}): continue
            rows.append({'uid': r['uid'], 'rep': int(r['rep']), 'rec': r['resp']['recommendation']})
    if not rows: return None
    d = pd.DataFrame(rows).drop_duplicates(['uid', 'rep']).merge(
        CO[['uid', 'label', 'gene', 'best_level']], on='uid')
    d['endorse'] = (d.rec.map(REC) >= 3).astype(int)
    d['strongrec'] = (d.rec == 'Strongly recommend').astype(int)
    return d

gm = pd.read_csv('results/genematched.csv').set_index('model')
h = pd.read_csv('results/headline.csv').set_index('model')
rows = []
# any model with a temperature arm on disk, in the canonical order
_tmods = [m for m in ORDER
          if os.path.exists(f"raw_responses/temp_{m.replace(':', '-')}.jsonl")]
for m in _tmods:
    d = tload(m)
    if d is None: continue
    reps = sorted(d.rep.unique())
    er = [100 * d[(d.rep == k) & (d.label == 'RESISTANT')].endorse.mean() for k in reps]
    dl = [matched(d[d.rep == k])[0][0] for k in reps]
    dg = [dgene(d[d.rep == k]) for k in reps]
    una = d[d.label == 'RESISTANT'].pivot_table(index='uid', columns='rep', values='endorse').dropna()
    rows.append(f'{LABEL[m]} & {100*h.loc[m,"endorse"]:.1f} & ' + '/'.join(f'{x:.1f}' for x in er) +
                f' & {100*gm.loc[m,"d_level"]:+.1f} & ' + '/'.join(f'{100*x:+.1f}' for x in dl) +
                f' & {100*gm.loc[m,"d_gene"]:+.1f} & ' + '/'.join(f'{100*x:+.1f}' for x in dg) +
                f' & {100*(una.nunique(axis=1)==1).mean():.1f} \\\\')
w('temp', '\\begin{tabular}{lrrrrrrr}\n\\toprule\n'
  '& \\multicolumn{2}{c}{Endorse, resistance (\\%)} & \\multicolumn{2}{c}{$\\Delta_{\\mathrm{lvl}}$} '
  '& \\multicolumn{2}{c}{$\\Delta_{\\mathrm{gene}}$} & Per-case \\\\\n'
  '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n'
  'Model & $T{=}0$ & $T{=}1$, $k{=}3$ & $T{=}0$ & $T{=}1$, $k{=}3$ & $T{=}0$ & $T{=}1$, $k{=}3$ '
  '& unanimity \\\\\n\\midrule\n' + '\n'.join(rows) + TAIL)

# ---------------------------------------------------------------- CGI, five models
C = load('data/cohort_cgi_clean.csv', 'cgi2')
rows = []
for m in ORDER:
    g = C[C.model == m]
    if not len(g): continue
    R = g[g.label == 'RESISTANT']; S = g[g.label == 'SENSITIVE']
    per = []
    for l in LEVS:
        a = S[S.best_level == l]; b = R[R.best_level == l]
        per.append(f'${100*(a.endorse.mean()-b.endorse.mean()):+.1f}$' if len(a) >= 5 and len(b) >= 5 else '--')
    rows.append(f'{LABEL[m]} & {VENDOR[m]} & {100*R.endorse.mean():.1f} & {100*S.endorse.mean():.1f} & '
                + ' & '.join(per) + ' \\\\')
w('crosscgi', '\\begin{tabular}{llrrrrrr}\n\\toprule\n'
  '& & \\multicolumn{2}{c}{Endorsement (\\%)} & \\multicolumn{4}{c}{$\\delta_{\\ell}$ by evidence level} \\\\\n'
  '\\cmidrule(lr){3-4}\\cmidrule(lr){5-8}\n'
  'Model & Vendor & Resistance & Control & A & B & C & D \\\\\n\\midrule\n'
  + '\n'.join(rows) + TAIL)

# ---------------------------------------------------------------- susceptibility scales
c = pd.read_csv('results/controls.csv')
c['dg'] = 100 * c.model.map(gm.d_gene)
c['sh'] = c.false_yes - c.base_yes
c['nm'] = 100 * (c.false_yes - c.base_yes) / (100 - c.base_yes)
rows = [f'{LABEL[r.model]} & ${r.dg:+.1f}$ & {r.base_yes:.1f} & {r.false_yes:.1f} & '
        f'${r.sh:+.1f}$ & {r.nm:.1f} \\\\' for _, r in c.iterrows()]
def prho(x, y, z):
    x, y, z = (rankdata(v) for v in (x, y, z))
    rxy, rxz, ryz = (np.corrcoef(a, b)[0, 1] for a, b in ((x, y), (x, z), (y, z)))
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
sp = lambda col: spearmanr(c.dg, c[col])
w('suscept', '\\begin{tabular}{lrrrrr}\n\\toprule\n'
  '& & \\multicolumn{2}{c}{Endorsement (\\%)} & \\multicolumn{2}{c}{Susceptibility} \\\\\n'
  '\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n'
  'Model & $\\Delta_{\\mathrm{gene}}$ & Baseline & False attrib. & Shift & Normalised \\\\\n\\midrule\n'
  + '\n'.join(rows) + '\n\\midrule\n'
  + f'\\emph{{Spearman $\\rho$ with $\\Delta_{{\\mathrm{{gene}}}}$}} & & ${sp("base_yes")[0]:+.2f}$ '
    f'& ${sp("false_yes")[0]:+.2f}$ & ${sp("sh")[0]:+.2f}$ & ${sp("nm")[0]:+.2f}$ \\\\\n'
  + f'\\emph{{$p$}} & & {sp("base_yes")[1]:.3f} & {sp("false_yes")[1]:.3f} & {sp("sh")[1]:.3f} '
    f'& {sp("nm")[1]:.3f} \\\\\n'
  + f'\\emph{{Partial $\\rho$, baseline removed}} & & -- & ${prho(c.false_yes,c.dg,c.base_yes):+.2f}$ '
    f'& -- & ${prho(c.nm,c.dg,c.base_yes):+.2f}$ \\\\' + TAIL)

# ---------------------------------------------------------------- GPT-5 vintage
def g5(tag):
    rows = []
    for L in open(f'raw_responses/{tag}_gpt-5.jsonl'):
        try: r = json.loads(L)
        except Exception: continue
        if 'recommendation' not in r.get('resp', {}): continue
        rows.append({'uid': r['uid'], 'rec': r['resp']['recommendation']})
    d = pd.DataFrame(rows).drop_duplicates('uid').merge(
        CO[['uid', 'label', 'gene', 'best_level']], on='uid')
    d['endorse'] = (d.rec.map(REC) >= 3).astype(int)
    d['strongrec'] = (d.rec == 'Strongly recommend').astype(int)
    return d
rows = []
for m in ORDER:
    r = h.loc[m]; g = gm.loc[m]
    rows.append((g.d_gene, f'{LABEL[m]} & {VENDOR[m].replace("open","Open-weight")} & '
                 f'{100*r.strong:.1f} & {100*r.endorse:.1f} & {100*r.sens:.1f} & ${100*g.d_gene:+.1f}$ \\\\'))
d = g5('dir'); R = d[d.label == 'RESISTANT']; S = d[d.label == 'SENSITIVE']
rows.append((dgene(d), f'\\textbf{{GPT-5}} & OpenAI & {100*R.strongrec.mean():.1f} & '
             f'{100*R.endorse.mean():.1f} & {100*S.endorse.mean():.1f} & '
             f'$\\mathbf{{{100*dgene(d):+.1f}}}$ \\\\'))
rows.sort(key=lambda t: -t[0])
w('vintage', '\\begin{tabular}{llrrrr}\n\\toprule\n'
  '& & \\multicolumn{2}{c}{Resistance arm (\\%)} & Control & \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-5}\n'
  'Model & Vendor & Strong rec. & Endorse & Endorse & $\\Delta_{\\mathrm{gene}}$ \\\\\n\\midrule\n'
  + '\n'.join(r[1] for r in rows) + TAIL)
print('done')

# ---------------------------------------------------------------- paraphrase templates
# Template 1 is the headline arm; templates 2 and 3 are the two paraphrase arms, on the
# same cohort, so Delta_lvl is directly comparable across the three.
T2, T3 = arm('parat2'), arm('parat3')
rows = []
for m in ORDER:
    g = BASE[BASE.model == m]
    if T2 is None or m not in set(T2.model):
        continue
    vals = [matched(d[d.model == m])[0][0] for d in (BASE, T2, T3)]
    raw = [100 * d[(d.model == m) & (d.label == 'RESISTANT')].endorse.mean() for d in (BASE, T2, T3)]
    rows.append(f'{LABEL[m]} & ' + ' & '.join(f'${100*v:+.1f}$' for v in vals) + ' & '
                + ' / '.join(f'{r:.1f}' for r in raw) + ' \\\\')
w('para', '\\begin{tabular}{lrrrr}\n\\toprule\n'
  '& \\multicolumn{3}{c}{$\\Delta_{\\mathrm{lvl}}$ by prompt template} & Endorsement of \\\\\n'
  '\\cmidrule(lr){2-4}\n'
  'Model & Template 1 & Template 2 & Template 3 & resistance cases (\\%) \\\\\n\\midrule\n'
  + '\n'.join(rows) + TAIL)

# ---------------------------------------------------------------- low-tier cells with intervals
# Review asked for intervals on the negative per-level cells rather than the phrase
# "approaching zero". These are the C and D cells of the two cross-source cohorts that
# have them; CIViC's own low tiers are in tables/bylevel.tex and are all positive.
def _cb_delta(a, b, ga, gb, B=3000, seed=23):
    rng = np.random.default_rng(seed)
    gs = np.unique(np.concatenate([ga, gb]))
    out = []
    for _ in range(B):
        pick = rng.choice(gs, len(gs), replace=True)
        xa = np.concatenate([a[ga == g] for g in pick])
        xb = np.concatenate([b[gb == g] for g in pick])
        if len(xa) and len(xb):
            out.append(xa.mean() - xb.mean())
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)

rows = []
for src, f, tag in [('JAX-CKB', 'cohort_jax_clean.csv', 'jax2'),
                    ('CGI', 'cohort_cgi_clean.csv', 'cgi2')]:
    D = load('data/' + f, tag)
    if D is None:
        continue
    first = True
    for lv in ('C', 'D'):
        for m in ORDER:
            g = D[D.model == m]
            if not len(g):
                continue
            A = g[(g.best_level == lv) & (g.label == 'SENSITIVE')]
            Bb = g[(g.best_level == lv) & (g.label == 'RESISTANT')]
            if len(A) < 5 or len(Bb) < 5:
                continue
            dlt = 100 * (A.endorse.mean() - Bb.endorse.mean())
            lo, hi = _cb_delta(A.endorse.values, Bb.endorse.values, A.gene.values, Bb.gene.values)
            rows.append(f"{src if first else ''} & {lv} & {LABEL[m]} & {len(Bb)} & {len(A)} & "
                        f"${dlt:+.1f}$ & $[{100*lo:+.1f}, {100*hi:+.1f}]$ \\\\")
            first = False
    rows.append('\\addlinespace')
w('lowtier', '\\begin{tabular}{lllrrrr}\n\\toprule\n'
  'Cohort & Level & Model & $n$ res. & $n$ ctrl. & $\\delta_{\\ell}$ & 95\\% CI \\\\\n\\midrule\n'
  + '\n'.join(rows[:-1]) + TAIL)
