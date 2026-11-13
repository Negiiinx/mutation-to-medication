"""Matched estimands, stability checks and the three safety endpoints.

1. Gene-matched Delta with cluster-bootstrap intervals, so it can serve as the primary
   estimator rather than a robustness check. Matching is on (gene, evidence level):
   a model is credited only for separating directions *within the same gene*, which
   removes gene identity, drug familiarity and cancer type from the contrast.
2. Leave-one-gene-out: recompute both estimators dropping each gene in turn, to show
   whether any single gene drives the point estimate or the ranking.
3. Safety endpoints reported as three explicit dimensions rather than one number:
   P(strong rec | resistance), P(endorse | sensitivity), and gene-matched Delta.

Writes results/genematched.csv, results/logo.csv, results/safety.csv.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from anlib import load, ORDER, LABEL, VENDOR, LEVS

D = load('data/cohort_direction.csv', 'dir')
D = D[D.model.isin(ORDER)].reset_index(drop=True)

MIN_CELL = 2   # minimum cases per arm within a (gene, level) cell


def _cells(g):
    """(weight, delta) per (gene, level) cell holding both arms."""
    out = []
    for (gene, lv), sub in g.groupby(['gene', 'best_level']):
        R = sub[sub.label == 'RESISTANT']
        S = sub[sub.label == 'SENSITIVE']
        if len(R) >= MIN_CELL and len(S) >= MIN_CELL:
            out.append((gene, len(R), S.endorse.mean() - R.endorse.mean()))
    return out


def delta_gene(g, B=1500, seed=17):
    """Gene-matched Delta with a cluster bootstrap over genes."""
    cells = _cells(g)
    if len(cells) < 5:
        return np.nan, np.nan, np.nan, 0, 0
    w = np.array([c[1] for c in cells], float)
    d = np.array([c[2] for c in cells], float)
    pt = np.average(d, weights=w)
    genes = sorted({c[0] for c in cells})
    by = {gn: [(c[1], c[2]) for c in cells if c[0] == gn] for gn in genes}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(B):
        pick = rng.choice(genes, len(genes), replace=True)
        ww, dd = [], []
        for gn in pick:
            for wi, di in by[gn]:
                ww.append(wi); dd.append(di)
        if ww:
            boot.append(np.average(dd, weights=ww))
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
    return pt, lo, hi, len(cells), len(genes)


def delta_level(g):
    """The evidence-level-matched Delta (previous headline), for comparison."""
    lv = np.asarray(g.best_level); res = np.asarray(g.label) == 'RESISTANT'
    e = g.endorse.values
    ok = [l for l in LEVS if ((lv == l) & res).sum() >= 5 and ((lv == l) & ~res).sum() >= 5]
    if not ok:
        return np.nan
    w = np.array([((lv == l) & res).sum() for l in ok], float)
    per = [e[(lv == l) & ~res].mean() - e[(lv == l) & res].mean() for l in ok]
    return np.average(per, weights=w)


# ---------------------------------------------------------------- 1. gene-matched
rows = []
for m in ORDER:
    g = D[D.model == m].reset_index(drop=True)
    if not len(g):
        continue
    pt, lo, hi, ncell, ngene = delta_gene(g)
    rows.append(dict(model=m, vendor=VENDOR[m], d_gene=pt, lo=lo, hi=hi,
                     n_cells=ncell, n_genes=ngene, d_level=delta_level(g)))
gm = pd.DataFrame(rows)
gm.to_csv('results/genematched.csv', index=False)
print('genematched.csv:', len(gm), 'models,',
      f'{gm.n_cells.iloc[0]} matched cells over {gm.n_genes.iloc[0]} genes')

# ------------------------------------------------------------ 2. leave-one-gene-out
allg = sorted(D.gene.unique())
logo = []
for m in ORDER:
    g = D[D.model == m].reset_index(drop=True)
    if not len(g):
        continue
    full_gene = delta_gene(g, B=1)[0]
    full_lev = delta_level(g)
    for drop in allg:
        h = g[g.gene != drop]
        logo.append(dict(model=m, dropped=drop,
                         d_gene=delta_gene(h, B=1)[0], d_level=delta_level(h),
                         full_gene=full_gene, full_lev=full_lev))
lg = pd.DataFrame(logo)
lg.to_csv('results/logo.csv', index=False)

sw = lg.groupby('model').agg(gene_min=('d_gene', 'min'), gene_max=('d_gene', 'max'),
                             lev_min=('d_level', 'min'), lev_max=('d_level', 'max'))
sw['gene_swing'] = sw.gene_max - sw.gene_min
sw['lev_swing'] = sw.lev_max - sw.lev_min
print('\nleave-one-gene-out swing (pp), gene-matched vs level-matched:')
for m in ORDER:
    if m in sw.index:
        r = sw.loc[m]
        print(f'  {LABEL[m]:18s} gene {100*r.gene_swing:5.1f}   level {100*r.lev_swing:5.1f}')

# ranking stability: Spearman of the per-model ranking, full vs each leave-one-out
from scipy.stats import spearmanr
base = gm.set_index('model').d_gene
rhos = []
for drop in allg:
    sub = lg[lg.dropped == drop].set_index('model').d_gene
    common = [m for m in ORDER if m in base.index and m in sub.index]
    if len(common) > 2:
        rhos.append(spearmanr(base[common], sub[common]).correlation)
print(f'\nranking stability (Spearman vs full, gene-matched): '
      f'min {min(rhos):.3f}, median {np.median(rhos):.3f}, over {len(rhos)} genes')
pd.DataFrame(dict(dropped=allg, rho=rhos)).to_csv('results/logo_rank.csv', index=False)

# ------------------------------------------------------------------ 3. safety endpoints
rows = []
for m in ORDER:
    g = D[D.model == m]
    if not len(g):
        continue
    R = g[g.label == 'RESISTANT']; S = g[g.label == 'SENSITIVE']
    pt, lo, hi, _, _ = delta_gene(g)
    rows.append(dict(model=m, vendor=VENDOR[m],
                     strong_R=100 * R.strongrec.mean(),      # primary safety endpoint
                     endorse_R=100 * R.endorse.mean(),
                     endorse_S=100 * S.endorse.mean(),       # sensitivity preservation
                     strong_S=100 * S.strongrec.mean(),
                     d_gene=100 * pt, lo=100 * lo, hi=100 * hi))
sf = pd.DataFrame(rows)
sf.to_csv('results/safety.csv', index=False)
print('\nsafety.csv written')
print(sf[['model', 'strong_R', 'endorse_R', 'endorse_S', 'd_gene']].round(1).to_string(index=False))


# ------------------------------------- 4. wild cluster bootstrap for the gene-matched estimator
# Percentile intervals are anti-conservative with few clusters, and the gene-matched estimator
# rests on 17 genes with qualifying cells. This imposes the within-cell null directly: the
# direction effect is set to zero inside every (gene, level) cell, residuals are sign-flipped by
# gene cluster (Rademacher), and Delta_gene is recomputed per replicate.
def wild_gene(g, B=4999, seed=29):
    cells, fit, resid, gidx, arm = [], [], [], [], []
    genes = sorted(g.gene.unique())
    gi = {gn: i for i, gn in enumerate(genes)}
    recs = []
    for (gn, lv), sub in g.groupby(['gene', 'best_level']):
        R = sub[sub.label == 'RESISTANT']; S = sub[sub.label == 'SENSITIVE']
        if len(R) < MIN_CELL or len(S) < MIN_CELL:
            continue
        mu = sub.endorse.mean()                       # cell mean under the null
        for _, row in sub.iterrows():
            recs.append((gi[gn], gn, lv, row.label == 'RESISTANT', row.endorse, mu))
    if len(recs) < 10:
        return np.nan, np.nan
    gidx = np.array([r[0] for r in recs])
    key = [(r[1], r[2]) for r in recs]
    res = np.array([r[3] for r in recs])
    e = np.array([r[4] for r in recs], float)
    mu = np.array([r[5] for r in recs], float)
    resid = e - mu

    def delta_of(vals):
        num = den = 0.0
        for k in sorted(set(key)):   # sorted, not set order: set iteration depends on
                                     # per-process string hashing, which perturbs the summation
                                     # order and flips tied bootstrap replicates
            sel = np.array([kk == k for kk in key])
            a = vals[sel & ~res]; b = vals[sel & res]
            if len(a) and len(b):
                num += len(b) * (a.mean() - b.mean()); den += len(b)
        return num / den if den else np.nan

    obs = delta_of(e)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(B):
        sgn = np.where(rng.random(len(genes)) < 0.5, 1.0, -1.0)
        star = mu + sgn[gidx] * resid
        d = delta_of(star)
        if not np.isnan(d) and abs(d) >= abs(obs):
            cnt += 1
    return obs, (cnt + 1) / (B + 1)


rows = []
for m in ORDER:
    g = D[D.model == m].reset_index(drop=True)
    if not len(g):
        continue
    obs, pv = wild_gene(g)
    rows.append(dict(model=m, d_gene=obs, wild_p=pv))
wg = pd.DataFrame(rows)
wg.to_csv('results/wild_gene.csv', index=False)
print('\nwild cluster bootstrap under the within-cell null, gene-matched Delta:')
for _, r in wg.iterrows():
    print(f'  {LABEL[r.model]:18s} d_gene {100*r.d_gene:+6.1f}   p = {r.wild_p:.4f}')


# ------------------------- 5. gene-matched Delta for the mitigation arm, before and after
# The oracle arm is estimable on the same 28 cells / 17 genes, so the mitigation table can
# report the primary estimator rather than the level-matched one.
RAGD = load('data/cohort_direction_rag.csv', 'rag')
rows = []
if RAGD is not None:
    for m in ORDER:
        b = D[D.model == m].reset_index(drop=True)
        a = RAGD[RAGD.model == m].reset_index(drop=True)
        if not len(b) or not len(a):
            continue
        pb, lb, hb, _, _ = delta_gene(b)
        pa, la, ha, nc, ng = delta_gene(a)
        rows.append(dict(model=m, vendor=VENDOR[m], dg_before=pb, dg_after=pa,
                         a_lo=la, a_hi=ha, n_cells=nc, n_genes=ng))
mg = pd.DataFrame(rows)
mg.to_csv('results/mitigation_gene.csv', index=False)
print('\ngene-matched Delta, mitigation arm (before -> after):')
for _, r in mg.iterrows():
    print(f'  {LABEL[r.model]:18s} {100*r.dg_before:+6.1f} -> {100*r.dg_after:+6.1f} '
          f'[{100*r.a_lo:+.0f},{100*r.a_hi:+.0f}]  cells={int(r.n_cells)} genes={int(r.n_genes)}')
