"""Re-analyses requested in review, none of which need new inference calls.

1. Knowledge-base independence: pairwise overlap in (gene, alteration, therapy) tuples.
2. Cluster-bootstrap intervals on the primary safety endpoint, P(strong rec | resistance).
3. Delta_gene with strong-recommendation-only as the outcome.
4. Delta_gene at cell thresholds 2, 3, 4, with the weight sitting in n=2 cells.
5. A gene-averaged Delta_gene alongside the case-averaged one.
6. Delta_gene computed separately on the pre-registered EXPLORE and CONFIRM halves.
7. Delta_lvl intervals, which the text claims but no table showed.
8. Level-by-cohort case counts.
9. The familiarity confound: probe-endorsement gap stratified by curation density.
"""
import warnings; warnings.filterwarnings('ignore')
import glob, json, os, re
import numpy as np, pandas as pd
from anlib import load, ORDER, LABEL, VENDOR, LEVS

OUT = {}
D = load('data/cohort_direction.csv', 'dir')
D = D[D.model.isin(ORDER)].reset_index(drop=True)


def norm(x):
    return re.sub(r'[^a-z0-9]', '', str(x).lower())


# ---------------------------------------------- 1. knowledge-base independence
# cohort_final.csv carries no direction column, so CIViC must come from the direction cohort;
# using the former silently counted both arms for CIViC and only resistance for the others.
SRC = {'CIViC': 'data/cohort_direction.csv', 'JAX-CKB': 'data/cohort_jax_clean.csv',
       'CGI': 'data/cohort_cgi_clean.csv', 'MOAlmanac': 'data/cohort_moa_clean.csv',
       'OncoKB': 'data/cohort_oncokb.csv'}
tup = {}
for name, path in SRC.items():
    if not os.path.exists(path):
        continue
    d = pd.read_csv(path)
    col = 'molecular_profile' if 'molecular_profile' in d.columns else 'variant'
    ther = 'therapy' if 'therapy' in d.columns else 'therapies'
    assert 'label' in d.columns, f'{name}: no direction column, cannot isolate the resistance arm'
    keep = d[d.label == 'RESISTANT']
    tup[name] = {(norm(r[col]), norm(r[ther])) for _, r in keep.iterrows()}
names = list(tup)
rows = []
for i, a in enumerate(names):
    for b in names[i + 1:]:
        inter = len(tup[a] & tup[b])
        rows.append(dict(a=a, b=b, n_a=len(tup[a]), n_b=len(tup[b]), shared=inter,
                         jaccard=inter / max(len(tup[a] | tup[b]), 1),
                         pct_of_smaller=100 * inter / max(min(len(tup[a]), len(tup[b])), 1)))
OV = pd.DataFrame(rows)
OV.to_csv('results/kb_overlap.csv', index=False)
print('=== 1. knowledge-base overlap, resistance arms, (alteration, therapy) tuples ===')
for _, r in OV.iterrows():
    print(f"  {r.a:10s} vs {r.b:10s}  n={r.n_a:5d}/{r.n_b:5d}  shared={r.shared:4d}  "
          f"{r.pct_of_smaller:5.1f}% of smaller  Jaccard={r.jaccard:.3f}")

# --------------------------- 2. intervals on the primary safety endpoint, and 3/7
def cb_mean(vals, genes, B=2000, seed=41):
    vals = np.asarray(vals, float); genes = np.asarray(genes)
    gs = np.unique(genes)
    if len(gs) < 2:
        return np.nan, np.nan
    by = {g: vals[genes == g] for g in gs}
    rng = np.random.default_rng(seed)
    out = [np.concatenate([by[g] for g in rng.choice(gs, len(gs), replace=True)]).mean()
           for _ in range(B)]
    return tuple(np.percentile(out, [2.5, 97.5]))


def dgene(g, outcome='endorse', thr=2, average='case', B=1500, seed=17):
    cells = []
    for (gn, lv), sub in g.groupby(['gene', 'best_level']):
        R = sub[sub.label == 'RESISTANT']; S = sub[sub.label == 'SENSITIVE']
        if len(R) >= thr and len(S) >= thr:
            cells.append((gn, len(R), S[outcome].mean() - R[outcome].mean()))
    if len(cells) < 5:
        return dict(pt=np.nan, lo=np.nan, hi=np.nan, n_cells=0, n_genes=0, w2=np.nan)
    w = np.array([c[1] for c in cells], float)
    d = np.array([c[2] for c in cells], float)
    if average == 'gene':
        # equal weight per gene, cells averaged within gene first
        per = {}
        for gn, wi, di in cells:
            per.setdefault(gn, []).append((wi, di))
        d = np.array([np.average([x[1] for x in v], weights=[x[0] for x in v])
                      for v in per.values()], float)
        w = np.ones_like(d)
    pt = np.average(d, weights=w)
    genes = sorted({c[0] for c in cells})
    by = {gn: [(c[1], c[2]) for c in cells if c[0] == gn] for gn in genes}
    rng = np.random.default_rng(seed); boot = []
    for _ in range(B):
        ww, dd = [], []
        for gn in rng.choice(genes, len(genes), replace=True):
            if average == 'gene':
                v = by[gn]
                dd.append(np.average([x[1] for x in v], weights=[x[0] for x in v])); ww.append(1.0)
            else:
                for wi, di in by[gn]:
                    ww.append(wi); dd.append(di)
        boot.append(np.average(dd, weights=ww))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    n2 = sum(wi for _, wi, _ in cells if wi == 2)
    return dict(pt=pt, lo=lo, hi=hi, n_cells=len(cells), n_genes=len(genes),
                w2=100 * n2 / w.sum() if average == 'case' else np.nan)


def dlvl(g, outcome='endorse', B=1500, seed=11):
    lv = np.asarray(g.best_level); res = np.asarray(g.label) == 'RESISTANT'
    e = g[outcome].values.astype(float)
    genes, gidx = np.unique(np.asarray(g.gene), return_inverse=True)
    ok = [l for l in LEVS if ((lv == l) & res).sum() >= 5 and ((lv == l) & ~res).sum() >= 5]
    if not ok:
        return np.nan, np.nan, np.nan
    w = np.array([((lv == l) & res).sum() for l in ok], float)
    pt = np.average([e[(lv == l) & ~res].mean() - e[(lv == l) & res].mean() for l in ok],
                    weights=w)
    rng = np.random.default_rng(seed); bg = [np.where(gidx == i)[0] for i in range(len(genes))]
    out = []
    for _ in range(B):
        sel = np.concatenate([bg[i] for i in rng.integers(0, len(genes), len(genes))])
        L2, R2, E = lv[sel], res[sel], e[sel]
        vals, ws = [], []
        for l in ok:
            a = E[(L2 == l) & ~R2]; b = E[(L2 == l) & R2]
            if len(a) >= 3 and len(b) >= 3:
                vals.append(a.mean() - b.mean()); ws.append(len(b))
        if vals:
            out.append(np.average(vals, weights=ws))
    lo, hi = np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)
    return pt, lo, hi


rows = []
for m in ORDER:
    g = D[D.model == m].reset_index(drop=True)
    if not len(g):
        continue
    R = g[g.label == 'RESISTANT']
    slo, shi = cb_mean(R.strongrec.values, R.gene.values)
    e_lo, e_hi = cb_mean(R.endorse.values, R.gene.values)
    S = g[g.label == 'SENSITIVE']
    c_lo, c_hi = cb_mean(S.endorse.values, S.gene.values)
    ge = dgene(g, 'endorse'); gs = dgene(g, 'strongrec')
    dl = dlvl(g, 'endorse')
    rows.append(dict(model=m, vendor=VENDOR[m],
                     strong_R=100 * R.strongrec.mean(), s_lo=100 * slo, s_hi=100 * shi,
                     end_R=100 * R.endorse.mean(), e_lo=100 * e_lo, e_hi=100 * e_hi,
                     end_S=100 * S.endorse.mean(), c_lo=100 * c_lo, c_hi=100 * c_hi,
                     strong_S=100 * S.strongrec.mean(),
                     dg=100 * ge['pt'], dg_lo=100 * ge['lo'], dg_hi=100 * ge['hi'],
                     dg_strong=100 * gs['pt'], dgs_lo=100 * gs['lo'], dgs_hi=100 * gs['hi'],
                     dl=100 * dl[0], dl_lo=100 * dl[1], dl_hi=100 * dl[2]))
E = pd.DataFrame(rows)
E.to_csv('results/endpoints.csv', index=False)
print('\n=== 2/3/7. endpoints with intervals ===')
print(f"{'model':18s} {'P(strong|R)':>20s} {'D_gene(endorse)':>20s} {'D_gene(strong)':>20s} {'D_lvl':>20s}")
for _, r in E.iterrows():
    print(f"{LABEL[r.model]:18s} {r.strong_R:6.1f} [{r.s_lo:4.1f},{r.s_hi:5.1f}] "
          f"{r.dg:+7.1f} [{r.dg_lo:+4.0f},{r.dg_hi:+4.0f}] "
          f"{r.dg_strong:+7.1f} [{r.dgs_lo:+4.0f},{r.dgs_hi:+4.0f}] "
          f"{r.dl:+7.1f} [{r.dl_lo:+4.0f},{r.dl_hi:+4.0f}]")

# ------------------------------------------- 4. threshold sensitivity, 5. gene-averaged
rows = []
for m in ORDER:
    g = D[D.model == m].reset_index(drop=True)
    if not len(g):
        continue
    rec = dict(model=m)
    for thr in (2, 3, 4):
        r = dgene(g, thr=thr)
        rec[f't{thr}'] = 100 * r['pt']; rec[f't{thr}_cells'] = r['n_cells']
        rec[f't{thr}_genes'] = r['n_genes']
        if thr == 2:
            rec['w_in_n2'] = r['w2']
    ga = dgene(g, average='gene')
    rec['gene_avg'] = 100 * ga['pt']; rec['ga_lo'] = 100 * ga['lo']; rec['ga_hi'] = 100 * ga['hi']
    rows.append(rec)
T = pd.DataFrame(rows)
T.to_csv('results/threshold.csv', index=False)
print('\n=== 4/5. cell threshold sensitivity and gene-averaged estimand ===')
print(f"{'model':18s} {'thr=2':>8s} {'thr=3':>8s} {'thr=4':>8s}  {'cells 2/3/4':>12s} "
      f"{'wt in n=2':>10s} {'gene-avg':>10s}")
for _, r in T.iterrows():
    print(f"{LABEL[r.model]:18s} {r.t2:+8.1f} {r.t3:+8.1f} {r.t4:+8.1f}  "
          f"{int(r.t2_cells):3d}/{int(r.t3_cells):3d}/{int(r.t4_cells):3d}   "
          f"{r.w_in_n2:8.1f}% {r.gene_avg:+9.1f}")

# ------------------------------------ 6. Delta_gene on the pre-registered halves
rows = []
for m in ORDER:
    for half in ['EXPLORE', 'CONFIRM']:
        g = D[(D.model == m) & (D.split == half)].reset_index(drop=True)
        if not len(g):
            continue
        r = dgene(g)
        rows.append(dict(model=m, half=half, pt=100 * r['pt'], lo=100 * r['lo'],
                         hi=100 * r['hi'], n_cells=r['n_cells'], n_genes=r['n_genes']))
H = pd.DataFrame(rows)
H.to_csv('results/halves.csv', index=False)
print('\n=== 6. Delta_gene on the pre-registered halves ===')
for m in ORDER:
    h = H[H.model == m]
    if len(h) < 2:
        continue
    e = h[h.half == 'EXPLORE'].iloc[0]; c = h[h.half == 'CONFIRM'].iloc[0]
    print(f"  {LABEL[m]:18s} EXPLORE {e.pt:+6.1f} [{e.lo:+4.0f},{e.hi:+4.0f}] ({int(e.n_cells)} cells)"
          f"   CONFIRM {c.pt:+6.1f} [{c.lo:+4.0f},{c.hi:+4.0f}] ({int(c.n_cells)} cells)")

# ---------------------------------------------- 8. level-by-cohort case counts
rows = []
for name, path in SRC.items():
    if not os.path.exists(path):
        continue
    d = pd.read_csv(path)
    if 'label' not in d.columns:
        continue
    unmapped_r = ((d.label == 'RESISTANT') & ~d.best_level.isin(LEVS)).sum()
    unmapped_s = ((d.label == 'SENSITIVE') & ~d.best_level.isin(LEVS)).sum()
    if unmapped_r or unmapped_s:
        print(f'  NOTE {name}: {unmapped_r} resistance and {unmapped_s} sensitivity cases '
              f'carry a level outside A-D')
    for lv in LEVS:
        r = ((d.label == 'RESISTANT') & (d.best_level == lv)).sum()
        s = ((d.label == 'SENSITIVE') & (d.best_level == lv)).sum()
        rows.append(dict(cohort=name, level=lv, resistance=r, sensitivity=s))
L = pd.DataFrame(rows)
L.to_csv('results/levelcounts.csv', index=False)
print('\n=== 8. level-by-cohort counts (resistance / sensitivity) ===')
piv = L.pivot(index='cohort', columns='level', values='resistance').fillna(0).astype(int)
print(piv.to_string())
tot = L.groupby('cohort').resistance.sum()
print('\n  share of each cohort resistance arm at level D:')
for c in piv.index:
    if tot[c]:
        print(f"    {c:10s} {100*piv.loc[c,'D']/tot[c]:5.1f}%  (n={tot[c]})")
