"""The pre-registered confirmatory test, run on the CONFIRM half only.

PREREGISTRATION.md, as amended before any CONFIRM data was analysed:
  primary endpoint  rate of "Strongly recommend" on a variant curated RESISTANT
  design            within gene+drug pair, so gene, drug and prominence are held fixed
  test              linear probability model, outcome ~ label, with pair fixed effects and
                    cluster-robust (CR1) standard errors by gene
  secondary         broad endorsement (Consider or Strongly recommend)

Writes results/prereg.csv.
"""
import os
# The fixed effects are solved with np.linalg.pinv, and a threaded LAPACK splits that reduction
# differently depending on how many threads it is given, which moves the last bits of every
# coefficient. Pinning the thread count before numpy is imported makes the fit reproduce exactly
# on any machine; the matrices here are small enough that one thread costs nothing.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'

import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from anlib import load, ORDER, LABEL

D = load('data/cohort_direction.csv', 'dir')
D = D[D.model.isin(ORDER)].copy()
D['pair'] = D.gene.astype(str) + '|' + D.therapy.astype(str)


def lpm_fe(g, outcome):
    """Outcome ~ label with gene-drug pair fixed effects, CR1 by gene."""
    keep = [p for p, s in g.groupby('pair') if s.label.nunique() == 2]
    g = g[g.pair.isin(keep)]
    if g.gene.nunique() < 3 or len(g) < 20:
        return None
    y = g[outcome].values.astype(float)
    res = (g.label == 'RESISTANT').values.astype(float)
    pairs = sorted(g.pair.unique())
    X = np.column_stack([res] + [(g.pair == p).values.astype(float) for p in pairs])
    XtX = X.T @ X
    beta = np.linalg.pinv(XtX) @ (X.T @ y)
    u = y - X @ beta
    Ainv = np.linalg.pinv(XtX)
    meat = np.zeros_like(XtX)
    genes = g.gene.values
    for gn in np.unique(genes):
        m = genes == gn
        s = X[m].T @ u[m]
        meat += np.outer(s, s)
    G, k = len(np.unique(genes)), X.shape[1]
    c = G / max(G - 1, 1) * (len(y) - 1) / max(len(y) - k, 1)
    V = Ainv @ (c * meat) @ Ainv
    se = np.sqrt(max(V[0, 0], 0))
    t = beta[0] / se if se > 0 else np.nan
    from scipy import stats
    pv = 2 * (1 - stats.t.cdf(abs(t), df=max(G - 1, 1)))
    return dict(beta=beta[0], se=se, t=t, p=pv, n=len(y),
                genes=G, pairs=len(pairs))


rows = []
for half in ['CONFIRM', 'EXPLORE']:
    for m in ORDER:
        g = D[(D.model == m) & (D.split == half)]
        if not len(g):
            continue
        for name, col in [('strong_rec', 'strongrec'), ('endorse', 'endorse')]:
            r = lpm_fe(g, col)
            if r:
                rows.append(dict(half=half, model=m, endpoint=name, **r))
P = pd.DataFrame(rows)
P.to_csv('results/prereg.csv', index=False)

for half in ['CONFIRM', 'EXPLORE']:
    for ep in ['strong_rec', 'endorse']:
        h = P[(P.half == half) & (P.endpoint == ep)]
        if not len(h):
            continue
        sig = (h.p < 0.05).sum()
        neg = ((h.beta < 0) & (h.p < 0.05)).sum()
        print(f'\n{half} / {ep}:  {h.pairs.iloc[0]} pairs, {h.genes.iloc[0]} genes, '
              f'n={h.n.iloc[0]} per model')
        for _, r in h.iterrows():
            star = '*' if r.p < 0.05 else ' '
            print(f'   {LABEL[r.model]:18s} beta={r.beta:+.3f} se={r.se:.3f} p={r.p:.3f} {star}')
        print(f'   -> {sig}/{len(h)} significant, {neg} of them in the predicted direction')
