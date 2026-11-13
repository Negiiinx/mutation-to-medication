"""Self-conditioning arm: endorsement when the model is shown its own prior conclusion.

Three conditions on the same 312 CIViC resistance cases:
  base  no prior conclusion            (dir arm)
  self  the model's own probe answer   (self arm)
  rag   the curated evidence statement (rag arm, oracle)

The decisive contrast is within the self arm: cases where the model itself said "yes, confers
resistance" against cases where it said "unknown". If quoting any text back merely makes models
cautious, both buckets fall together. If the failure is one of *using* knowledge the model already
had, only the "yes" bucket falls.

Writes results/selfcond.csv and results/selfcond_split.csv.
"""
import warnings; warnings.filterwarnings('ignore')
import json, glob
import numpy as np, pandas as pd
from anlib import load, ORDER, LABEL, VENDOR, REC

YES = 'Yes, confers resistance'
UNK = 'Unknown or insufficient evidence'

base = load('data/cohort_direction.csv', 'dir')
base = base[(base.label == 'RESISTANT') & base.model.isin(ORDER)]
gene = dict(zip(base.uid, base.gene))
B = {(r.model, r.uid): (r.endorse, r.strongrec) for _, r in base.iterrows()}

rag = load('data/cohort_direction_rag.csv', 'rag')
R = {}
if rag is not None:
    rag = rag[rag.label == 'RESISTANT']
    R = {(r.model, r.uid): r.endorse for _, r in rag.iterrows()}

rows = []
for f in glob.glob('raw_responses/self_*.jsonl'):
    for L in open(f):
        try:
            r = json.loads(L)
        except Exception:
            continue
        resp = r.get('resp', {})
        if 'error' in resp or 'recommendation' not in resp:
            continue
        rows.append(dict(uid=r['uid'], model=r['model'], prior=r['prior'],
                         endorse=int(REC[resp['recommendation']] >= 3),
                         strong=int(resp['recommendation'] == 'Strongly recommend')))
S = pd.DataFrame(rows).drop_duplicates(['uid', 'model'])
S['gene'] = S.uid.map(gene)


def cb_diff(pairs, genes, B_=2000, seed=23):
    """Cluster bootstrap over genes for a mean paired difference."""
    genes = np.asarray(genes); d = np.asarray(pairs, float)
    gs = np.unique(genes)
    if len(gs) < 2:
        return np.nan, np.nan
    by = {g: d[genes == g] for g in gs}
    rng = np.random.default_rng(seed)
    out = [np.concatenate([by[g] for g in rng.choice(gs, len(gs), replace=True)]).mean()
           for _ in range(B_)]
    return tuple(np.percentile(out, [2.5, 97.5]))


main, split = [], []
for m in ORDER:
    g = S[S.model == m]
    if not len(g):
        continue
    keep = [(u, e, st, pr, gn) for u, e, st, pr, gn in
            zip(g.uid, g.endorse, g.strong, g.prior, g.gene) if (m, u) in B]
    if not keep:
        continue
    n = len(keep)
    be = np.array([B[(m, u)][0] for u, *_ in keep], float)
    bs = np.array([B[(m, u)][1] for u, *_ in keep], float)
    se = np.array([e for _, e, *_ in keep], float)
    ss = np.array([st for _, _, st, *_ in keep], float)
    gn = [x[4] for x in keep]
    lo, hi = cb_diff(se - be, gn)
    rg = [R[(m, u)] for u, *_ in keep if (m, u) in R]
    main.append(dict(model=m, vendor=VENDOR[m], n=n,
                     base_end=100 * be.mean(), self_end=100 * se.mean(),
                     rag_end=100 * np.mean(rg) if rg else np.nan,
                     base_str=100 * bs.mean(), self_str=100 * ss.mean(),
                     d_lo=100 * lo, d_hi=100 * hi))
    for lab, key in [('yes', YES), ('unknown', UNK)]:
        h = [(u, e, gg) for u, e, st, pr, gg in keep if pr == key]
        if len(h) < 10:
            continue
        b2 = np.array([B[(m, u)][0] for u, _, _ in h], float)
        s2 = np.array([e for _, e, _ in h], float)
        l2, h2 = cb_diff(s2 - b2, [x[2] for x in h])
        split.append(dict(model=m, bucket=lab, n=len(h),
                          base=100 * b2.mean(), self=100 * s2.mean(),
                          lo=100 * l2, hi=100 * h2))

M = pd.DataFrame(main); P = pd.DataFrame(split)

# Reconciliation guard. The base and bucket rates in this table must equal the ones the
# knowledge-probe table reports, since both are the same models on the same cases. Anything
# else means the two arms disagree about the case set or about the model's own answer.
kn = load('data/cohort_direction.csv', 'dir')
kn = kn[(kn.label == 'RESISTANT') & kn.model.isin(ORDER)]
probe = {}
for m_ in ORDER:
    path = f"raw_responses/know_{m_.replace(':', '-')}.jsonl"
    import os as _os
    if not _os.path.exists(path):
        continue
    d = {}
    for L in open(path):
        try:
            r = json.loads(L)
        except Exception:
            continue
        rs = r.get('resp', {})
        if 'error' in rs or 'resistance_known' not in rs:
            continue
        d.setdefault(r['uid'], rs['resistance_known'])
    probe[m_] = d
bad = []
for _, r in P.iterrows():
    key = YES if r.bucket == 'yes' else UNK
    g = kn[kn.model == r.model]
    ref = g[g.uid.map(lambda u: probe.get(r.model, {}).get(u)) == key]
    if len(ref) != r.n or abs(100 * ref.endorse.mean() - r.base) > 0.05:
        bad.append((r.model, r.bucket, int(r.n), len(ref),
                    round(r.base, 1), round(100 * ref.endorse.mean(), 1)))
if bad:
    print('\nRECONCILIATION FAILURES (model, bucket, n_self, n_probe, base_self, base_probe):')
    for b in bad:
        print('  ', b)
else:
    print('\nreconciliation: self-arm buckets match the knowledge-probe table exactly')
M.to_csv('results/selfcond.csv', index=False)
P.to_csv('results/selfcond_split.csv', index=False)

print('=== endorsement of resistance cases (%) ===')
print(f"{'model':18s} {'n':>4s} {'base':>6s} {'self':>6s} {'oracle':>7s}   95% CI on self-base")
for _, r in M.iterrows():
    rag_s = f"{r.rag_end:6.1f}" if not np.isnan(r.rag_end) else "    --"
    print(f"{LABEL[r.model]:18s} {r.n:4d} {r.base_end:6.1f} {r.self_end:6.1f} {rag_s}   "
          f"[{r.d_lo:+.1f}, {r.d_hi:+.1f}]")

print('\n=== within the self arm, by what the model itself said ===')
for m in ORDER:
    h = P[P.model == m]
    if not len(h):
        continue
    parts = []
    for _, r in h.iterrows():
        parts.append(f"{r.bucket:7s} n={int(r.n):3d} {r.base:5.1f}->{r['self']:5.1f} "
                     f"[{r.lo:+.0f},{r.hi:+.0f}]")
    print(f"  {LABEL[m]:18s} " + "   ".join(parts))
