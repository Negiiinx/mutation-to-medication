"""Controls for the self-conditioning arm.

The self arm shows endorsement collapsing when a model's own resistance conclusion is quoted
back to it. Two alternative explanations remain, and each gets an arm on the same cases:

  falseself  On the cases where the model itself said "yes, confers resistance", it is instead
             told it had concluded "No, does not confer resistance". If the collapse is deference
             to whatever conclusion is quoted, endorsement should stay high or rise. If the model
             is reading the evidence, it should reject the false attribution.

  noinstr    The self prompt without its closing instruction ("If you depart from your earlier
             conclusion, your judgement should reflect the evidence..."). If the instruction is
             doing the work rather than the quoted conclusion, removing it should undo the effect.

Writes results/controls.csv.
"""
import warnings; warnings.filterwarnings('ignore')
import glob, json
import numpy as np, pandas as pd
from anlib import load, ORDER, LABEL, REC

YES = 'Yes, confers resistance'
NO = 'No, does not confer resistance'

base = load('data/cohort_direction.csv', 'dir')
base = base[(base.label == 'RESISTANT') & base.model.isin(ORDER)]
gene = dict(zip(base.uid, base.gene))
B = {(r.model, r.uid): (r.endorse, r.strongrec) for _, r in base.iterrows()}


def read(tag):
    rows = []
    for f in glob.glob(f'raw_responses/{tag}_*.jsonl'):
        for L in open(f):
            try:
                r = json.loads(L)
            except Exception:
                continue
            resp = r.get('resp', {})
            if 'error' in resp or 'recommendation' not in resp:
                continue
            rows.append(dict(uid=r['uid'], model=r['model'], prior=r.get('prior'),
                             shown=r.get('shown'),
                             endorse=int(REC[resp['recommendation']] >= 3),
                             strong=int(resp['recommendation'] == 'Strongly recommend')))
    if not rows:
        return pd.DataFrame(columns=['uid', 'model', 'prior', 'shown', 'endorse', 'strong'])
    return pd.DataFrame(rows).drop_duplicates(['uid', 'model'])


SELF, FALSE, NOI = read('self'), read('falseself'), read('noinstr')


def cb(pairs, genes, B_=2000, seed=31):
    genes = np.asarray(genes); d = np.asarray(pairs, float)
    gs = np.unique(genes)
    if len(gs) < 2:
        return np.nan, np.nan
    by = {g: d[genes == g] for g in gs}
    rng = np.random.default_rng(seed)
    out = [np.concatenate([by[g] for g in rng.choice(gs, len(gs), replace=True)]).mean()
           for _ in range(B_)]
    return tuple(np.percentile(out, [2.5, 97.5]))


rows = []
for m in ORDER:
    sf = SELF[SELF.model == m]
    if not len(sf):
        continue
    # the yes-bucket is the comparison set: cases the model itself called resistance
    yes = sf[sf.prior == YES]
    yu = set(yes.uid)
    fa = FALSE[(FALSE.model == m) & FALSE.uid.isin(yu)]
    ni = NOI[(NOI.model == m) & NOI.uid.isin(yu)]

    def rate(df, col='endorse'):
        return 100 * df[col].mean() if len(df) else np.nan

    common = yu & set(fa.uid) if len(fa) else set()
    lo = hi = np.nan
    if common:
        a = fa[fa.uid.isin(common)].set_index('uid')
        b = yes[yes.uid.isin(common)].set_index('uid')
        idx = sorted(common)
        lo, hi = cb((a.loc[idx].endorse - b.loc[idx].endorse).values,
                    [gene[u] for u in idx])
    rows.append(dict(
        model=m,
        n_yes=len(yes),
        base_yes=100 * np.mean([B[(m, u)][0] for u in yes.uid]),
        # strong-rec baseline on the SAME subset, so the false-attribution comparison is like-for-like
        base_strong=100 * np.mean([B[(m, u)][1] for u in yes.uid]),
        self_yes=rate(yes),
        n_false=len(fa), false_yes=rate(fa), false_strong=rate(fa, 'strong'),
        d_lo=100 * lo if lo == lo else np.nan,
        d_hi=100 * hi if hi == hi else np.nan,
        n_noinstr=len(ni), noinstr_yes=rate(ni)))
C = pd.DataFrame(rows)
C.to_csv('results/controls.csv', index=False)

print('Endorsement (%) on the cases where the model itself said "confers resistance":')
print(f"{'model':18s} {'n':>4s} {'base':>6s} {'self':>6s} {'FALSE':>7s} {'no-instr':>9s}"
      f"   95% CI on false-self")
for _, r in C.iterrows():
    ci = (f"[{r.d_lo:+.1f}, {r.d_hi:+.1f}]" if r.d_lo == r.d_lo else '--')
    print(f"{LABEL[r.model]:18s} {int(r.n_yes):4d} {r.base_yes:6.1f} {r.self_yes:6.1f} "
          f"{r.false_yes:7.1f} {r.noinstr_yes:9.1f}   {ci}")

if C.false_yes.notna().any():
    print(f"\nfalse-self mean endorsement {C.false_yes.mean():.1f}% "
          f"vs baseline {C.base_yes.mean():.1f}% vs true-self {C.self_yes.mean():.1f}%")
    print(f"strong-rec under false attribution: {C.false_strong.mean():.1f}%")

print('\nstrong recommendation on the same subset, baseline -> false attribution:')
for _, r in C.iterrows():
    print(f"  {LABEL[r.model]:18s} {r.base_strong:5.1f} -> {r.false_strong:5.1f}  "
          f"({r.false_strong - r.base_strong:+.1f} pp)")
print(f"\nlargest rise: {(C.false_strong - C.base_strong).max():+.1f} pp "
      f"({LABEL[C.loc[(C.false_strong - C.base_strong).idxmax(), 'model']]})")
print(f"smallest:     {(C.false_strong - C.base_strong).min():+.1f} pp "
      f"({LABEL[C.loc[(C.false_strong - C.base_strong).idxmin(), 'model']]})")
