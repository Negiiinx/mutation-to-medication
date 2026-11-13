"""Build the two artifacts the paper promises for independent clinical adjudication.

The submitted paper commits to releasing, on acceptance:
  * "a random sample of retained resistance cases" for an adjudication study, and
  * "the flagged cases", meaning the entries the exclusion rules removed.

Neither existed as a file; both are generated here, deterministically, so a third party
can reproduce the exact sample. Writes into release/adjudication/.
"""
import warnings; warnings.filterwarnings('ignore')
import os, re
import pandas as pd

OUT = 'release/adjudication'
os.makedirs(OUT, exist_ok=True)
SEED = 20260910          # fixed so the sample is reproducible from the paper alone
N_SAMPLE = 50            # the size the paper names in Limitations

# ---------------------------------------------------------------- retained resistance cases
d = pd.read_csv('data/cohort_direction.csv')
res = d[d.label == 'RESISTANT'].copy()
samp = res.sample(n=min(N_SAMPLE, len(res)), random_state=SEED).sort_values(['gene', 'therapy'])
samp = samp[['uid', 'gene', 'molecular_profile', 'disease', 'therapy', 'best_level', 'n_items']]
samp.insert(len(samp.columns), 'adjudicator_verdict', '')      # supports / refutes / unclear
samp.insert(len(samp.columns), 'adjudicator_notes', '')
samp.to_csv(f'{OUT}/adjudication_sample_50.csv', index=False)
print(f'adjudication_sample_50.csv: {len(samp)} cases, {samp.gene.nunique()} genes, seed {SEED}')
print('  level mix:', samp.best_level.value_counts().sort_index().to_dict())

# ------------------------------------------------------------------- excluded ("flagged") entries
GENERIC = re.compile(r'^\s*[A-Z0-9\-]+\s*$')
COND = re.compile(r'\b(secondary|acquired|compound|reversion)\b', re.I)
rows = []
for name, pre, post in [('CGI', 'cohort_cgi.csv', 'cohort_cgi_clean.csv'),
                        ('JAX-CKB', 'cohort_jax.csv', 'cohort_jax_clean.csv'),
                        ('MOAlmanac', 'cohort_moa.csv', 'cohort_moa_clean.csv')]:
    pa, pb = f'data/{pre}', f'data/{post}'
    if not (os.path.exists(pa) and os.path.exists(pb)):
        continue
    a = pd.read_csv(pa); b = pd.read_csv(pb)
    a = a[a.label == 'RESISTANT']; b = b[b.label == 'RESISTANT']
    # uids are regenerated between the pre- and post-exclusion files, so match on content
    def key(d):
        return list(zip(d.molecular_profile.astype(str), d.therapy.astype(str),
                        d.disease.astype(str)))
    kept = set(key(b))
    dropped = a[[k not in kept for k in key(a)]].copy()
    dropped['source'] = name
    mp = dropped.molecular_profile.astype(str)
    dropped['exclusion_reason'] = [
        'gene-level, no specific alteration' if GENERIC.match(m)
        else ('conditional qualifier' if COND.search(m) else 'conditional qualifier in description')
        for m in mp]
    keep = [c for c in ['source', 'uid', 'gene', 'molecular_profile', 'disease', 'therapy',
                        'best_level', 'exclusion_reason'] if c in dropped.columns]
    rows.append(dropped[keep])
    print(f'{name}: {len(a)} -> {len(b)} retained, {len(dropped)} excluded')

if rows:
    exc = pd.concat(rows, ignore_index=True)
    exc.to_csv(f'{OUT}/excluded_resistance_entries.csv', index=False)
    print(f'excluded_resistance_entries.csv: {len(exc)} rows')
    print('  by reason:', exc.exclusion_reason.value_counts().to_dict())
