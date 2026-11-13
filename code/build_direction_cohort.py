"""Sensitivity vs resistance within the SAME gene+drug pair.
Pre-registered: gene-level 50/50 split into EXPLORATORY and CONFIRMATORY is fixed here,
before any model is run, and written to disk."""
import pandas as pd, numpy as np, re, hashlib, json
e=pd.read_csv('data/nightly-ClinicalEvidenceSummaries.tsv',sep='\t',dtype=str)
def gfm(mp):
    s=str(mp)
    if '::' in s:
        p=[x for x in re.split(r'::',s.split(' ')[0]) if x and x.lower()!='v']
        if p: return p[-1]
    t=re.sub(r'[^A-Za-z0-9\-]','',s.split(' ')[0]); return t if len(t)>1 else None
ep=e[(e.evidence_type=='Predictive')&(e.evidence_direction=='Supports')].copy()
ep['gene']=ep.molecular_profile.map(gfm); ep['therapies']=ep.therapies.fillna('')
ep=ep[(ep.gene.notna())&(ep.therapies.str.len()>0)]
ep['label']=np.where(ep.significance=='Sensitivity/Response','SENSITIVE',
             np.where(ep.significance.isin(['Resistance','Reduced Sensitivity']),'RESISTANT',None))
ep=ep[ep.label.notna()]
LR={'A':5,'B':4,'C':3,'D':2,'E':1}; ep['lvl']=ep.evidence_level.map(LR)

k=['gene','therapies']
pairs=ep.groupby(k).label.nunique()
both=set(pairs[pairs==2].index)
sub=ep[[ (g,t) in both for g,t in zip(ep.gene,ep.therapies)]].copy()

trip=sub.groupby(['gene','therapies','molecular_profile','disease','label']).agg(
    best_lvl=('lvl','max'), n_items=('evidence_id','count')).reset_index()
trip=trip[trip.best_lvl>=2]                       # drop level E
trip['best_level']=trip.best_lvl.map({v:k2 for k2,v in LR.items()})
trip['uid']=[hashlib.md5(f"DIR|{r.molecular_profile}|{r.disease}|{r.therapies}|{r.label}".encode()).hexdigest()[:12]
             for r in trip.itertuples()]
# keep only pairs that still have both labels after filtering
ok=trip.groupby(k).label.nunique(); keep=set(ok[ok==2].index)
trip=trip[[ (g,t) in keep for g,t in zip(trip.gene,trip.therapies)]]

cache=json.load(open('data/pm_cache.json'))
trip['n_gene']=trip.gene.map(lambda g: cache.get(f'"{g}"[tiab]'))

print(f"COHORT: {len(trip)} triples | {trip.gene.nunique()} genes | "
      f"{trip.groupby(k).ngroups} gene-drug pairs")
print(trip.label.value_counts().to_dict())
print("\nby label x evidence level:")
print(pd.crosstab(trip.label,trip.best_level).to_string())
print(f"\ncases per gene-drug pair: median {trip.groupby(k).size().median():.0f}, max {trip.groupby(k).size().max()}")

# ---- PRE-REGISTERED SPLIT, fixed seed, by GENE so no leakage across the split ----
genes=sorted(trip.gene.unique())
rng=np.random.default_rng(20260831)
perm=rng.permutation(len(genes))
explore={genes[i] for i in perm[:len(genes)//2]}
trip['split']=np.where(trip.gene.isin(explore),'EXPLORE','CONFIRM')
print("\nPRE-REGISTERED SPLIT (by gene, seed 20260831, fixed before any model call):")
print(trip.groupby(['split','label']).size().unstack().to_string())
print("genes:", trip.groupby('split').gene.nunique().to_dict())
trip.to_csv('data/cohort_direction.csv',index=False)

open('PREREGISTRATION.md','w').write(f"""# Pre-registration — direction discrimination
Written {pd.Timestamp('2026-08-31').date()}, BEFORE any model was run on this cohort.

## Question
For gene-drug pairs where CIViC contains BOTH sensitivity and resistance variants, do LLMs
endorse the drug less for the resistance variants than for the sensitivity variants?

## Design
Within gene+drug pair. Gene identity, drug identity and gene prominence are held constant by
construction; only the variant and its curated direction change.

## Primary endpoint
Endorsement rate (recommendation of "Consider" or "Strongly recommend") for RESISTANT variants
minus the same for SENSITIVE variants, within gene-drug pair. Under correct behaviour this gap
should be strongly negative.

## Primary test
Linear probability model, endorse ~ label, with gene-drug pair fixed effects, cluster-robust SEs
by gene. Confirmatory analysis is run ONLY on the CONFIRM split.

## Splits (fixed here, seed 20260831, split by gene so no gene appears in both)
EXPLORE genes: {sorted(explore)}
CONFIRM genes: {sorted(set(genes)-explore)}

## Declared exploratory (not confirmatory) analyses
- interaction with CIViC evidence level of the resistance finding
- interaction with gene prominence (NOTE: a prior prominence hypothesis on a different cohort
  failed held-out replication; treated as exploratory only)
""")
print("\nwrote PREREGISTRATION.md")
