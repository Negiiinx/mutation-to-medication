"""Expanded cohort: relax variant requirement, add combination therapies.
Goal is maximum gene clusters for inference."""
import pandas as pd, numpy as np, re, json, hashlib
e=pd.read_csv('data/nightly-ClinicalEvidenceSummaries.tsv',sep='\t',dtype=str)
def gene_from_mp(mp):
    s=str(mp)
    if '::' in s:
        parts=[p for p in re.split(r'::',s.split(' ')[0]) if p and p.lower()!='v']
        if parts: return parts[-1]
    tok=re.sub(r'[^A-Za-z0-9\-]','',s.split(' ')[0]); return tok if len(tok)>1 else None

ep=e[(e.evidence_type=='Predictive')&(e.evidence_direction=='Supports')
     &(e.significance=='Sensitivity/Response')].copy()
ep['therapies']=ep.therapies.fillna('')
ep=ep[ep.therapies.str.len()>0]                      # <-- combos NOW INCLUDED
ep['therapy']=ep.therapies.str.strip()
ep['n_agents']=ep.therapy.str.count(',')+1
ep['gene']=ep.molecular_profile.map(gene_from_mp); ep=ep[ep.gene.notna()]
ep['pmid']=ep.apply(lambda r: r.citation_id if r.source_type=='PubMed' else None,axis=1)
LR={'A':5,'B':4,'C':3,'D':2,'E':1}; ep['lvl']=ep.evidence_level.map(LR)

trip=ep.groupby(['gene','molecular_profile','disease','therapy']).agg(
    best_lvl=('lvl','max'), n_items=('evidence_id','count'), n_sources=('citation_id','nunique'),
    n_agents=('n_agents','first'),
    mean_rating=('rating',lambda s: pd.to_numeric(s,errors='coerce').mean()),
    pmids=('pmid',lambda s: ','.join(sorted({x for x in s if pd.notna(x)}))),
).reset_index()
trip['best_level']=trip.best_lvl.map({v:k for k,v in LR.items()})
trip=trip[trip.best_level!='E']
trip['strong']=trip.best_level.isin(['A','B']); trip['weak']=trip.best_level.isin(['C','D'])
trip['uid']=[hashlib.md5(f"{r.molecular_profile}|{r.disease}|{r.therapy}".encode()).hexdigest()[:12] for r in trip.itertuples()]

cache=json.load(open('data/pm_cache.json'))
trip['n_gene']=trip.gene.map(lambda g: cache.get(f'"{g}"[tiab]'))
print(f"all triples (incl. combos): {len(trip)} | genes {trip.gene.nunique()} | combos {(trip.n_agents>1).sum()}")
print("missing gene PubMed counts:", trip.n_gene.isna().sum())

g=trip.groupby('gene').agg(s=('strong','sum'),w=('weak','sum'))
keep=g[(g.s>0)&(g.w>0)].index
sub=trip[trip.gene.isin(keep)&trip.n_gene.notna()].copy()
print(f"\nWITHIN-GENE SET: {len(sub)} triples, {sub.gene.nunique()} gene clusters")
print("levels:",sub.best_level.value_counts().sort_index().to_dict())
print("cases per gene: median %.0f, min %d, max %d"%(sub.groupby('gene').size().median(),
      sub.groupby('gene').size().min(),sub.groupby('gene').size().max()))
sub.to_csv('data/cohort_v2.csv',index=False)

old=set(pd.read_csv('data/cohort_final.csv').uid)
print(f"\nreuse: {len(set(sub.uid)&old)} already probed, {len(set(sub.uid)-old)} NEW cases needed per model")
