import pandas as pd, numpy as np, re
e = pd.read_csv('data/nightly-ClinicalEvidenceSummaries.tsv', sep='\t', dtype=str)
ep = e[(e.evidence_type=='Predictive') & (e.evidence_direction=='Supports')
       & (e.significance=='Sensitivity/Response')].copy()
ep['therapies'] = ep.therapies.fillna('')
ep = ep[(ep.therapies.str.len()>0) & (ep.therapies.str.count(',')==0)]
ep['therapy'] = ep.therapies.str.strip()

def gene_from_mp(mp):
    s = str(mp)
    if '::' in s:                       # fusion: 3' partner is the kinase/target
        parts = re.split(r'::', s.split(' ')[0])
        gs = [p for p in parts if p and p.lower() != 'v']
        if gs: return gs[-1]
    tok = re.sub(r'[^A-Za-z0-9\-]','', s.split(' ')[0])
    return tok if len(tok)>1 else None

ep['gene'] = ep.molecular_profile.map(gene_from_mp)
ep = ep[ep.gene.notna()]
ep['pmid'] = ep.apply(lambda r: r.citation_id if r.source_type=='PubMed' else None, axis=1)
LR = {'A':5,'B':4,'C':3,'D':2,'E':1}
ep['lvl'] = ep.evidence_level.map(LR)

trip = ep.groupby(['gene','molecular_profile','disease','therapy']).agg(
    best_lvl=('lvl','max'), n_items=('evidence_id','count'),
    n_sources=('citation_id','nunique'),
    mean_rating=('rating', lambda s: pd.to_numeric(s,errors='coerce').mean()),
    pmids=('pmid', lambda s: ','.join(sorted({x for x in s if pd.notna(x)}))),
).reset_index()
trip['best_level'] = trip.best_lvl.map({v:k for k,v in LR.items()})
trip.to_csv('data/cohort.csv', index=False)
print("triples:", len(trip), "| genes:", trip.gene.nunique(), "| therapies:", trip.therapy.nunique(), "| diseases:", trip.disease.nunique())
print(trip.best_level.value_counts().sort_index().to_dict())
allp = sorted({p for s in trip.pmids for p in str(s).split(',') if p and p!='nan'})
open('data/pmids.txt','w').write('\n'.join(allp))
print("unique PMIDs for era lookup:", len(allp))
print("\nFusion spot check:", [gene_from_mp(x) for x in ['EML4::ALK Fusion','ETV6::NTRK3 Fusion','BCR::ABL1 e13e2','v::RET Fusion']])
