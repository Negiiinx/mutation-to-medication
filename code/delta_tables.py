import pandas as pd, numpy as np, json, glob, os
REC={'Strongly recommend':4,'Consider':3,'Insufficient evidence':2,'Do not recommend':1}
def load(c,tag):
    co=pd.read_csv(c); rows=[]
    for f in glob.glob(f'raw_responses/{tag}_*.jsonl'):
        for L in open(f):
            r=json.loads(L)
            if 'error' in r['resp']: continue
            rows.append({'uid':r['uid'],'model':r['model'],**r['resp']})
    d=pd.DataFrame(rows).drop_duplicates(['uid','model']).merge(co,on='uid',how='inner')
    d['endorse']=(d.recommendation.map(REC)>=3).astype(int); return d
def cell_ci(g,l,B=2000,seed=3):
    """bootstrap the within-level difference, resampling genes"""
    gn=g.gene.values; res=(g.label.values=='RESISTANT'); e=g.endorse.values; lv=g.best_level.values
    genes,gidx=np.unique(gn,return_inverse=True); bg=[np.where(gidx==i)[0] for i in range(len(genes))]
    rng=np.random.default_rng(seed); out=[]
    for _ in range(B):
        sel=np.concatenate([bg[i] for i in rng.integers(0,len(genes),len(genes))])
        a=e[sel][(lv[sel]==l)&~res[sel]]; b=e[sel][(lv[sel]==l)&res[sel]]
        if len(a)>=3 and len(b)>=3: out.append(a.mean()-b.mean())
    return np.percentile(out,[2.5,97.5]) if len(out)>50 else (np.nan,np.nan)
API=['gpt-4.1','gpt-4o','gpt-4o-mini']
print("PER-MODEL, PER-LEVEL DELTA WITH INTERVALS\n")
for name,c,tag in [('CIViC','data/cohort_direction.csv','dir'),('JAX-CKB','data/cohort_jax_clean.csv','jax2'),
                   ('CGI','data/cohort_cgi_clean.csv','cgi2')]:
    d=load(c,tag); print(f"### {name}")
    for l in ['A','B','C','D']:
        line=[]; nr=None
        for m in API:
            g=d[d.model==m].reset_index(drop=True)
            s=g[(g.best_level==l)&(g.label=='SENSITIVE')].endorse; r=g[(g.best_level==l)&(g.label=='RESISTANT')].endorse
            if len(s)<5 or len(r)<5: line.append("--"); continue
            nr=len(r); lo,hi=cell_ci(g,l)
            line.append(f"{s.mean()-r.mean():+.0%} [{lo:+.0%},{hi:+.0%}]")
        print(f"  {l} (n_res={nr}): "+" | ".join(f"{x:>18s}" for x in line))
    print()
print("\nFOUR-WAY RESPONSE DISTRIBUTION, CIViC resistance cases")
d=load('data/cohort_direction.csv','dir'); R=d[d.label=='RESISTANT']
t=pd.crosstab(R.model,R.recommendation,normalize='index')[['Strongly recommend','Consider','Insufficient evidence','Do not recommend']]
print((t*100).round(1).to_string())
print("\nsame, sensitivity controls")
S=d[d.label=='SENSITIVE']
t2=pd.crosstab(S.model,S.recommendation,normalize='index').reindex(columns=['Strongly recommend','Consider','Insufficient evidence','Do not recommend']).fillna(0)
print((t2*100).round(1).to_string())
