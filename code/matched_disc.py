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
LEVS=['A','B','C','D']
def matched(g,B=1500,seed=11):
    gn=g.gene.values; lv=g.best_level.values; res=(g.label.values=='RESISTANT'); e=g.endorse.values
    genes,gidx=np.unique(gn,return_inverse=True)
    ok=[l for l in LEVS if ((lv==l)&res).sum()>=5 and ((lv==l)&~res).sum()>=5]
    if not ok: return np.nan,np.nan,np.nan,{}
    w=np.array([((lv==l)&res).sum() for l in ok],float)
    per={l: e[(lv==l)&~res].mean()-e[(lv==l)&res].mean() for l in ok}
    pt=np.average([per[l] for l in ok],weights=w)
    rng=np.random.default_rng(seed); out=[]
    bygene=[np.where(gidx==i)[0] for i in range(len(genes))]
    for _ in range(B):
        sel=np.concatenate([bygene[i] for i in rng.integers(0,len(genes),len(genes))])
        L=lv[sel]; R=res[sel]; E=e[sel]; vals=[]; ws=[]
        for l in ok:
            a=E[(L==l)&~R]; b=E[(L==l)&R]
            if len(a)>=3 and len(b)>=3: vals.append(a.mean()-b.mean()); ws.append(len(b))
        if vals: out.append(np.average(vals,weights=ws))
    lo,hi=np.percentile(out,[2.5,97.5]) if out else (np.nan,np.nan)
    return pt,lo,hi,per
rows=[]
for name,c,tag in [('CIViC','data/cohort_direction.csv','dir'),('JAX-CKB','data/cohort_jax_clean.csv','jax2'),
                   ('CGI','data/cohort_cgi_clean.csv','cgi2'),('MOAlmanac','data/cohort_moa_clean.csv','moa2'),
                   ('OncoKB','data/cohort_oncokb.csv','oncokb')]:
    if not os.path.exists(c): continue
    d=load(c,tag); print(f"  {name}",flush=True)
    for m,g in d.groupby('model'):
        g=g.reset_index(drop=True)
        raw=g[g.label=='SENSITIVE'].endorse.mean()-g[g.label=='RESISTANT'].endorse.mean()
        pt,lo,hi,per=matched(g)
        if np.isnan(pt): print(f"    {m:14s} raw {raw:+6.1%}   matched: too few per level",flush=True); continue
        pl=' '.join(f"{l}:{v:+.0%}" for l,v in per.items())
        print(f"    {m:14s} raw {raw:+6.1%}   MATCHED {pt:+6.1%} [{lo:+.0%},{hi:+.0%}]  {pl}",flush=True)
        rows.append(dict(kb=name,model=m,raw=raw,matched=pt,lo=lo,hi=hi))
    print(flush=True)
pd.DataFrame(rows).to_csv('data/discrimination_matched.csv',index=False)
