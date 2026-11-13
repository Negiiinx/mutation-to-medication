"""Cross-knowledge-base prevalence audit. One table, five sources."""
import pandas as pd, numpy as np, json, glob, os
REC={'Strongly recommend':4,'Consider':3,'Insufficient evidence':2,'Do not recommend':1}
SRC=[('CIViC','data/cohort_direction.csv','dir'),
     ('CGI-clean','data/cohort_cgi_clean.csv','cgi2'),
     ('JAX-clean','data/cohort_jax_clean.csv','jax2'),
     ('MOA-clean','data/cohort_moa_clean.csv','moa2'),
     ('OncoKB','data/cohort_oncokb.csv','oncokb')]
def load(cohort,tag):
    if not os.path.exists(cohort): return None
    c=pd.read_csv(cohort); rows=[]
    for f in glob.glob(f'raw_responses/{tag}_*.jsonl'):
        for L in open(f):
            r=json.loads(L)
            if 'error' in r['resp']: continue
            rows.append({'uid':r['uid'],'model':r['model'],**r['resp']})
    if not rows: return None
    d=pd.DataFrame(rows).drop_duplicates(['uid','model']).merge(c,on='uid',how='inner')
    d['rec']=d.recommendation.map(REC); d['endorse']=(d.rec>=3).astype(int)
    d['strongrec']=(d.recommendation=='Strongly recommend').astype(int)
    return d
def cb(sub,col,B=4000,seed=5):
    rng=np.random.default_rng(seed); gs=sub.gene.unique()
    byg={g:sub[sub.gene==g][col].values for g in gs}
    out=[np.concatenate([byg[g] for g in rng.choice(gs,len(gs),replace=True)]).mean() for _ in range(B)]
    return np.percentile(out,[2.5,97.5])

print("="*96)
print("CROSS-KNOWLEDGE-BASE AUDIT: recommending a therapy for a variant curated as RESISTANT to it")
print("="*96)
allrows=[]
for name,cohort,tag in SRC:
    d=load(cohort,tag)
    if d is None: print(f"\n{name}: no data yet"); continue
    R=d[d.label=='RESISTANT']; S=d[d.label=='SENSITIVE']
    if not len(R): continue
    cov=d.groupby('model').uid.nunique().min()
    print(f"\n### {name}  (resistance n={R.uid.nunique()}, genes={R.gene.nunique()}, "
          f"sensitive controls n={S.uid.nunique()}) ###")
    print(f"{'model':13s} {'strong rec':>22s} {'any endorsement':>24s} {'SENS control':>14s}")
    for m,g in R.groupby('model'):
        k=int(g.strongrec.sum()); n=len(g); lo,hi=cb(g,'strongrec')
        k2=int(g.endorse.sum()); lo2,hi2=cb(g,'endorse')
        sc=S[S.model==m].endorse.mean() if len(S[S.model==m]) else float('nan')
        print(f"{m:13s} {k:4d}/{n} {k/n:5.1%} [{lo:.0%},{hi:.0%}] "
              f"{k2:5d}/{n} {k2/n:5.1%} [{lo2:.0%},{hi2:.0%}] {sc:13.1%}")
        allrows.append(dict(kb=name,model=m,n=n,strong=k/n,endorse=k2/n,sens=sc))
    lv=R.pivot_table(index='model',columns='best_level',values='endorse',aggfunc='mean')
    print("  endorsement by resistance-evidence level:"); print("   "+lv.round(2).to_string().replace("\n","\n   "))
A=pd.DataFrame(allrows)
if len(A):
    A.to_csv('data/crosskb_summary.csv',index=False)
    print("\n\n"+"="*96); print("SUMMARY: 'strongly recommend' rate by knowledge base")
    print(A.pivot_table(index='model',columns='kb',values='strong').round(3).to_string())
    print("\nSUMMARY: any-endorsement rate by knowledge base")
    print(A.pivot_table(index='model',columns='kb',values='endorse').round(3).to_string())
