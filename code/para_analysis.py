import pandas as pd, numpy as np, json, glob
REC={'Strongly recommend':4,'Consider':3,'Insufficient evidence':2,'Do not recommend':1}
co=pd.read_csv('data/cohort_direction.csv'); rows=[]
for f in glob.glob('raw_responses/dir_*.jsonl'):
    m=f.split('dir_')[1].replace('.jsonl','')
    if m not in ['gpt-4.1','gpt-4o','gpt-4o-mini']: continue
    for L in open(f):
        r=json.loads(L)
        if 'error' in r['resp']: continue
        rows.append({'uid':r['uid'],'model':m,'template':'t1',**r['resp']})
for f in glob.glob('raw_responses/parat*_*.jsonl'):
    t=f.split('para')[1][:2]
    for L in open(f):
        r=json.loads(L)
        if 'error' in r['resp']: continue
        rows.append({'uid':r['uid'],'model':r['model'],'template':t,**r['resp']})
d=pd.DataFrame(rows).drop_duplicates(['uid','model','template']).merge(co,on='uid',how='inner')
d['endorse']=(d.recommendation.map(REC)>=3).astype(int)
d['strongrec']=(d.recommendation=='Strongly recommend').astype(int)
LEVS=['A','B','C','D']
def matched(g,B=1200,seed=11):
    lv=np.asarray(g.best_level); res=np.asarray(g.label)=='RESISTANT'; e=g.endorse.values
    genes,gidx=np.unique(np.asarray(g.gene),return_inverse=True)
    ok=[l for l in LEVS if ((lv==l)&res).sum()>=5 and ((lv==l)&~res).sum()>=5]
    w=np.array([((lv==l)&res).sum() for l in ok],float)
    pt=np.average([e[(lv==l)&~res].mean()-e[(lv==l)&res].mean() for l in ok],weights=w)
    rng=np.random.default_rng(seed); out=[]; bg=[np.where(gidx==i)[0] for i in range(len(genes))]
    for _ in range(B):
        sel=np.concatenate([bg[i] for i in rng.integers(0,len(genes),len(genes))])
        L2=lv[sel];R2=res[sel];E=e[sel]; vals=[];ws=[]
        for l in ok:
            a=E[(L2==l)&~R2]; b=E[(L2==l)&R2]
            if len(a)>=3 and len(b)>=3: vals.append(a.mean()-b.mean()); ws.append(len(b))
        if vals: out.append(np.average(vals,weights=ws))
    lo,hi=np.percentile(out,[2.5,97.5])
    return pt,lo,hi
print("coverage:"); print(d.groupby(['model','template']).uid.nunique().unstack().to_string())
print("\nMATCHED DELTA BY PROMPT TEMPLATE (CIViC, 730 cases each)")
print(f"{'model':13s} {'T1 (original)':>24s} {'T2':>24s} {'T3':>24s}")
for m in ['gpt-4.1','gpt-4o','gpt-4o-mini']:
    cells=[]
    for t in ['t1','t2','t3']:
        g=d[(d.model==m)&(d.template==t)].reset_index(drop=True)
        pt,lo,hi=matched(g); cells.append(f"{pt:+.1%} [{lo:+.0%},{hi:+.0%}]")
    print(f"{m:13s} "+" ".join(f"{c:>24s}" for c in cells))
R=d[d.label=='RESISTANT']
print("\nraw endorsement of resistance cases (%)")
print((R.pivot_table(index='model',columns='template',values='endorse',aggfunc='mean')*100).round(1).to_string())
print("\nstrongly-recommend rate (%)")
print((R.pivot_table(index='model',columns='template',values='strongrec',aggfunc='mean')*100).round(1).to_string())
