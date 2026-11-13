import pandas as pd, numpy as np, json, glob, sys, re
sys.path.insert(0,'.'); from stats_core import ols_cluster
rng=np.random.default_rng(7)
REC={'Strongly recommend':4,'Consider':3,'Insufficient evidence':2,'Do not recommend':1}
EVS={'Established clinical evidence':5,'Moderate clinical evidence':4,'Limited clinical evidence':3,
     'Preclinical only':2,'No supporting evidence':1}
c=pd.read_csv('data/cohort_v2.csv')
ev=pd.read_csv('data/nightly-ClinicalEvidenceSummaries.tsv',sep='\t',dtype=str)
def gfm(mp):
    s=str(mp)
    if '::' in s:
        p=[x for x in re.split(r'::',s.split(' ')[0]) if x and x.lower()!='v']
        if p: return p[-1]
    t=re.sub(r'[^A-Za-z0-9\-]','',s.split(' ')[0]); return t if len(t)>1 else None
ev['gene']=ev.molecular_profile.map(gfm)
c=c.merge(ev.groupby('gene').size().rename('civic_n'),left_on='gene',right_index=True,how='left')
rows=[]
for f in glob.glob('raw_responses/main_*.jsonl'):
    for L in open(f):
        r=json.loads(L)
        if 'error' in r['resp']: continue
        rows.append({'uid':r['uid'],'model':r['model'],**r['resp']})
d=pd.DataFrame(rows).drop_duplicates(['uid','model']).merge(c,on='uid',how='inner')
d['rec_score']=d.recommendation.map(REC); d['endorse']=(d.rec_score>=3).astype(int)
d['lvl']=d.best_lvl.astype(float); d['lg']=np.log10(d.n_gene+1); d['lc']=np.log10(d.civic_n.fillna(0)+1)
for x in ['lvl','lg','lc']: d[x+'_c']=d[x]-d[x].mean()

def wild_boot(y,X,cl,k,B=2999):
    r,_=ols_cluster(y,X,cl,[f'v{i}' for i in range(X.shape[1])]); t_obs=r.iloc[k]['t']
    Xr=np.delete(X,k,axis=1); br=np.linalg.pinv(Xr.T@Xr)@Xr.T@y; yhat=Xr@br; er=y-yhat
    s=pd.Series(cl).reset_index(drop=True); groups={g:i.values for g,i in s.groupby(s).groups.items()}
    cnt=0
    for _ in range(B):
        ys=yhat.copy()
        for g,i in groups.items(): ys[i]=yhat[i]+(1.0 if rng.random()<0.5 else -1.0)*er[i]
        rb,_=ols_cluster(ys,X,cl,[f'v{i}' for i in range(X.shape[1])])
        if abs(rb.iloc[k]['t'])>=abs(t_obs): cnt+=1
    return t_obs,(cnt+1)/(B+1)

print(f"EXPANDED COHORT: {d.uid.nunique()} triples, {d.gene.nunique()} gene clusters")
print(f"models present: {sorted(d.model.unique())}\n")
print(f"{'model':14s} {'n':>5} {'clust':>5} {'bE':>8} {'bP':>8} {'bEP':>8} {'p analytic':>11} {'p bootstrap':>12} {'bEP|civic':>10}")
for m,g in d.groupby('model'):
    g=g.reset_index(drop=True)
    if g.gene.nunique()<10: continue
    X=np.column_stack([np.ones(len(g)),g.lvl_c,g.lg_c,g.lvl_c*g.lg_c])
    r,ng=ols_cluster(g.endorse,X,g.gene,['b0','bE','bP','bEP'])
    X2=np.column_stack([np.ones(len(g)),g.lvl_c,g.lg_c,g.lvl_c*g.lg_c,g.lc_c,g.lvl_c*g.lc_c])
    r2,_=ols_cluster(g.endorse,X2,g.gene,['b0','bE','bP','bEP','bC','bExC'])
    t,pb=wild_boot(g.endorse.values.astype(float),X,g.gene.values,3)
    print(f"{m:14s} {len(g):5d} {ng:5d} {r.loc['bE','coef']:+8.4f} {r.loc['bP','coef']:+8.4f} "
          f"{r.loc['bEP','coef']:+8.4f} {r.loc['bEP','p']:11.4f} {pb:12.4f} {r2.loc['bEP','coef']:+10.4f}")
d.to_csv('data/probe_results_v2.csv',index=False)
