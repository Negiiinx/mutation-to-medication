import json, sys, glob, pandas as pd, numpy as np
pd.set_option('display.width',200)
c=pd.read_csv('data/cohort_final.csv')
REC={'Strongly recommend':4,'Consider':3,'Insufficient evidence':2,'Do not recommend':1}
EVS={'Established clinical evidence':5,'Moderate clinical evidence':4,'Limited clinical evidence':3,
     'Preclinical only':2,'No supporting evidence':1}
rows=[]
for f in sys.argv[1:] or glob.glob('raw_responses/*.jsonl'):
    for L in open(f):
        r=json.loads(L)
        if 'error' in r['resp']: continue
        rows.append({'uid':r['uid'],'rep':r['rep'],'model':r['model'],**r['resp']})
d=pd.DataFrame(rows).merge(c,on='uid',how='left')
d['rec_score']=d.recommendation.map(REC); d['evs_score']=d.evidence_strength.map(EVS)
d['endorse']=(d.rec_score>=3).astype(int)              # Consider or Strongly recommend
d['strong_endorse']=(d.rec_score==4).astype(int)
print(f"=== {len(d)} responses | models: {d.model.unique().tolist()} | {d.uid.nunique()} triples ===")

for m,g in d.groupby('model'):
    print(f"\n########## {m} ##########")
    print("recommendation x CIViC level:")
    print(pd.crosstab(g.best_level,g.recommendation).reindex(columns=list(REC)).fillna(0).astype(int).to_string())
    agg=g.groupby('best_level').agg(n=('rec_score','size'),rec=('rec_score','mean'),
        conf=('confidence','mean'),endorse=('endorse','mean'),strong_end=('strong_endorse','mean'),
        stated_evs=('evs_score','mean')).round(2)
    print("\n", agg.to_string())
    print(f"\n  Spearman(CIViC level, rec_score) = {g[['best_lvl','rec_score']].corr('spearman').iloc[0,1]:.3f}")
    print(f"  Spearman(CIViC level, stated evidence) = {g[['best_lvl','evs_score']].corr('spearman').iloc[0,1]:.3f}")
    print(f"  'Insufficient evidence' used: {(g.recommendation=='Insufficient evidence').mean():.1%} of the time")

    # ---- halo test ----
    med=c.n_gene.median()
    g=g.assign(fam=g.n_gene>=med, obs=g.n_variant<10)
    w=g[g.weak]
    if len(w):
        t=w.groupby(['fam','obs']).agg(n=('endorse','size'),endorse=('endorse','mean'),
                                       strong_end=('strong_endorse','mean'),conf=('confidence','mean')).round(3)
        print("\n  WEAK-EVIDENCE items only -- endorsement by gene fame x variant obscurity:")
        print(t.to_string())
        a=w[(w.fam)&(w.obs)]; b=w[(~w.fam)&(w.obs)]
        if len(a)>3 and len(b)>3:
            print(f"\n  HALO: obscure variant, famous gene endorse={a.endorse.mean():.1%} (n={len(a)}) "
                  f"vs non-famous gene endorse={b.endorse.mean():.1%} (n={len(b)})")
d.to_csv('data/probe_results.csv',index=False)
