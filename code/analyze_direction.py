"""Pre-registered analysis. Default = EXPLORE half only. --confirm unlocks the held-out half."""
import pandas as pd, numpy as np, json, glob, sys, argparse
sys.path.insert(0,'.'); from stats_core import ols_cluster
ap=argparse.ArgumentParser(); ap.add_argument('--confirm',action='store_true')
ap.add_argument('--split',default=None); a=ap.parse_args()

REC={'Strongly recommend':4,'Consider':3,'Insufficient evidence':2,'Do not recommend':1}
c=pd.read_csv('data/cohort_direction.csv')
rows=[]
for f in glob.glob('raw_responses/dir_*.jsonl'):
    for L in open(f):
        r=json.loads(L)
        if 'error' in r['resp']: continue
        rows.append({'uid':r['uid'],'model':r['model'],**r['resp']})
d=pd.DataFrame(rows).drop_duplicates(['uid','model']).merge(c,on='uid',how='inner')
d['rec_score']=d.recommendation.map(REC); d['endorse']=(d.rec_score>=3).astype(int)
d['resistant']=(d.label=='RESISTANT').astype(int)
d['pair']=d.gene+'|'+d.therapies

use = a.split if a.split else ('CONFIRM' if a.confirm else 'EXPLORE')
d=d[d.split==use]
print(f"### {use} HALF ###  {d.uid.nunique()} triples, {d.gene.nunique()} genes, "
      f"{d.pair.nunique()} gene-drug pairs, models {sorted(d.model.unique())}\n")

for m,g in d.groupby('model'):
    g=g.reset_index(drop=True)
    er_s=g[g.resistant==0].endorse.mean(); er_r=g[g.resistant==1].endorse.mean()
    print(f"--- {m} ---")
    print(f"  endorsement: SENSITIVE {er_s:.1%} (n={(g.resistant==0).sum()})   "
          f"RESISTANT {er_r:.1%} (n={(g.resistant==1).sum()})   raw gap {er_r-er_s:+.1%}")
    # within-pair fixed effects via demeaning
    keep=g.groupby('pair').resistant.nunique(); ok=set(keep[keep==2].index)
    gg=g[g.pair.isin(ok)].reset_index(drop=True)
    if len(gg)<30: print("  too few within-pair cases\n"); continue
    y=gg.endorse-gg.groupby('pair').endorse.transform('mean')
    x=gg.resistant-gg.groupby('pair').resistant.transform('mean')
    X=np.column_stack([x])
    r,ng=ols_cluster(y.values,X,gg.gene.values,['resistant'])
    print(f"  within-pair effect of RESISTANT on endorsement: {r.loc['resistant','coef']:+.4f} "
          f"(p={r.loc['resistant','p']:.4f}, {ng} gene clusters, {gg.pair.nunique()} pairs, n={len(gg)})")
    strong=(g[g.resistant==1].recommendation=='Strongly recommend').mean()
    print(f"  'Strongly recommend' given a RESISTANCE variant: {strong:.1%}")
    print(f"  endorsement of resistant by evidence level: "
          f"{g[g.resistant==1].groupby('best_level').endorse.mean().round(2).to_dict()}\n")
