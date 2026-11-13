import pandas as pd, numpy as np, json, glob, os
from anlib import *
os.makedirs('results',exist_ok=True); OUT={}
d=load('data/cohort_direction.csv','dir')
if d is not None:
    R=d[d.label=='RESISTANT']; S=d[d.label=='SENSITIVE']; rows=[]
    for m in mods(d):
        a=R[R.model==m]
        if a.uid.nunique()<300: continue
        (pt,lo,hi),_=matched(d[d.model==m].reset_index(drop=True))
        s1,s2=cb(a,'strongrec'); e1,e2=cb(a,'endorse')
        rows.append(dict(model=m,vendor=VENDOR[m],n=a.uid.nunique(),strong=a.strongrec.mean(),
            s_lo=s1,s_hi=s2,endorse=a.endorse.mean(),e_lo=e1,e_hi=e2,
            sens=S[S.model==m].endorse.mean(),delta=pt,d_lo=lo,d_hi=hi))
    OUT['headline']=pd.DataFrame(rows)
rows=[]
for nm,c,t in [('CIViC','data/cohort_direction.csv','dir'),('JAX-CKB','data/cohort_jax_clean.csv','jax2'),
               ('CGI','data/cohort_cgi_clean.csv','cgi2')]:
    x=load(c,t)
    if x is None: continue
    for m in mods(x):
        g=x[x.model==m].reset_index(drop=True)
        if g.uid.nunique()<100: continue
        _,per=matched(g)
        for l,v in per.items():
            rows.append(dict(kb=nm,model=m,level=l,delta=v,
                             n=int(((g.best_level==l)&(g.label=='RESISTANT')).sum())))
OUT['bylevel']=pd.DataFrame(rows)
rows=[]
for nm,c,t in [('CIViC','data/cohort_direction.csv','dir'),('JAX-CKB','data/cohort_jax_clean.csv','jax2'),
               ('CGI','data/cohort_cgi_clean.csv','cgi2'),('MOAlmanac','data/cohort_moa_clean.csv','moa2'),
               ('OncoKB','data/cohort_oncokb.csv','oncokb')]:
    x=load(c,t)
    if x is None: continue
    Rx=x[x.label=='RESISTANT']
    for m in mods(x):
        a=Rx[Rx.model==m]
        if not len(a): continue
        (pt,_,_),_=matched(x[x.model==m].reset_index(drop=True))
        rows.append(dict(kb=nm,model=m,n=a.uid.nunique(),strong=a.strongrec.mean(),
                         endorse=a.endorse.mean(),delta=pt))
OUT['crosskb']=pd.DataFrame(rows)
if d is not None and os.path.exists('data/oncokb_by_uid2.json'):
    dm=d.merge(oncokb_match(),on='uid',how='left')
    cons=dm[((dm.label=='RESISTANT')&(dm.match=='R'))|((dm.label=='SENSITIVE')&(dm.match=='S'))]
    rows=[]
    for m in mods(dm):
        g=cons[cons.model==m].reset_index(drop=True)
        Rr=g[g.label=='RESISTANT']; Ss=g[g.label=='SENSITIVE']
        if len(Rr)<20: continue
        (pt,lo,hi),_=matched(g)
        rows.append(dict(model=m,vendor=VENDOR[m],n_res=len(Rr),res_end=Rr.endorse.mean(),
            res_strong=Rr.strongrec.mean(),ctl_end=Ss.endorse.mean(),delta=pt,lo=lo,hi=hi))
    OUT['consensus']=pd.DataFrame(rows)
kn=[]
for f in glob.glob('raw_responses/know_*.jsonl'):
    for L in open(f):
        try: r=json.loads(L)
        except Exception: continue
        if 'error' in r.get('resp',{}): continue
        kn.append({'uid':r['uid'],'model':r['model'],**r['resp']})
if kn and d is not None:
    k=pd.DataFrame(kn).drop_duplicates(['uid','model'])
    j=k.merge(d[['uid','model','endorse','label']],on=['uid','model'])
    j=j[j.label=='RESISTANT']; rows=[]
    for m in mods(j):
        g=j[j.model==m]
        if len(g)<200: continue
        for lab in ['Yes, confers resistance','Unknown or insufficient evidence','No, does not confer resistance']:
            s=g[g.resistance_known==lab]
            rows.append(dict(model=m,answer=lab,n=len(s),endorse=s.endorse.mean() if len(s) else np.nan))
    OUT['knowledge']=pd.DataFrame(rows)
rag=load('data/cohort_direction_rag.csv','rag')
if d is not None and rag is not None:
    rows=[]
    for m in mods(rag):
        b=d[d.model==m]; r=rag[rag.model==m]
        if r.uid.nunique()<600: continue
        Rb=b[b.label=='RESISTANT']; Rr=r[r.label=='RESISTANT']
        Sb=b[b.label=='SENSITIVE']; Sr=r[r.label=='SENSITIVE']
        (pb,_,_),_=matched(b.reset_index(drop=True)); (pr,lo,hi),_=matched(r.reset_index(drop=True))
        rows.append(dict(model=m,vendor=VENDOR[m],end_b=Rb.endorse.mean(),end_r=Rr.endorse.mean(),
            str_b=Rb.strongrec.mean(),str_r=Rr.strongrec.mean(),sens_b=Sb.endorse.mean(),
            sens_r=Sr.endorse.mean(),d_b=pb,d_r=pr,d_lo=lo,d_hi=hi))
    OUT['mitigation']=pd.DataFrame(rows)
if d is not None:
    rows=[]
    for arm in ['RESISTANT','SENSITIVE']:
        sub=d[d.label==arm]
        for m in mods(sub):
            vc=sub[sub.model==m].recommendation.value_counts(normalize=True)
            rows.append(dict(arm=arm,model=m,**{k2:vc.get(k2,0.0) for k2 in REC}))
    OUT['dist']=pd.DataFrame(rows)
for k2,v in OUT.items():
    v.to_csv(f'results/{k2}.csv',index=False); print(f"  results/{k2}.csv ({len(v)} rows)")
if 'headline' in OUT:
    print("\n=== HEADLINE (CIViC) ===")
    for _,r in OUT['headline'].iterrows():
        print(f"  {LABEL[r.model]:20s} {r.vendor:10s} strong {r.strong:5.1%}  endorse {r.endorse:5.1%}  "
              f"sens {r.sens:5.1%}  delta {r.delta:+6.1%} [{r.d_lo:+.0%},{r.d_hi:+.0%}]")
