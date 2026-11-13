import pandas as pd, numpy as np, json, glob, os, re
REC={'Strongly recommend':4,'Consider':3,'Insufficient evidence':2,'Do not recommend':1}
LEVS=['A','B','C','D']
ORDER=['o4-mini','gpt-4.1','gpt-4o','gpt-4o-mini','claude-opus-5','claude-haiku-4-5',
       'gemini-2.5-flash','qwen2.5:7b','medgemma:4b']  # gemini-2.5-pro excluded: daily quota. gpt-5 excluded pending completion of all its arms.
LABEL={'gpt-5':'GPT-5','o4-mini':'o4-mini','gpt-4.1':'GPT-4.1','gpt-4o':'GPT-4o','gpt-4o-mini':'GPT-4o-mini',
 'claude-opus-5':'Claude Opus 5','claude-haiku-4-5':'Claude Haiku 4.5','gemini-2.5-pro':'Gemini 2.5 Pro',
 'gemini-2.5-flash':'Gemini 2.5 Flash','qwen2.5:7b':'Qwen 2.5 7B','medgemma:4b':'MedGemma 4B'}
VENDOR={'gpt-5':'OpenAI','o4-mini':'OpenAI','gpt-4.1':'OpenAI','gpt-4o':'OpenAI','gpt-4o-mini':'OpenAI',
 'claude-opus-5':'Anthropic','claude-haiku-4-5':'Anthropic','gemini-2.5-pro':'Google',
 'gemini-2.5-flash':'Google','qwen2.5:7b':'open','medgemma:4b':'open'}
def load(cohort,tag):
    if not os.path.exists(cohort): return None
    co=pd.read_csv(cohort); rows=[]
    for f in glob.glob(f'raw_responses/{tag}_*.jsonl'):
        for L in open(f):
            try: r=json.loads(L)
            except Exception: continue
            if 'error' in r.get('resp',{}): continue
            rows.append({'uid':r['uid'],'model':r['model'],**r['resp']})
    if not rows: return None
    d=pd.DataFrame(rows).drop_duplicates(['uid','model']).merge(co,on='uid',how='inner')
    d['endorse']=(d.recommendation.map(REC)>=3).astype(int)
    d['strongrec']=(d.recommendation=='Strongly recommend').astype(int)
    return d
def cb(s,col,B=3000,seed=5):
    if not len(s) or s.gene.nunique()<2: return (np.nan,np.nan)
    rng=np.random.default_rng(seed); gs=s.gene.unique(); byg={g:s[s.gene==g][col].values for g in gs}
    o=[np.concatenate([byg[g] for g in rng.choice(gs,len(gs),replace=True)]).mean() for _ in range(B)]
    return tuple(np.percentile(o,[2.5,97.5]))
def matched(g,B=1500,seed=11):
    if not len(g): return (np.nan,np.nan,np.nan),{}
    lv=np.asarray(g.best_level); res=np.asarray(g.label)=='RESISTANT'; e=g.endorse.values
    genes,gidx=np.unique(np.asarray(g.gene),return_inverse=True)
    ok=[l for l in LEVS if ((lv==l)&res).sum()>=5 and ((lv==l)&~res).sum()>=5]
    if not ok: return (np.nan,np.nan,np.nan),{}
    w=np.array([((lv==l)&res).sum() for l in ok],float)
    per={l:e[(lv==l)&~res].mean()-e[(lv==l)&res].mean() for l in ok}
    pt=np.average([per[l] for l in ok],weights=w)
    rng=np.random.default_rng(seed); out=[]; bg=[np.where(gidx==i)[0] for i in range(len(genes))]
    for _ in range(B):
        sel=np.concatenate([bg[i] for i in rng.integers(0,len(genes),len(genes))])
        L2=lv[sel];R2=res[sel];E=e[sel]; vals=[];ws=[]
        for l in ok:
            a=E[(L2==l)&~R2]; b=E[(L2==l)&R2]
            if len(a)>=3 and len(b)>=3: vals.append(a.mean()-b.mean()); ws.append(len(b))
        if vals: out.append(np.average(vals,weights=ws))
    lo,hi=np.percentile(out,[2.5,97.5]) if out else (np.nan,np.nan)
    return (pt,lo,hi),per
def mods(d): return [m for m in ORDER if m in set(d.model)]
def oncokb_match():
    ann=json.load(open('data/oncokb_by_uid2.json')); co=pd.read_csv('data/cohort_direction.csv')
    n=lambda x: re.sub(r'[^a-z]','',str(x).lower())
    out=[]
    for uid,a in ann.items():
        rr=co[co.uid==uid]
        if not len(rr): continue
        rr=rr.iloc[0]; ours=[n(t) for t in str(rr.therapy).split(',')]
        lv=None
        for t in a.get('treatments',[]):
            dr=[n(z) for z in t.get('drugs',[])]
            if any(o and any(o in z or z in o for z in dr) for o in ours):
                L=str(t.get('level'))
                if L.startswith('LEVEL_R'): lv='R'
                elif lv is None: lv='S'
        out.append(dict(uid=uid,match=lv))
    return pd.DataFrame(out)
