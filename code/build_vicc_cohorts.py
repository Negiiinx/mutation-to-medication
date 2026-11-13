import json, pandas as pd, hashlib, re
d=json.load(open('data/vicc_by_source.json'))
RES={'resistant','no benefit','no responsive'}
SEN={'sensitive','responsive','sensitivity','sensitivity/response'}
LVLOK={'A','B','C','D'}
rows=[]
for src in ['oncokb','cgi','jax']:
    for x in d[src]:
        a=x.get('association',{})
        rt=str(a.get('response_type') or '').strip().lower()
        lab='RESISTANT' if rt in RES else ('SENSITIVE' if rt in SEN else None)
        if not lab: continue
        genes=x.get('genes') or []
        if not genes: continue
        g=genes[0]
        feat=(x.get('feature_names') or '').strip()
        mp=feat if feat.upper().startswith(g.upper()) else (f"{g} {feat}".strip() if feat else f"{g} Mutation")
        _d=x.get('diseases')
        dis=(_d[0] if isinstance(_d,list) and _d else _d) or a.get('disease_labels_truncated')
        if dis: dis=re.sub(r'\s*\(disease\)$','',str(dis)).strip()
        dr=x.get('drugs')
        if isinstance(dr,str) and ',' in dr: dr=dr.split(',')[0]
        tx=(dr[0] if isinstance(dr,list) and dr else dr) if dr else a.get('drug_labels')
        lvl=(a.get('evidence_label') or x.get('evidence_label') or '').strip().upper()
        if not (dis and tx and lvl in LVLOK): continue
        rows.append(dict(source=src.upper(), gene=g, molecular_profile=mp, disease=str(dis),
                         therapy=str(tx), therapies=str(tx), label=lab, best_level=lvl,
                         best_lvl={'A':5,'B':4,'C':3,'D':2}[lvl]))
t=pd.DataFrame(rows).drop_duplicates(['source','gene','molecular_profile','disease','therapy','label'])
t['uid']=[hashlib.md5(f"{r.source}|{r.molecular_profile}|{r.disease}|{r.therapies}|{r.label}".encode()).hexdigest()[:12]
          for r in t.itertuples()]
t['split']='ALL'; t['n_gene']=None; t['n_items']=1
print("PER-SOURCE COHORTS")
print(t.groupby(['source','label']).size().unstack(fill_value=0).to_string())
print("\nresistance cases by evidence level:")
print(pd.crosstab(t[t.label=='RESISTANT'].source, t[t.label=='RESISTANT'].best_level).to_string())
print("\ngenes per source:", t.groupby('source').gene.nunique().to_dict())
for s in t.source.unique():
    sub=t[t.source==s]
    sub.to_csv(f'data/cohort_{s.lower()}.csv',index=False)
    print(f"  wrote data/cohort_{s.lower()}.csv  ({len(sub)} rows)")
print("\nsample RESISTANT rows:")
print(t[t.label=='RESISTANT'][['source','gene','molecular_profile','disease','therapy','best_level']].head(10).to_string(index=False))
