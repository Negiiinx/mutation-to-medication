import json, pandas as pd, hashlib
d=json.load(open('data/moa.json'))
TIER={'FDA-Approved':5,'Guideline':5,'Clinical trial':4,'Clinical evidence':4,'Preclinical':2,'Inferential':1}
LVL={5:'A',4:'B',2:'D',1:'E'}
rows=[]
for a in d:
    if not (a.get('therapy_resistance') or a.get('therapy_sensitivity')): continue
    tx=a.get('therapy_name'); dis=a.get('disease')
    if not tx or not dis: continue
    for f in a.get('features',[]):
        at=(f.get('attributes') or [{}])[0]
        g=at.get('gene') or at.get('gene1')
        if not g: continue
        ft=at.get('feature_type')
        if ft=='somatic_variant':
            pc=(at.get('protein_change') or '').replace('p.','')
            mp=f"{g} {pc}" if pc else f"{g} Mutation"
        elif ft=='copy_number':
            mp=f"{g} {at.get('direction','Amplification')}"
        elif ft=='rearrangement':
            mp=f"{at.get('gene1')}::{at.get('gene2')} Fusion"
        else: continue
        rows.append(dict(source='MOAlmanac', gene=g, molecular_profile=mp, disease=dis, therapy=tx,
            therapies=tx, label='RESISTANT' if a.get('therapy_resistance') else 'SENSITIVE',
            best_lvl=TIER.get(a.get('predictive_implication'),2),
            moa_tier=a.get('predictive_implication')))
t=pd.DataFrame(rows).drop_duplicates(['gene','molecular_profile','disease','therapy','label'])
t['best_level']=t.best_lvl.map(LVL)
t=t[t.best_level.notna()]
t['uid']=[hashlib.md5(f"MOA|{r.molecular_profile}|{r.disease}|{r.therapies}|{r.label}".encode()).hexdigest()[:12]
          for r in t.itertuples()]
t['split']='ALL'; t['n_gene']=None; t['n_items']=1
print(f"MOALMANAC COHORT: {len(t)} triples | {t.gene.nunique()} genes")
print(t.label.value_counts().to_dict())
print("\nresistance by tier:")
print(t[t.label=='RESISTANT'].moa_tier.value_counts().to_string())
print("\nresistance examples:")
print(t[t.label=='RESISTANT'][['gene','molecular_profile','disease','therapy','moa_tier']].head(12).to_string(index=False))
t.to_csv('data/cohort_moa.csv',index=False)
