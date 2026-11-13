"""Blinded clinician adjudication packet.
Cases where a model recommended a therapy against a variant curated as resistant to it.
Prioritised by agreement across models and across knowledge bases."""
import pandas as pd, numpy as np, json, glob, os
REC={'Strongly recommend':4,'Consider':3,'Insufficient evidence':2,'Do not recommend':1}
SRC=[('CIViC','data/cohort_direction.csv','dir'),('MOAlmanac','data/cohort_moa.csv','moa'),
     ('CGI','data/cohort_cgi.csv','cgi'),('OncoKB','data/cohort_oncokb.csv','oncokb'),
     ('JAX-CKB','data/cohort_jax_sub.csv','jax')]
frames=[]
for name,coh,tag in SRC:
    if not os.path.exists(coh): continue
    c=pd.read_csv(coh); rows=[]
    for f in glob.glob(f'raw_responses/{tag}_*.jsonl'):
        for L in open(f):
            r=json.loads(L)
            if 'error' in r['resp']: continue
            rows.append({'uid':r['uid'],'model':r['model'],**r['resp']})
    if not rows: continue
    d=pd.DataFrame(rows).drop_duplicates(['uid','model']).merge(c,on='uid',how='inner')
    d['kb']=name; frames.append(d)
d=pd.concat(frames,ignore_index=True)
d['rec']=d.recommendation.map(REC)
d['strongrec']=(d.recommendation=='Strongly recommend').astype(int)
R=d[d.label=='RESISTANT'].copy()

# blind the models
models=sorted(R.model.unique()); blind={m:f"Model {chr(65+i)}" for i,m in enumerate(models)}
R['blinded']=R.model.map(blind)

# a "case" = alteration + cancer + therapy ; may appear in >1 KB
key=['gene','molecular_profile','disease','therapy']
flag=R[R.strongrec==1]
agg=flag.groupby(key).agg(
    n_models_strongly_rec=('blinded','nunique'),
    which_models=('blinded',lambda s:', '.join(sorted(set(s)))),
    kbs=('kb',lambda s:', '.join(sorted(set(s)))),
    n_kbs=('kb','nunique'),
    kb_evidence_level=('best_level',lambda s:'/'.join(sorted(set(map(str,s))))),
    model_confidence=('confidence','mean'),
).reset_index()
agg=agg.sort_values(['n_kbs','n_models_strongly_rec','gene'],ascending=[False,False,True])
out=pd.DataFrame({
 'Case ID':[f"R{i+1:03d}" for i in range(len(agg))],
 'Cancer type':agg.disease, 'Molecular alteration':agg.molecular_profile,
 'Proposed therapy':agg.therapy, 'Gene':agg.gene,
 'Knowledge base(s) stating RESISTANCE':agg.kbs,
 'KB evidence level for the resistance':agg.kb_evidence_level,
 '# knowledge bases':agg.n_kbs,
 '# models that STRONGLY RECOMMENDED it':agg.n_models_strongly_rec,
 'Which models':agg.which_models,
 'Mean model confidence (0-100)':agg.model_confidence.round(0),
 'Is this a genuine clinical error? (Yes / No / Equivocal)':'',
 'If No or Equivocal, why?':'',
 'Reviewer comments':'',
})
with pd.ExcelWriter('Clinician_Adjudication_Packet.xlsx',engine='openpyxl') as xl:
    inst=pd.DataFrame({'':[
      'CLINICIAN ADJUDICATION PACKET',
      '',
      'What this is:',
      'Each row is a case where at least one large language model issued an unqualified',
      '"strongly recommend" for a therapy, when a curated oncology knowledge base states that the',
      'listed molecular alteration confers RESISTANCE to that therapy.',
      '',
      'What we are asking:',
      'For each row, judge whether recommending this therapy for this alteration in this cancer',
      'would be a genuine clinical error. Please use the Yes / No / Equivocal column.',
      '',
      'Answer "No" or "Equivocal" if, for example: the resistance evidence is weak or contested,',
      'the drug would still reasonably be considered in some clinical context, the alteration is',
      'too non-specific to act on, or the knowledge base entry looks wrong.',
      '',
      'Notes:',
      '- Model identities are blinded (Model A / B / C).',
      '- Rows are sorted so the most corroborated cases appear first (agreement across knowledge',
      '  bases, then across models).',
      '- KB evidence level: A = validated/guideline, B = clinical, C = case study, D = preclinical.',
      '- We are testing the models, not the knowledge bases, but if you think a KB entry is wrong',
      '  that is exactly what we need to know.',
      '',
      f'Total cases in this packet: {len(out)}',
      f'Cases corroborated by 2 or more knowledge bases: {int((agg.n_kbs>=2).sum())}',
      f'Cases strongly recommended by 2 or more models: {int((agg.n_models_strongly_rec>=2).sum())}',
    ]})
    inst.to_excel(xl,sheet_name='Instructions',index=False)
    out.to_excel(xl,sheet_name='Cases for review',index=False)
    ws=xl.sheets['Cases for review']
    for col,w in zip('ABCDEFGHIJKLMN',[9,28,30,26,10,26,14,10,14,18,14,30,26,30]):
        ws.column_dimensions[col].width=w
    ws.freeze_panes='A2'
    ws2=xl.sheets['Instructions']; ws2.column_dimensions['A'].width=100
print(f"packet: {len(out)} cases")
print(f"  corroborated by >=2 KBs: {(agg.n_kbs>=2).sum()}")
print(f"  strongly recommended by >=2 models: {(agg.n_models_strongly_rec>=2).sum()}")
print(f"  by all {len(models)} models: {(agg.n_models_strongly_rec==len(models)).sum()}")
print("\nblinding map (KEEP PRIVATE):",blind)
print("\ntop rows:")
print(out.head(12)[['Case ID','Cancer type','Molecular alteration','Proposed therapy',
                    'Knowledge base(s) stating RESISTANCE','# models that STRONGLY RECOMMENDED it']].to_string(index=False))
json.dump(blind,open('data/blinding_key.json','w'))
