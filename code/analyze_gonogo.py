import pandas as pd, numpy as np, json, re
pd.set_option('display.width',220)

c = pd.read_csv('data/cohort.csv')
cache = json.load(open('data/pm_cache.json'))
years = json.load(open('data/pmid_years.json'))

c['n_gene']  = c.gene.map(lambda g: cache.get(f'"{g}"[tiab]'))
c['n_drug']  = c.therapy.map(lambda t: cache.get(f'"{t}"[tiab]'))
c['n_pair']  = [cache.get(f'"{g}"[tiab] AND "{t}"[tiab]') for g,t in zip(c.gene,c.therapy)]
c = c.dropna(subset=['n_pair','n_gene','n_drug'])
c[['n_gene','n_drug','n_pair']] = c[['n_gene','n_drug','n_pair']].astype(int)

# era = earliest source publication year for this triple
def era(s):
    ys=[years.get(p) for p in str(s).split(',') if p and p!='nan']
    ys=[y for y in ys if y]; return min(ys) if ys else np.nan
c['era'] = c.pmids.map(era)

c['log_pair'] = np.log10(c.n_pair+1)
c['strong'] = c.best_level.isin(['A','B'])          # strong clinical evidence
c['weak']   = c.best_level.isin(['C','D','E'])
# therapy class proxy from INN stem
def tclass(t):
    t=str(t).lower()
    for stem,lab in [('tinib','kinase-inhibitor'),('ciclib','CDK-inhibitor'),('parib','PARP-inhibitor'),
                     ('zomib','proteasome-inhibitor'),('limus','mTOR-inhibitor'),('mab','antibody'),
                     ('platin','platinum'),('rubicin','anthracycline'),('taxel','taxane')]:
        if t.endswith(stem) or stem in t[-8:]: return lab
    return 'other'
c['tclass']=c.therapy.map(tclass)
c.to_csv('data/cohort_enriched.csv',index=False)

print(f"=== COHORT: {len(c)} triples with complete PubMed data ===")
print("evidence:", c.best_level.value_counts().sort_index().to_dict())
print("era available:", c.era.notna().sum(), "| era range:", c.era.min(), "-", c.era.max())
print("\nlog10(pair count) by evidence level:")
print(c.groupby('best_level').agg(n=('n_pair','size'), median_pair=('n_pair','median'),
      mean_logpair=('log_pair','mean')).round(2).to_string())
print("\nSpearman(evidence rank, log_pair):",
      round(c[['best_lvl','log_pair']].corr(method='spearman').iloc[0,1],3))

# ---------- GO/NO-GO 1: discordant quadrants ----------
med = c.log_pair.median()
c['hi_lit'] = c.log_pair > med
q = pd.crosstab(np.where(c.strong,'STRONG (A/B)','WEAK (C/D/E)'),
                np.where(c.hi_lit,'HI lit','LO lit'))
print("\n=== QUADRANTS (unmatched, median split) ===\n", q.to_string())
print(f"  * critical cells: STRONG/LO-lit = {q.loc['STRONG (A/B)','LO lit']}, WEAK/HI-lit = {q.loc['WEAK (C/D/E)','HI lit']}")

# ---------- matched discordant pairs ----------
def matched_pairs(df, keys, lit_ratio=5.0, era_tol=None):
    out=[]
    for k, g in df.groupby(keys):
        S=g[g.strong]; W=g[g.weak]
        for _,s in S.iterrows():
            for _,w in W.iterrows():
                if w.n_pair >= lit_ratio*max(s.n_pair,1):     # weak one is much MORE prominent
                    if era_tol is not None:
                        if pd.isna(s.era) or pd.isna(w.era) or abs(s.era-w.era)>era_tol: continue
                    out.append(dict(key=str(k), strong_mp=s.molecular_profile, strong_dx=s.disease,
                        strong_tx=s.therapy, strong_lvl=s.best_level, strong_n=s.n_pair, strong_era=s.era,
                        weak_mp=w.molecular_profile, weak_dx=w.disease, weak_tx=w.therapy,
                        weak_lvl=w.best_level, weak_n=w.n_pair, weak_era=w.era,
                        ratio=round(w.n_pair/max(s.n_pair,1),1)))
    return pd.DataFrame(out)

for name,keys in [("disease only",['disease']), ("disease + therapy class",['disease','tclass'])]:
    for r in [3.0,5.0,10.0]:
        mp = matched_pairs(c, keys, lit_ratio=r)
        print(f"  matched on {name:24s} lit-ratio>={r:4.0f}x : {len(mp):4d} pairs "
              f"({mp.strong_mp.nunique() if len(mp) else 0} distinct strong anchors)")

mp5 = matched_pairs(c, ['disease'], 5.0); mp5.to_csv('data/matched_pairs_disease.csv',index=False)

# ---------- GO/NO-GO 2: era-matched, volume-discordant ----------
print("\n=== ERA-MATCHED, VOLUME-DISCORDANT ===")
for tol in [2,3,5]:
    for r in [3.0,5.0]:
        mpe = matched_pairs(c, ['disease'], lit_ratio=r, era_tol=tol)
        print(f"  era tol +/-{tol}y, lit-ratio>={r:.0f}x : {len(mpe):4d} pairs")
mpe = matched_pairs(c, ['disease'], 3.0, era_tol=3); mpe.to_csv('data/matched_pairs_era.csv',index=False)
if len(mpe): print("\nExamples (era-matched, volume-discordant):\n",
    mpe[['strong_mp','strong_tx','strong_lvl','strong_n','strong_era','weak_mp','weak_tx','weak_lvl','weak_n','weak_era','ratio']].head(10).to_string(index=False))
