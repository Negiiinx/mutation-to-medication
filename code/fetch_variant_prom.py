import pandas as pd, json, os, re, time, subprocess, urllib.parse
EU="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"; CACHE='data/pm_cache.json'
cache=json.load(open(CACHE))
def get(u):
    for i in range(4):
        r=subprocess.run(['curl','-s','--max-time','40',u],capture_output=True,text=True)
        if r.returncode==0 and r.stdout.strip(): return r.stdout
        time.sleep(2*(i+1))
    return ""
def count(term):
    if term in cache and cache[term] is not None: return cache[term]
    u=EU+"esearch.fcgi?db=pubmed&rettype=count&retmode=json&term="+urllib.parse.quote(term)
    try: v=int(json.loads(get(u))['esearchresult']['count'])
    except Exception: v=None
    cache[term]=v; time.sleep(0.36); return v

c=pd.read_csv('data/cohort_enriched.csv')
def vtok(mp):
    s=str(mp)
    m=re.search(r'\b([A-Z]\d{1,4}[A-Z*]?(?:fs)?)\b', s.split(' ',1)[1] if ' ' in s else '')
    if m: return m.group(1)
    for k in ['ITD','Exon 14 Skipping','Exon 19 Deletion','Exon 20 Insertion','Amplification','Fusion','Overexpression']:
        if k.lower() in s.lower(): return k
    return None
c['vtok']=c.molecular_profile.map(vtok)
sub=c[c.vtok.notna()].copy()
qs=sorted({(g,v) for g,v in zip(sub.gene,sub.vtok)})
print(f"variant-level queries: {len(qs)} (covering {len(sub)}/{len(c)} triples)", flush=True)
for i,(g,v) in enumerate(qs):
    count(f'"{g}"[tiab] AND "{v}"[tiab]')
    if i%100==0: json.dump(cache,open(CACHE,'w')); print(f"  {i}/{len(qs)}", flush=True)
json.dump(cache,open(CACHE,'w'))
c['n_variant']=[cache.get(f'"{g}"[tiab] AND "{v}"[tiab]') if pd.notna(v) else None for g,v in zip(c.gene,c.vtok)]
c.to_csv('data/cohort_enriched.csv',index=False)
print("DONE. variant counts resolved:", c.n_variant.notna().sum(), flush=True)
