import pandas as pd, json, os, time, subprocess, urllib.parse, re, sys

EU="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
CACHE='data/pm_cache.json'
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

def get(url, tries=4):
    for i in range(tries):
        r = subprocess.run(['curl','-s','--max-time','40',url], capture_output=True, text=True)
        if r.returncode==0 and r.stdout.strip(): return r.stdout
        time.sleep(2*(i+1))
    return ""

def count(term):
    if term in cache and cache[term] is not None: return cache[term]
    url = EU+"esearch.fcgi?db=pubmed&rettype=count&retmode=json&term="+urllib.parse.quote(term)
    try: c = int(json.loads(get(url))['esearchresult']['count'])
    except Exception: c = None
    cache[term]=c; time.sleep(0.36)
    if len(cache)%100==0:
        json.dump(cache, open(CACHE,'w')); print(f"  ...{len(cache)} cached", flush=True)
    return c

# ---- 1. publication years ----
pmids=[p.strip() for p in open('data/pmids.txt') if p.strip()]
years = json.load(open('data/pmid_years.json')) if os.path.exists('data/pmid_years.json') else {}
todo=[p for p in pmids if p not in years]
print(f"PMID years to fetch: {len(todo)}", flush=True)
for i in range(0,len(todo),200):
    b=todo[i:i+200]
    try:
        d=json.loads(get(EU+"esummary.fcgi?db=pubmed&retmode=json&id="+",".join(b)))['result']
        for p in b:
            pdte=(d.get(p,{}).get('pubdate') or d.get(p,{}).get('epubdate') or '')
            m=re.match(r'(\d{4})', str(pdte)); years[p]=int(m.group(1)) if m else None
    except Exception as ex: print("  batch err:", ex, flush=True)
    time.sleep(0.36); print(f"  years {min(i+200,len(todo))}/{len(todo)}", flush=True)
json.dump(years, open('data/pmid_years.json','w'))
print("years resolved:", sum(1 for v in years.values() if v), "/", len(years), flush=True)

# ---- 2. prominence counts ----
c = pd.read_csv('data/cohort.csv')
genes=sorted(c.gene.unique()); drugs=sorted(c.therapy.unique())
pairs=sorted({(g,t) for g,t in zip(c.gene,c.therapy)})
print(f"queries: genes={len(genes)} drugs={len(drugs)} pairs={len(pairs)}", flush=True)
for g in genes: count(f'"{g}"[tiab]')
json.dump(cache, open(CACHE,'w')); print("genes done", flush=True)
for d in drugs: count(f'"{d}"[tiab]')
json.dump(cache, open(CACHE,'w')); print("drugs done", flush=True)
for n,(g,t) in enumerate(pairs):
    count(f'"{g}"[tiab] AND "{t}"[tiab]')
    if n%200==0: print(f"  pairs {n}/{len(pairs)}", flush=True)
json.dump(cache, open(CACHE,'w'))
ok=sum(1 for v in cache.values() if v is not None)
print(f"DONE. cache={len(cache)} resolved={ok}", flush=True)
