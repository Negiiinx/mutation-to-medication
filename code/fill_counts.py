import pandas as pd, json, subprocess, time, urllib.parse, re
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

import build_cohort_v2 as _  # rebuild not needed; read csv
