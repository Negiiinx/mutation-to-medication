"""Cross-annotate JAX-CKB and CGI cohorts through OncoKB."""
import json,os,sys,re,time,subprocess,threading
import pandas as pd, urllib.parse as up
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from oncokb_token import get_token
from annotate_oncokb import parse
TOK=get_token(); B="https://www.oncokb.org/api/v1"
CACHE='data/oncokb_annot_all.json'
cache=json.load(open(CACHE)) if os.path.exists(CACHE) else {}
_lock=threading.Lock()
def get(url):
    for i in range(3):
        r=subprocess.run(['curl','-s','--max-time','60',url,'-H','Authorization: Bearer '+TOK],
                         capture_output=True,text=True)
        if r.returncode==0 and r.stdout.strip().startswith('{'):
            try: return json.loads(r.stdout)
            except Exception: pass
        time.sleep(2*(i+1))
    return None
def annotate(gene,alt,tumor,fp=None):
    key=f"{gene}|{alt}|{tumor}|{fp}"
    if key in cache: return cache[key]
    if fp:
        url=(f"{B}/annotate/structuralVariants?hugoSymbolA={up.quote(fp)}&hugoSymbolB={up.quote(gene)}"
             f"&structuralVariantType=FUSION&isFunctionalFusion=true&tumorType={up.quote(tumor)}")
    else:
        url=(f"{B}/annotate/mutations/byProteinChange?hugoSymbol={up.quote(gene)}"
             f"&alteration={up.quote(alt)}&tumorType={up.quote(tumor)}")
    d=get(url)
    if d is None: return None
    o={'oncogenic':d.get('oncogenic'),'highestResistanceLevel':d.get('highestResistanceLevel'),
       'treatments':[{'level':t.get('level'),'drugs':[x.get('drugName') for x in t.get('drugs',[])]}
                     for t in d.get('treatments',[])]}
    with _lock: cache[key]=o
    time.sleep(0.12); return o
if __name__=="__main__":
    dmap=json.load(open('data/disease_to_oncotree_all.json'))
    dmap={k.lower():v for k,v in dmap.items()}
    for nm,f,out in [('CGI','data/cohort_cgi_clean.csv','data/oncokb_cgi.json')]:
        c=pd.read_csv(f)
        jobs=[]
        for _,r in c.iterrows():
            g,a,fp=parse(r.molecular_profile)
            t=dmap.get(str(r.disease).lower())
            if g and t: jobs.append((r.uid,g,a,t,fp))
        print(f"{nm}: {len(jobs)} annotatable of {len(c)}",flush=True)
        res={}; n=[0]
        def work(j):
            uid,g,a,t,fp=j; o=annotate(g,a,t,fp)
            with _lock:
                res[uid]=o; n[0]+=1
                if n[0]%200==0: print(f"  {n[0]}/{len(jobs)}",flush=True); json.dump(cache,open(CACHE,'w'))
        with ThreadPoolExecutor(max_workers=4) as ex: list(ex.map(work,jobs))
        json.dump(cache,open(CACHE,'w'))
        json.dump({k:v for k,v in res.items() if v},open(out,'w'))
        print(f"{nm}: annotated {sum(1 for v in res.values() if v)}",flush=True)
