"""Cross-annotate our CIViC cases through OncoKB, for case-level agreement between
knowledge bases (rather than agreement of aggregate rates)."""
import json,os,sys,re,time,subprocess,threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from oncokb_token import get_token
TOK=get_token(); B="https://www.oncokb.org/api/v1"
CACHE='data/oncokb_annot2.json'
cache=json.load(open(CACHE)) if os.path.exists(CACHE) else {}
_lock=threading.Lock()
def get(url):
    for i in range(4):
        r=subprocess.run(['curl','-s','--max-time','60',url,'-H','Authorization: Bearer '+TOK],
                         capture_output=True,text=True)
        if r.returncode==0 and r.stdout.strip().startswith('{'):
            try: return json.loads(r.stdout)
            except Exception: pass
        time.sleep(2*(i+1))
    return None
import urllib.parse as up
def annotate(gene,alt,tumor,fusion_partner=None):
    key=f"{gene}|{alt}|{tumor}|{fusion_partner}"
    if key in cache: return cache[key]
    if fusion_partner:
        url=(f"{B}/annotate/structuralVariants?hugoSymbolA={up.quote(fusion_partner)}"
             f"&hugoSymbolB={up.quote(gene)}&structuralVariantType=FUSION"
             f"&isFunctionalFusion=true&tumorType={up.quote(tumor)}")
    else:
        url=(f"{B}/annotate/mutations/byProteinChange?hugoSymbol={up.quote(gene)}"
             f"&alteration={up.quote(alt)}&tumorType={up.quote(tumor)}")
    d=get(url)
    if d is None: return None
    out={'oncogenic':d.get('oncogenic'),'highestResistanceLevel':d.get('highestResistanceLevel'),
         'treatments':[{'level':t.get('level'),'drugs':[x.get('drugName') for x in t.get('drugs',[])]}
                       for t in d.get('treatments',[])]}
    with _lock: cache[key]=out
    time.sleep(0.15)
    return out
def parse(mp):
    """molecular_profile -> (gene, alteration, fusion_partner)"""
    s=str(mp)
    # CGI style: "ESR1 (Y537S)" or "ALK (C1156Y,L1196M)" -> take gene + first variant
    m0=re.match(r'^([A-Z0-9\-]+)\s*\(([^)]+)\)\s*$', s)
    if m0:
        first=m0.group(2).split(',')[0].strip()
        if re.match(r'^[A-Za-z]\d+[A-Za-z*]*$', first): return m0.group(1), first, None
    if '::' in s:
        head=s.split(' ')[0]; parts=[p for p in head.split('::') if p and p.lower()!='v']
        if len(parts)==2: return parts[1],'Fusion',parts[0]
        if len(parts)==1: return parts[0],'Fusion',None
    m=re.match(r'^([A-Z0-9\-]+)\s+([A-Za-z]\d+[A-Za-z*]*(?:fs)?)\b',s)
    if m: return m.group(1),m.group(2),None
    m2=re.match(r'^([A-Z0-9\-]+)\s+(.*)$',s)
    if m2: return m2.group(1),m2.group(2),None
    return None,None,None
if __name__=="__main__":
    c=pd.read_csv('data/cohort_direction.csv')
    dmap=json.load(open('data/disease_to_oncotree.json'))
    jobs=[]
    for _,r in c.iterrows():
        g,a,f=parse(r.molecular_profile)
        t=dmap.get(str(r.disease)) or str(r.disease)   # OncoTree code where available
        if g: jobs.append((r.uid,g,a,t,f))
    print(f"annotating {len(jobs)} cases through OncoKB",flush=True)
    res={}; n=[0]
    def work(j):
        uid,g,a,t,f=j
        o=annotate(g,a,t,f)
        with _lock:
            res[uid]=o; n[0]+=1
            if n[0]%100==0:
                print(f"  {n[0]}/{len(jobs)}",flush=True); json.dump(cache,open(CACHE,'w'))
    with ThreadPoolExecutor(max_workers=4) as ex: list(ex.map(work,jobs))
    json.dump(cache,open(CACHE,'w'))
    json.dump({k:v for k,v in res.items() if v},open('data/oncokb_by_uid2.json','w'))
    ok=sum(1 for v in res.values() if v)
    print(f"done. annotated {ok}/{len(jobs)}",flush=True)
