"""Probe LLM therapeutic actionability judgments on CIViC-derived vignettes.
Blinded: nothing in the prompt reveals CIViC evidence level or literature counts."""
import json, os, sys, time, argparse, hashlib, subprocess, threading
import pandas as pd, numpy as np
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from keyloader import get_key
KEY = get_key()

SCHEMA = {"type":"object","additionalProperties":False,
 "properties":{
  "recommendation":{"type":"string","enum":["Strongly recommend","Consider","Insufficient evidence","Do not recommend"]},
  "evidence_strength":{"type":"string","enum":["Established clinical evidence","Moderate clinical evidence","Limited clinical evidence","Preclinical only","No supporting evidence"]},
  "confidence":{"type":"integer"}},
 "required":["recommendation","evidence_strength","confidence"]}

SYSTEM = ("You are assisting a molecular tumor board. Judge whether a proposed therapy is "
          "clinically actionable for the specific molecular alteration described. Base your judgment "
          "on the strength of evidence for THIS specific alteration, not the gene in general. "
          "Report confidence as an integer 0-100.")

def vignette(r):
    return (f"A 62-year-old patient with metastatic {r.disease}. "
            f"Tumor molecular profiling identifies: {r.molecular_profile}. "
            f"The tumor board is considering {r.therapy}.\n\n"
            f"Is {r.therapy} clinically actionable for this patient's specific alteration?")

_lock = threading.Lock()
def call(model, msg, temp, seed=None):
    body = {"model":model,"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":msg}],
            "response_format":{"type":"json_schema","json_schema":{"name":"actionability","strict":True,"schema":SCHEMA}},
            "max_completion_tokens":2000 if model.startswith(("o1","o3","o4","gpt-5")) else 200}
    if temp is not None and not model.startswith(("o1","o3","o4","gpt-5")): body["temperature"]=temp
    if seed is not None: body["seed"]=seed
    for attempt in range(5):
        p = subprocess.run(['curl','-s','--max-time','90','https://api.openai.com/v1/chat/completions',
             '-H','Content-Type: application/json','-H','Authorization: Bearer '+KEY,'-d',json.dumps(body)],
             capture_output=True, text=True)
        try:
            d = json.loads(p.stdout)
            if 'error' in d:
                if 'rate' in d['error'].get('message','').lower(): time.sleep(5*(attempt+1)); continue
                return {"error": d['error'].get('message','')[:200]}
            return json.loads(d['choices'][0]['message']['content'])
        except Exception:
            time.sleep(2*(attempt+1))
    return {"error":"failed"}

def run(df, model, reps, temp, outpath):
    done = {}
    if os.path.exists(outpath):
        for L in open(outpath):
            try:
                r=json.loads(L)
                # only treat a row as done if it actually succeeded; error rows must retry
                if 'error' not in r.get('resp',{}): done[(r['uid'],r['rep'])]=1
            except Exception: pass
    jobs = [(i,r,k) for i,r in df.iterrows() for k in range(reps) if (r.uid,k) not in done]
    print(f"{model}: {len(jobs)} calls to make ({len(done)} cached)", flush=True)
    f = open(outpath,'a'); n=[0]
    def work(j):
        i,r,k = j
        out = call(model, vignette(r), temp, seed=1000+k)
        rec = {"uid":r.uid,"rep":k,"model":model,"temp":temp,**({"resp":out})}
        with _lock:
            f.write(json.dumps(rec)+"\n"); f.flush(); n[0]+=1
            if n[0]%100==0: print(f"  {n[0]}/{len(jobs)}", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex: list(ex.map(work, jobs))
    f.close(); print(f"{model}: done", flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='gpt-4o-mini'); ap.add_argument('--reps',type=int,default=1)
    ap.add_argument('--temp',type=float,default=0.0); ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--tag',default='pilot')
    a=ap.parse_args()
    c=pd.read_csv(os.environ.get('COHORT','data/cohort_v2.csv'))
    if 'uid' not in c.columns:
        c['uid']=[hashlib.md5(f"{r.molecular_profile}|{r.disease}|{r.therapy}".encode()).hexdigest()[:12] for r in c.itertuples()]
    sub=c.copy()
    if a.limit:
        idx=[i for _,gg in sub.groupby('best_level') for i in gg.sample(min(len(gg),a.limit),random_state=0).index]
        sub=sub.loc[idx]
    os.makedirs('raw_responses',exist_ok=True)
    print(f"analysis set: {len(sub)} triples across {sub.gene.nunique()} genes; levels {sub.best_level.value_counts().sort_index().to_dict()}")
    run(sub, a.model, a.reps, a.temp, f'raw_responses/{a.tag}_{a.model}.jsonl')
