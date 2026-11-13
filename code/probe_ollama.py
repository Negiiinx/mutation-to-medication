"""Same probe, against local Ollama models. Identical prompt and schema as probe.py."""
import json, os, sys, time, argparse, hashlib, subprocess, threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
SCHEMA={"type":"object",
 "properties":{
  "recommendation":{"type":"string","enum":["Strongly recommend","Consider","Insufficient evidence","Do not recommend"]},
  "evidence_strength":{"type":"string","enum":["Established clinical evidence","Moderate clinical evidence","Limited clinical evidence","Preclinical only","No supporting evidence"]},
  "confidence":{"type":"integer"}},
 "required":["recommendation","evidence_strength","confidence"]}
SYSTEM=("You are assisting a molecular tumor board. Judge whether a proposed therapy is "
        "clinically actionable for the specific molecular alteration described. Base your judgment "
        "on the strength of evidence for THIS specific alteration, not the gene in general. "
        "Report confidence as an integer 0-100.")
def vignette(r):
    return (f"A 62-year-old patient with metastatic {r.disease}. "
            f"Tumor molecular profiling identifies: {r.molecular_profile}. "
            f"The tumor board is considering {r.therapy}.\n\n"
            f"Is {r.therapy} clinically actionable for this patient's specific alteration?")
_lock=threading.Lock()
def call(model,msg):
    body={"model":model,"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":msg}],
          "format":SCHEMA,"stream":False,"options":{"temperature":0.0,"num_predict":200}}
    for a in range(4):
        p=subprocess.run(['curl','-s','--max-time','300','http://localhost:11434/api/chat',
            '-H','Content-Type: application/json','-d',json.dumps(body)],capture_output=True,text=True)
        try:
            d=json.loads(p.stdout)
            if 'error' in d: return {"error":str(d['error'])[:200]}
            return json.loads(d['message']['content'])
        except Exception: time.sleep(2*(a+1))
    return {"error":"failed"}
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True)
    ap.add_argument('--limit',type=int,default=0); ap.add_argument('--tag',default='main')
    ap.add_argument('--workers',type=int,default=2)
    a=ap.parse_args()
    c=pd.read_csv(os.environ.get('COHORT','data/cohort_v2.csv'))
    if a.limit: c=c.head(a.limit)
    os.makedirs('raw_responses',exist_ok=True)
    out=f"raw_responses/{a.tag}_{a.model.replace(':','-')}.jsonl"
    done=set()
    if os.path.exists(out):
        for L in open(out):
            try:
                r=json.loads(L)
                if 'error' not in r.get('resp',{}): done.add(r['uid'])
            except Exception: pass
    jobs=[r for _,r in c.iterrows() if r.uid not in done]
    print(f"{a.model}: {len(jobs)} to do ({len(done)} cached)",flush=True)
    f=open(out,'a'); n=[0]; t0=time.time()
    def work(r):
        o=call(a.model,vignette(r))
        with _lock:
            f.write(json.dumps({"uid":r.uid,"rep":0,"model":a.model,"temp":0.0,"resp":o})+"\n"); f.flush()
            n[0]+=1
            if n[0]%50==0:
                el=time.time()-t0
                print(f"  {n[0]}/{len(jobs)}  {el/n[0]:.1f}s/case  eta {(len(jobs)-n[0])*el/n[0]/60:.0f}min",flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex: list(ex.map(work,jobs))
    f.close(); print(f"{a.model}: done in {(time.time()-t0)/60:.1f} min",flush=True)
