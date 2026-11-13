"""Knowledge probe: does the model KNOW the resistance finding, asked directly and
separately from any recommendation? Separates a knowledge gap from a weighting failure."""
import json,os,sys,time,argparse,subprocess,threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from keyloader import get_key
KEY=get_key()
SCHEMA={"type":"object","additionalProperties":False,"properties":{
 "resistance_known":{"type":"string","enum":["Yes, confers resistance","No, does not confer resistance","Unknown or insufficient evidence"]},
 "confidence":{"type":"integer"}},"required":["resistance_known","confidence"]}
SYSTEM=("You are a precision oncology knowledge assistant. Answer a factual question about whether a "
        "specific molecular alteration is associated with resistance to a specific drug. Do not make "
        "a treatment recommendation. Report confidence as an integer 0-100.")
def q(r): return (f"In {r.disease}, is the alteration {r.molecular_profile} associated with "
                  f"resistance to {r.therapy}?")
_lock=threading.Lock()
def call(model,msg):
    body={"model":model,"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":msg}],
          "response_format":{"type":"json_schema","json_schema":{"name":"k","strict":True,"schema":SCHEMA}},
          "max_completion_tokens":2000 if model.startswith(("o1","o3","o4","gpt-5")) else 200}
    if not model.startswith(("o1","o3","o4","gpt-5")): body["temperature"]=0
    for a in range(5):
        p=subprocess.run(['curl','-s','--max-time','90','https://api.openai.com/v1/chat/completions',
            '-H','Content-Type: application/json','-H','Authorization: Bearer '+KEY,'-d',json.dumps(body)],
            capture_output=True,text=True)
        try:
            d=json.loads(p.stdout)
            if 'error' in d:
                if 'rate' in d['error'].get('message','').lower(): time.sleep(5*(a+1)); continue
                return {"error":d['error'].get('message','')[:150]}
            return json.loads(d['choices'][0]['message']['content'])
        except Exception: time.sleep(2*(a+1))
    return {"error":"failed"}
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); a=ap.parse_args()
    c=pd.read_csv('data/cohort_direction.csv'); c=c[c.label=='RESISTANT']
    out=f"raw_responses/know_{a.model.replace(':','-')}.jsonl"; done=set()
    if os.path.exists(out):
        for L in open(out):
            try:
                r=json.loads(L)
                if 'error' not in r.get('resp',{}): done.add(r['uid'])
            except Exception: pass
    jobs=[r for _,r in c.iterrows() if r.uid not in done]
    print(f"{a.model}: {len(jobs)} knowledge probes",flush=True)
    f=open(out,'a'); n=[0]
    def work(r):
        o=call(a.model,q(r))
        with _lock:
            f.write(json.dumps({"uid":r.uid,"model":a.model,"resp":o})+"\n"); f.flush(); n[0]+=1
            if n[0]%100==0: print(f"  {n[0]}/{len(jobs)}",flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex: list(ex.map(work,jobs))
    f.close(); print(f"{a.model}: done",flush=True)
