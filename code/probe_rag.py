"""Mitigation arm: identical vignette plus the retrieved CIViC evidence statement."""
import json,os,sys,time,argparse,subprocess,threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from keyloader import get_key
KEY=get_key()
SCHEMA={"type":"object","additionalProperties":False,"properties":{
 "recommendation":{"type":"string","enum":["Strongly recommend","Consider","Insufficient evidence","Do not recommend"]},
 "evidence_strength":{"type":"string","enum":["Established clinical evidence","Moderate clinical evidence","Limited clinical evidence","Preclinical only","No supporting evidence"]},
 "confidence":{"type":"integer"}},"required":["recommendation","evidence_strength","confidence"]}
SYSTEM=("You are assisting a molecular tumor board. Judge whether a proposed therapy is "
        "clinically actionable for the specific molecular alteration described. Base your judgment "
        "on the strength of evidence for THIS specific alteration, not the gene in general. "
        "Report confidence as an integer 0-100.")
def vig(r):
    return (f"A 62-year-old patient with metastatic {r.disease}. Tumor molecular profiling "
            f"identifies: {r.molecular_profile}. The tumor board is considering {r.therapy}.\n\n"
            f"Retrieved literature context:\n\"{str(r.stmt)[:900]}\"\n\n"
            f"Is {r.therapy} clinically actionable for this patient's specific alteration?")
_lock=threading.Lock()
def call(model,msg):
    body={"model":model,"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":msg}],
          "response_format":{"type":"json_schema","json_schema":{"name":"a","strict":True,"schema":SCHEMA}},
          "max_completion_tokens":2000 if model.startswith(("o1","o3","o4","gpt-5")) else 200}
    if not model.startswith(("o1","o3","o4","gpt-5")): body["temperature"]=0
    for a in range(5):
        p=subprocess.run(['curl','-s','--max-time','120','https://api.openai.com/v1/chat/completions',
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
    c=pd.read_csv('data/cohort_direction_rag.csv'); c=c[c.stmt.notna()]
    out=f"raw_responses/rag_{a.model.replace(':','-')}.jsonl"; done=set()
    if os.path.exists(out):
        for L in open(out):
            try:
                r=json.loads(L)
                if 'error' not in r.get('resp',{}): done.add(r['uid'])
            except Exception: pass
    jobs=[r for _,r in c.iterrows() if r.uid not in done]
    print(f"{a.model}: {len(jobs)} RAG calls",flush=True)
    f=open(out,'a'); n=[0]
    def work(r):
        o=call(a.model,vig(r))
        with _lock:
            f.write(json.dumps({"uid":r.uid,"model":a.model,"resp":o})+"\n"); f.flush(); n[0]+=1
            if n[0]%200==0: print(f"  {n[0]}/{len(jobs)}",flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex: list(ex.map(work,jobs))
    f.close(); print(f"{a.model}: done",flush=True)
