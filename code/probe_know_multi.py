"""Knowledge probe across vendors."""
import json,os,sys,time,argparse,subprocess,threading,pathlib
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import keys
ANS=["Yes, confers resistance","No, does not confer resistance","Unknown or insufficient evidence"]
SYS=("You are a precision oncology knowledge assistant. Answer a factual question about whether a "
     "specific molecular alteration is associated with resistance to a specific drug. Do not make "
     "a treatment recommendation. Report confidence as an integer 0-100.")
def q(r): return f"In {r.disease}, is the alteration {r.molecular_profile} associated with resistance to {r.therapy}?"
def _curl(a):
    return subprocess.run(['curl','-s','--max-time','120']+a,capture_output=True,text=True).stdout
def openai_(m,msg):
    sch={"type":"object","additionalProperties":False,"properties":{
      "resistance_known":{"type":"string","enum":ANS},"confidence":{"type":"integer"}},
      "required":["resistance_known","confidence"]}
    b={"model":m,"messages":[{"role":"system","content":SYS},{"role":"user","content":msg}],
       "response_format":{"type":"json_schema","json_schema":{"name":"k","strict":True,"schema":sch}},
       "max_completion_tokens":2000 if m.startswith(("o1","o3","o4","gpt-5")) else 200}
    if not m.startswith(("o1","o3","o4","gpt-5")): b["temperature"]=0
    o=_curl(['https://api.openai.com/v1/chat/completions','-H','Content-Type: application/json',
             '-H','Authorization: Bearer '+keys.openai(),'-d',json.dumps(b)])
    try: d=json.loads(o)
    except Exception: return {"error":"unparseable"}
    if 'error' in d: return {"error":str(d['error'].get('message'))[:120]}
    try: return json.loads(d['choices'][0]['message']['content'])
    except Exception: return {"error":"content parse"}
def anthropic_(m,msg):
    wid=(pathlib.Path.home()/'.anthropic_workspace').read_text().strip()
    tool={"name":"record","description":"Record the factual answer.","input_schema":{"type":"object",
      "properties":{"resistance_known":{"type":"string","enum":ANS},"confidence":{"type":"integer"}},
      "required":["resistance_known","confidence"],"additionalProperties":False}}
    b={"model":m,"max_tokens":1024,"system":SYS,"messages":[{"role":"user","content":msg}],
       "tools":[tool],"tool_choice":{"type":"tool","name":"record"}}
    o=_curl(['https://api.anthropic.com/v1/messages','-H','content-type: application/json',
             '-H','x-api-key: '+keys.anthropic(),'-H','anthropic-version: 2023-06-01',
             '-H','anthropic-workspace-id: '+wid,'-d',json.dumps(b)])
    try: d=json.loads(o)
    except Exception: return {"error":"unparseable"}
    if 'error' in d: return {"error":str(d['error'].get('message'))[:120]}
    for x in d.get('content',[]):
        if x.get('type')=='tool_use': return x['input']
    return {"error":"no tool_use"}
def gemini_(m,msg):
    sch={"type":"OBJECT","properties":{"resistance_known":{"type":"STRING","enum":ANS},
         "confidence":{"type":"INTEGER"}},"required":["resistance_known","confidence"]}
    b={"systemInstruction":{"parts":[{"text":SYS}]},"contents":[{"role":"user","parts":[{"text":msg}]}],
       "generationConfig":{"temperature":0,"responseMimeType":"application/json","responseSchema":sch,
                           "maxOutputTokens":2048}}
    o=_curl([f'https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={keys.gemini()}',
             '-H','Content-Type: application/json','-d',json.dumps(b)])
    try: d=json.loads(o)
    except Exception: return {"error":"unparseable"}
    if 'error' in d: return {"error":str(d['error'].get('message'))[:120]}
    try: return json.loads(d['candidates'][0]['content']['parts'][0]['text'])
    except Exception: return {"error":"gemini parse"}
def dispatch(m,msg):
    if m.startswith('claude'): return anthropic_(m,msg)
    if m.startswith('gemini'): return gemini_(m,msg)
    return openai_(m,msg)
_lock=threading.Lock()
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--workers',type=int,default=4)
    a=ap.parse_args()
    c=pd.read_csv('data/cohort_direction.csv'); c=c[c.label=='RESISTANT']
    out=f"raw_responses/know_{a.model.replace(':','-')}.jsonl"; done=set()
    if os.path.exists(out):
        for L in open(out):
            try:
                r=json.loads(L)
                if 'error' not in r.get('resp',{}): done.add(r['uid'])
            except Exception: pass
    jobs=[r for _,r in c.iterrows() if r.uid not in done]
    print(f"{a.model}: {len(jobs)} probes",flush=True)
    f=open(out,'a'); n=[0]
    def work(r):
        try: o=dispatch(a.model,q(r))
        except Exception as ex: o={"error":f"worker {type(ex).__name__}"}
        with _lock:
            f.write(json.dumps({"uid":r.uid,"model":a.model,"resp":o})+"\n"); f.flush(); n[0]+=1
            if n[0]%100==0: print(f"  {n[0]}/{len(jobs)}",flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex: list(ex.map(work,jobs))
    f.close(); print(f"{a.model}: done",flush=True)
