"""Vendor-agnostic probe. OpenAI json_schema, Anthropic tool-use, Gemini responseSchema."""
import json,os,sys,time,argparse,subprocess,threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import keys, pathlib
TEMP=0
REC=["Strongly recommend","Consider","Insufficient evidence","Do not recommend"]
EVS=["Established clinical evidence","Moderate clinical evidence","Limited clinical evidence",
     "Preclinical only","No supporting evidence"]
SYSTEM=("You are assisting a molecular tumor board. Judge whether a proposed therapy is "
        "clinically actionable for the specific molecular alteration described. Base your judgment "
        "on the strength of evidence for THIS specific alteration, not the gene in general. "
        "Report confidence as an integer 0-100.")
def vignette(r,stmt=None):
    v=(f"A 62-year-old patient with metastatic {r.disease}. Tumor molecular profiling identifies: "
       f"{r.molecular_profile}. The tumor board is considering {r.therapy}.\n\n")
    if stmt: v+=f"Retrieved literature context:\n\"{str(stmt)[:900]}\"\n\n"
    return v+f"Is {r.therapy} clinically actionable for this patient's specific alteration?"
def _curl(args):
    r=subprocess.run(['curl','-s','--max-time','120']+args,capture_output=True,text=True)
    return r.stdout
# ---------- vendors ----------
def call_openai(model,msg):
    schema={"type":"object","additionalProperties":False,"properties":{
      "recommendation":{"type":"string","enum":REC},
      "evidence_strength":{"type":"string","enum":EVS},
      "confidence":{"type":"integer"}},"required":["recommendation","evidence_strength","confidence"]}
    body={"model":model,"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":msg}],
      "response_format":{"type":"json_schema","json_schema":{"name":"a","strict":True,"schema":schema}},
      "max_completion_tokens":(8000 if model.startswith("gpt-5") else 2000)
        if model.startswith(("o1","o3","o4","gpt-5")) else 200}  # gpt-5 spends more on reasoning tokens
    if not model.startswith(("o1","o3","o4","gpt-5")): body["temperature"]=TEMP
    out=_curl(['https://api.openai.com/v1/chat/completions','-H','Content-Type: application/json',
               '-H','Authorization: Bearer '+keys.openai(),'-d',json.dumps(body)])
    try: d=json.loads(out)
    except Exception: return {"error":"unparseable openai response"}
    if 'error' in d: return {"error":str(d['error'].get('message',''))[:150]}
    try: return json.loads(d['choices'][0]['message']['content'])
    except Exception: return {"error":"openai content parse"}
def call_anthropic(model,msg):
    _w=pathlib.Path.home()/'.anthropic_workspace'
    wid=_w.read_text().strip() if _w.exists() else None
    tool={"name":"record","description":"Record the tumour board judgement.","input_schema":{
      "type":"object","properties":{
        "recommendation":{"type":"string","enum":REC},
        "evidence_strength":{"type":"string","enum":EVS},
        "confidence":{"type":"integer"}},
      "required":["recommendation","evidence_strength","confidence"],"additionalProperties":False}}
    body={"model":model,"max_tokens":1024,"system":SYSTEM,
          "messages":[{"role":"user","content":msg}],
          "tools":[tool],"tool_choice":{"type":"tool","name":"record"}}
    out=_curl(['https://api.anthropic.com/v1/messages','-H','content-type: application/json',
               '-H','x-api-key: '+keys.anthropic(),'-H','anthropic-version: 2023-06-01']
              +(['-H','anthropic-workspace-id: '+wid] if wid else [])
              +['-d',json.dumps(body)])
    try: d=json.loads(out)
    except Exception: return {"error":"unparseable anthropic response"}
    if 'error' in d: return {"error":str(d['error'].get('message'))[:150]}
    for b in d.get('content',[]):
        if b.get('type')=='tool_use': return b['input']
    return {"error":"no tool_use block"}
def call_gemini(model,msg):
    schema={"type":"OBJECT","properties":{
      "recommendation":{"type":"STRING","enum":REC},
      "evidence_strength":{"type":"STRING","enum":EVS},
      "confidence":{"type":"INTEGER"}},
      "required":["recommendation","evidence_strength","confidence"]}
    body={"systemInstruction":{"parts":[{"text":SYSTEM}]},
          "contents":[{"role":"user","parts":[{"text":msg}]}],
          "generationConfig":{"temperature":0,"responseMimeType":"application/json",
                              "responseSchema":schema,"maxOutputTokens":2048}}
    url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={keys.gemini()}"
    out=_curl([url,'-H','Content-Type: application/json','-d',json.dumps(body)])
    try: d=json.loads(out)
    except Exception: return {"error":"unparseable gemini response"}
    if 'error' in d: return {"error":str(d['error'].get('message'))[:150]}
    try: return json.loads(d['candidates'][0]['content']['parts'][0]['text'])
    except Exception: return {"error":f"gemini parse: {str(d)[:100]}"}
def dispatch(model,msg):
    if model.startswith('claude'): return call_anthropic(model,msg)
    if model.startswith('gemini'): return call_gemini(model,msg)
    return call_openai(model,msg)
_lock=threading.Lock()
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True); ap.add_argument('--cohort',default='data/cohort_direction.csv')
    ap.add_argument('--tag',default='dir'); ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--rag',action='store_true'); ap.add_argument('--workers',type=int,default=4)
    a=ap.parse_args()
    c=pd.read_csv(a.cohort if not a.rag else 'data/cohort_direction_rag.csv')
    if a.rag: c=c[c.stmt.notna()]
    if a.limit: c=c.head(a.limit)
    out=f"raw_responses/{a.tag}_{a.model.replace(':','-')}.jsonl"; done=set()
    if os.path.exists(out):
        for L in open(out):
            try:
                r=json.loads(L)
                if 'error' not in r.get('resp',{}): done.add(r['uid'])
            except Exception: pass
    jobs=[r for _,r in c.iterrows() if r.uid not in done]
    print(f"{a.model} [{a.tag}]: {len(jobs)} calls",flush=True)
    f=open(out,'a'); n=[0]
    def work(r):
        try: o=dispatch(a.model, vignette(r, r.stmt if a.rag else None))
        except Exception as ex: o={"error":f"worker: {type(ex).__name__}"}
        with _lock:
            f.write(json.dumps({"uid":r.uid,"model":a.model,"resp":o})+"\n"); f.flush(); n[0]+=1
            if n[0]%100==0: print(f"  {n[0]}/{len(jobs)}",flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex: list(ex.map(work,jobs))
    f.close(); print(f"{a.model}: done",flush=True)
