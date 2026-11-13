import json, subprocess, time, collections
BASE="https://search.cancervariants.org/api/v1/associations"
def get(u):
    for i in range(4):
        r=subprocess.run(['curl','-sL','--max-time','120',u],capture_output=True,text=True)
        if r.returncode==0 and r.stdout.strip().startswith('{'):
            try: return json.loads(r.stdout)
            except Exception: pass
        time.sleep(3*(i+1))
    return None
out={}
for src in ['oncokb','cgi','jax','pmkb','molecularmatch','civic']:
    hits=[]; frm=0
    while True:
        d=get(f"{BASE}?size=500&from={frm}&q=source:{src}")
        if not d or 'hits' not in d: break
        h=d['hits']; lst=h['hits'] if isinstance(h,dict) else h
        tot=h.get('total') if isinstance(h,dict) else None
        if not lst: break
        hits+=lst; frm+=len(lst)
        if tot and frm>=min(tot,9500): break
        time.sleep(0.25)
    hits=[x for x in hits if x.get('source')==src]
    out[src]=hits
    rt=collections.Counter(x.get('association',{}).get('response_type') for x in hits)
    print(f"{src:16s} {len(hits):6d} assoc | response types: {rt.most_common(6)}",flush=True)
json.dump(out,open('data/vicc_by_source.json','w'))
print("saved data/vicc_by_source.json")
