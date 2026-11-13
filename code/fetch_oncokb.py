"""Pull the full OncoKB actionable-variant set, including resistance levels R1/R2."""
import json, subprocess, sys, os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from oncokb_token import get_token
TOK=get_token(); B="https://www.oncokb.org/api/v1"
def get(ep):
    r=subprocess.run(['curl','-s','--max-time','120',f"{B}/{ep}",
        '-H',f'Authorization: Bearer {TOK}'],capture_output=True,text=True)
    try: return json.loads(r.stdout)
    except Exception: return {'_raw':r.stdout[:300]}
d=get('utils/allActionableVariants')
if isinstance(d,dict) and ('_raw' in d or 'status' in d):
    print("FAILED:", str(d)[:300]); sys.exit(1)
json.dump(d,open('data/oncokb_actionable.json','w'))
import collections
lv=collections.Counter(x.get('level') for x in d)
print(f"OncoKB actionable variants: {len(d)}")
print("levels:", lv.most_common())
res=[x for x in d if str(x.get('level','')).upper().startswith('R')]
print(f"RESISTANCE entries (R1/R2): {len(res)}")
print(f"  distinct genes: {len({x.get('gene') for x in res})}")
for x in res[:8]:
    print(f"   {str(x.get('gene')):8s} {str(x.get('alteration'))[:22]:22s} "
          f"{str(x.get('cancerType'))[:26]:26s} {str(x.get('drugs'))[:34]:34s} {x.get('level')}")
