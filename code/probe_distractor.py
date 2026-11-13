"""Distractor control for the oracle-evidence arm.

The mitigation arm supplies the curated resistance statement and endorsement falls. That may
be the content, or it may be that any inserted evidence block induces caution. This arm keeps
the block and swaps its content: for each resistance case it supplies a *sensitising* CIViC
statement for the same gene instead of the resistance statement it is scored against.

Usage: python probe_distractor.py --model gpt-4o [--workers 6]
"""
import json, os, sys, argparse, threading
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_multi import dispatch, vignette

_lock = threading.Lock()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tag', default='distract')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    c = pd.read_csv('data/cohort_direction_rag.csv')
    c = c[c.stmt.notna()]
    res = c[c.label == 'RESISTANT'].copy()
    sens = c[c.label == 'SENSITIVE']
    # a sensitising statement from the same gene, deterministically chosen
    by_gene = {g: s.sort_values('uid').stmt.tolist() for g, s in sens.groupby('gene')}
    res['stmt'] = [by_gene.get(g, [None])[0] for g in res.gene]
    res = res[res.stmt.notna()]
    if a.limit:
        res = res.head(a.limit)

    out = f"raw_responses/{a.tag}_{a.model.replace(':', '-')}.jsonl"
    done = set()
    if os.path.exists(out):
        for L in open(out):
            try:
                r = json.loads(L)
                if 'error' not in r.get('resp', {}):
                    done.add(r['uid'])
            except Exception:
                pass
    jobs = [r for _, r in res.iterrows() if r.uid not in done]
    print(f"{a.model} [{a.tag}]: {len(jobs)} calls", flush=True)
    f = open(out, 'a'); n = [0]

    def work(r):
        try:
            o = dispatch(a.model, vignette(r, r.stmt))
        except Exception as ex:
            o = {"error": f"worker: {type(ex).__name__}"}
        with _lock:
            f.write(json.dumps({"uid": r.uid, "model": a.model, "resp": o}) + "\n")
            f.flush(); n[0] += 1
            if n[0] % 100 == 0:
                print(f"  {n[0]}/{len(jobs)}", flush=True)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, jobs))
    f.close(); print(f"{a.model}: done", flush=True)
