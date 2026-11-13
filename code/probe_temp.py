"""Variability: k samples at temperature 1 on the CIViC resistance arm.

All headline runs are temperature 0, one sample per case, and hosted endpoints are not
strictly deterministic even then. This measures how much a per-case judgement moves.

Usage: python probe_temp.py --model gpt-4o --k 3
"""
import json, os, sys, argparse, threading
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_multi as PM

_lock = threading.Lock()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--label', default='RESISTANT')
    a = ap.parse_args()

    # temperature 1 for the sampling check
    _orig = PM.call_openai
    def hot(model, msg):
        return _orig(model, msg)
    PM.TEMP = 1.0

    c = pd.read_csv('data/cohort_direction.csv')
    c = c[c.label == a.label]
    suf = "" if a.label == "RESISTANT" else "_sens"
    out = f"raw_responses/temp{suf}_{a.model.replace(':', '-')}.jsonl"
    done = {}
    if os.path.exists(out):
        for L in open(out):
            try:
                r = json.loads(L)
                if 'error' not in r.get('resp', {}):
                    done[(r['uid'], r['rep'])] = 1
            except Exception:
                pass
    jobs = [(r, k) for _, r in c.iterrows() for k in range(a.k) if (r.uid, k) not in done]
    print(f"{a.model} [temp k={a.k}]: {len(jobs)} calls", flush=True)
    f = open(out, 'a'); n = [0]

    def work(t):
        r, k = t
        try:
            o = PM.dispatch(a.model, PM.vignette(r, None))
        except Exception as ex:
            o = {"error": f"worker: {type(ex).__name__}"}
        with _lock:
            f.write(json.dumps({"uid": r.uid, "model": a.model, "rep": k, "resp": o}) + "\n")
            f.flush(); n[0] += 1
            if n[0] % 200 == 0:
                print(f"  {n[0]}/{len(jobs)}", flush=True)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, jobs))
    f.close(); print(f"{a.model}: done", flush=True)
