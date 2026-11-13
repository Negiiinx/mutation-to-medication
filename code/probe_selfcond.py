"""Self-conditioning arm: does a model act on a resistance conclusion it just produced itself?

The knowledge probe and the recommendation are separate calls, so the recommendation is not
literally conditioned on the model having stated the resistance finding. This arm closes that
gap: for each case we take the model's OWN stored knowledge-probe answer, quote it back to it,
and then ask for the recommendation.

That gives a three-condition comparison on identical cases:
  (a) no prior conclusion      -> raw_responses/dir_*.jsonl   (already collected)
  (b) the model's own conclusion -> this arm, raw_responses/self_*.jsonl
  (c) the curated statement    -> raw_responses/rag_*.jsonl   (already collected)

Usage: python probe_selfcond.py --model claude-opus-5 [--workers 4] [--limit N]
"""
import json, os, sys, argparse, threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_multi import dispatch, SYSTEM

_lock = threading.Lock()


def own_answers(model):
    """The model's own knowledge-probe answers, keyed by uid."""
    path = f"raw_responses/know_{model.replace(':', '-')}.jsonl"
    if not os.path.exists(path):
        return {}
    out = {}
    for L in open(path):
        try:
            r = json.loads(L)
        except Exception:
            continue
        resp = r.get('resp', {})
        if 'error' in resp or 'resistance_known' not in resp:
            continue
        # first occurrence wins, matching anlib.load's drop_duplicates, so this arm and the
        # knowledge-probe table bucket every case identically even where a uid was probed twice
        out.setdefault(r['uid'], resp['resistance_known'])
    return out


YES = 'Yes, confers resistance'
NO = 'No, does not confer resistance'

DEPART = (" If you depart from your earlier conclusion, your judgement should reflect the "
          "evidence for this specific alteration.")


def vignette_instr_only(r):
    """T1 plus the trailing instruction, with no conclusion quoted."""
    return (f"A 62-year-old patient with metastatic {r.disease}. Tumor molecular profiling "
            f"identifies: {r.molecular_profile}. The tumor board is considering {r.therapy}.\n\n"
            f"Is {r.therapy} clinically actionable for this patient's specific alteration?"
            + DEPART.replace(" If you depart from your earlier conclusion, your judgement should",
                             " Your judgement should"))


def vignette(r, prior, instruct=True):
    return (f"A 62-year-old patient with metastatic {r.disease}. Tumor molecular profiling "
            f"identifies: {r.molecular_profile}. The tumor board is considering {r.therapy}.\n\n"
            f"Earlier you were asked whether {r.molecular_profile} is associated with resistance "
            f"to {r.therapy} in {r.disease}. You concluded: \"{prior}\"\n\n"
            f"Taking your own earlier conclusion into account, is {r.therapy} clinically "
            f"actionable for this patient's specific alteration?" + (DEPART if instruct else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--cohort', default='data/cohort_direction.csv')
    ap.add_argument('--tag', default='self')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--mode', default='self', choices=['self', 'false', 'noinstr', 'instronly'],
                    help="self: quote the model's own answer. "
                         "false: on cases where it said resistance, quote the opposite instead. "
                         "noinstr: as self, without the departure instruction.")
    a = ap.parse_args()

    prior = own_answers(a.model)
    if not prior:
        print(f"{a.model}: no knowledge-probe answers on disk, skipping", flush=True)
        sys.exit(0)

    c = pd.read_csv(a.cohort)
    c = c[c.label == 'RESISTANT']              # the arm where acting on the finding matters
    c = c[c.uid.isin(prior)]
    if a.mode == 'false':
        # only cases the model itself called resistance; it is then told the opposite
        c = c[c.uid.map(lambda u: prior.get(u)) == YES]
    if a.limit:
        c = c.head(a.limit)

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
    jobs = [r for _, r in c.iterrows() if r.uid not in done]
    print(f"{a.model} [{a.tag}]: {len(jobs)} calls", flush=True)

    f = open(out, 'a')
    n = [0]

    def work(r):
        shown = NO if a.mode == 'false' else prior[r.uid]
        try:
            if a.mode == 'instronly':
                shown = None
                o = dispatch(a.model, vignette_instr_only(r))
            else:
                o = dispatch(a.model, vignette(r, shown, instruct=(a.mode != 'noinstr')))
        except Exception as ex:
            o = {"error": f"worker: {type(ex).__name__}"}
        with _lock:
            f.write(json.dumps({"uid": r.uid, "model": a.model, "mode": a.mode,
                                "prior": prior[r.uid], "shown": shown, "resp": o}) + "\n")
            f.flush()
            n[0] += 1
            if n[0] % 100 == 0:
                print(f"  {n[0]}/{len(jobs)}", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, jobs))
    f.close()
    print(f"{a.model}: done", flush=True)
