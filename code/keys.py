"""Runtime key loader. Reads from ~/.<vendor>_token; never copies into the repo."""
import os, pathlib
def _read(env, fn):
    v=os.environ.get(env)
    if v: return v.strip()
    p=pathlib.Path.home()/fn
    if p.exists():
        t=p.read_text().strip()
        if t: return t
    raise RuntimeError(f"no key: set {env} or write ~/{fn}")
def openai():    return _read('OPENAI_API_KEY','.openai_token')
def anthropic(): return _read('ANTHROPIC_API_KEY','.anthropic_token')
def gemini():    return _read('GEMINI_API_KEY','.gemini_token')
def oncokb():    return _read('ONCOKB_TOKEN','.oncokb_token')
