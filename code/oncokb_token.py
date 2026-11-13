"""Read the OncoKB token at runtime. Never copied into the repo."""
import os, pathlib
CANDIDATES=[os.environ.get('ONCOKB_TOKEN'),
            pathlib.Path.home()/'.oncokb_token',
            pathlib.Path.home()/'.config'/'oncokb'/'token']
def get_token():
    v=CANDIDATES[0]
    if v: return v.strip()
    for p in CANDIDATES[1:]:
        p=pathlib.Path(p)
        if p.exists():
            t=p.read_text().strip()
            if t: return t
    raise RuntimeError("No OncoKB token found. Write it to ~/.oncokb_token")
