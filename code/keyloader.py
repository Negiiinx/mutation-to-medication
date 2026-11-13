"""Read the OpenAI key at runtime. Reads from the environment only.
Deliberately makes no second copy of the secret on disk."""
import os, re, pathlib
_FALLBACK = pathlib.Path(os.environ.get("OPENAI_KEY_FILE", ""))
def get_key():
    k = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
    if k: return k.strip()
    if _FALLBACK.exists():
        m = re.search(r'OPENAI_KEY\s*=\s*["\']([^"\']+)["\']', _FALLBACK.read_text())
        if m: return m.group(1)
    raise RuntimeError("No OpenAI key found (set OPENAI_API_KEY).")
