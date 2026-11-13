#!/bin/bash
cd "$(dirname "$0")"
for m in o4-mini gpt-4.1 gpt-4o gpt-4o-mini claude-opus-5 claude-haiku-4-5 gemini-2.5-flash; do
  python3 probe_selfcond.py --model "$m" --workers 6 2>&1 | grep -v Warning | grep -v "from pandas"
done
echo "ALL SELF-CONDITIONING RUNS COMPLETE"
