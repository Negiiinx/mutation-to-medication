#!/bin/bash
cd "$(dirname "$0")"
M="o4-mini gpt-4.1 gpt-4o gpt-4o-mini claude-opus-5 claude-haiku-4-5 gemini-2.5-flash"
echo "=== FALSE-SELF ARM ==="
for m in $M; do python3 probe_selfcond.py --model "$m" --mode false --tag falseself --workers 6 2>&1 | grep -Ev "Warning|from pandas"; done
echo "=== NO-INSTRUCTION ARM ==="
for m in $M; do python3 probe_selfcond.py --model "$m" --mode noinstr --tag noinstr --workers 6 2>&1 | grep -Ev "Warning|from pandas"; done
echo "ALL CONTROL ARMS COMPLETE"
