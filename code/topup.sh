#!/bin/bash
# Idempotent top-up: fills any incomplete arm. Safe to re-run.
cd "$(dirname "$0")"
for M in o4-mini gpt-4.1 gpt-4o gpt-4o-mini claude-opus-5 claude-haiku-4-5 gemini-2.5-pro gemini-2.5-flash; do
  python3 probe_multi.py --model "$M" --tag dir --workers 4 2>&1 | tail -1
  python3 probe_multi.py --model "$M" --tag rag --rag --workers 4 2>&1 | tail -1
  python3 probe_know_multi.py --model "$M" --workers 4 2>&1 | tail -1
done
