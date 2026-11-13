#!/usr/bin/env bash
# Rebuild every result file and table from the raw model outputs. Run from the
# repository root. No network access needed.
set -e
cd "$(dirname "$0")"

python3 code/analyze_all.py          # headline, by-level, cross-source, knowledge, distribution
python3 code/analyze_reviewer2.py    # endpoints, outcome threshold, KB overlap, level counts, halves
python3 code/reviewer_analyses.py    # matched estimands, wild bootstrap, leave-one-gene-out
python3 code/analyze_selfcond.py     # self-conditioning arm
python3 code/analyze_controls.py     # false-attribution and no-instruction controls
python3 code/prereg_confirm.py       # pre-registered confirmatory test

python3 code/make_tables.py          # LaTeX tables into tables/
python3 code/make_tables_cr.py       # tables for the later arms

python3 code/check_paper.py          # cross-check the paper's prose, given manuscript/main.tex
