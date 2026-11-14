# From Mutation to Medication

Negin Ashrafi, Roxana Daneshjou. Stanford University.

## What the study measures

Precision oncology turns on the specific alteration, not the gene. A tumour carrying
*EGFR* L858R may respond to an EGFR inhibitor; one carrying *ABL1* T315I will not respond
to imatinib. This project measures how often a language model recommends a therapy for an
alteration that a curated knowledge base records as conferring **resistance** to it.

Each case is a tumour board vignette naming a cancer, an alteration and a proposed
therapy, and asking whether the therapy is clinically actionable for that alteration. The
prompt does not reveal the curated direction, the evidence level or the source.

A raw failure rate is not comparable across models, because a model that agrees with
everything will look bad for reasons unrelated to evidence. One open-weight model endorses
81.4 percent of resistance cases and 82.3 percent of drug sensitivity controls, which says
nothing about whether it reads the evidence. So every resistance case is contrasted with
sensitivity controls sharing its gene and its evidence level, and the reported quantity is
the difference between the two.

1,464 curated resistance cases, five knowledge bases, ten models, 58,910 judgments.

## Reproducing the paper

```bash
pip install pandas numpy scipy statsmodels
./reproduce.sh
```

The run takes a few minutes and needs no network access. It rebuilds every file in `results/` from
the raw model outputs and regenerates the paper's LaTeX tables into `tables/`. It then runs
`check_paper.py`, which asserts that every counted claim in the manuscript matches the result
files; the paper itself is published rather than distributed here, so put its `main.tex` in
`manuscript/` to run those checks.

The `results/` files in this repository are the ones the published paper was written from,
so diffing them after a run is a direct check that nothing has drifted. All 22 rebuild byte-identical: every bootstrap is
seeded, the estimators accumulate in a fixed order, and the one script that solves a regression
pins its thread count so the linear algebra reduces in the same order on any machine.

The `code/probe_*.py` scripts are the collection step and call paid vendor APIs. Nothing
above needs them, since every judgment is already in `raw_responses/`. They are included
because the prompts and the collection logic are part of the method. Credentials are read
from environment variables by `code/keys.py` and `code/keyloader.py`.

## Layout

```
reproduce.sh     rebuild everything and verify it
code/            the pipeline, 54 Python scripts and 3 shell scripts
data/            cohorts, knowledge-base sources, OncoKB annotations
raw_responses/   112 JSONL files; 80,132 records, of which the analysis keeps 58,910
results/         22 result CSVs, rebuilt by the analysis scripts
adjudication/    a 50-case clinician packet, and the entries the exclusion rules removed
PREREGISTRATION.md
```

## How the cases are built

Knowledge bases often state resistance conditionally, in prose rather than a structured
field. Flattening those statements produces cases that are wrong rather than hard: a naive
extraction marks ponatinib contraindicated for *ABL1* T315I, which is the mutation
ponatinib exists to treat, and olaparib contraindicated for *BRCA1* mutation, which is its
indication. The MOAlmanac entry behind the second reads "secondary mutations in *BRCA1*
may confer resistance to PARP inhibition".

Two rules drop these from the resistance arm: entries specified at gene level with no
particular alteration, and entries whose description carries a conditional qualifier such
as secondary, acquired, compound or reversion. They matter for MOAlmanac, which goes from
88 resistance cases to 20, and for CGI, 226 to 113. They barely touch CIViC, which is
built from specific molecular profiles: they flag none of its 418 sensitivity cases and
none of its 312 retained resistance cases.

`adjudication/excluded_resistance_entries.csv` lists all 949 flagged entries with the rule
that removed each, and the pre-exclusion cohorts are in `data/`, so the rules can be
re-applied from scratch.

## The matched statistics

Both are stratum-weighted risk differences with a cluster bootstrap over genes, which is
direct standardisation to the resistance-arm case distribution.

The level-matched version holds evidence level and source fixed, using cells with at least
five cases per arm. The gene-and-level matched version additionally holds the gene fixed,
so a model is credited only for separating directions within the same gene rather than for
recognising a familiar gene-drug pair. It uses cells with at least two cases per arm,
giving 28 cells over 17 of the 28 CIViC genes. Those cells hold 268 resistance cases, 85.9
percent of the arm: 11 singleton genes contribute no cell at all, and a further 28 cases sit in
(gene, level) cells where one arm holds fewer than two.

The two thresholds differ by design. `code/reviewer_analyses.py` holds the definitions.

## Pre-registration

`PREREGISTRATION.md` has the original, written before any model was run, with the gene
split into exploratory and confirmatory halves.

Amendment 1 changed the primary endpoint from endorsement, meaning Consider or Strongly
recommend, to unqualified Strongly recommend alone, on the recorded ground that Consider
is not an unambiguous error for a resistance variant since a clinician may still weigh a
therapy in some contexts. It was filed after the exploratory half was analysed and before
any confirmatory data was touched. The gene-matched estimand came later still, after the
confirmatory result had been seen, and is exploratory for that reason.

## Coverage

| Block | Judgments |
|---|---|
| Main arms | 36,590 |
| Self-conditioning controls | 3,818 |
| Control and robustness arms | 18,502 |
| Total | 58,910 |

`raw_responses/` holds 80,132 records in all. The analysis keeps the 58,910 above and drops the
rest: 5,733 failed API calls, the pre-exclusion cohort versions (`jax_`, `cgi_`, `moa_`), the pilot
files, and 27 surplus records in the `main_gpt-*` files that fall outside the final cohort.

Coverage is uneven, and which models a result rests on matters. The CIViC cohort covers
all nine main models; the knowledge probe, both repair arms and all three self-conditioning
controls cover the seven hosted models; CGI covers all seven across three vendors; JAX-CKB,
MOAlmanac, OncoKB and the paraphrase arms cover three. Running `code/make_tables.py` writes
the full breakdown to `tables/coverage.tex`, which is Table 2 of the paper.

`raw_responses/` also holds 12,626 valid judgments from earlier runs against pre-exclusion
cohorts (12,966 records, 340 of them failed calls), none of which are analysed. They are kept so the effect of the exclusion rules can
be measured directly, by comparing an arm against its own earlier version.

## Note on the data

`data/vicc_by_source.json`, the raw VICC harmonized meta-knowledgebase dump, is 360 MB and
is not included; regenerate it with `code/fetch_vicc.py` to rebuild the cross-source
cohorts from scratch. The JAX-CKB, CGI, PMKB and MolecularMatch cohorts derived from it are
all here.

## Citation

```bibtex
@inproceedings{ashrafi2026mutation,
  title     = {From Mutation to Medication: Characterizing Failures of Clinical
               {LLM}s to Respect Genomic Resistance},
  author    = {Ashrafi, Negin and Daneshjou, Roxana},
  booktitle = {Proceedings of the Machine Learning for Health (ML4H) Symposium},
  series    = {Proceedings of Machine Learning Research},
  year      = {2026}
}
```
