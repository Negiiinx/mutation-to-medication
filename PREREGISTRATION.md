# Pre-registration — direction discrimination
Written 2026-08-31, BEFORE any model was run on this cohort.

## Question
For gene-drug pairs where CIViC contains BOTH sensitivity and resistance variants, do LLMs
endorse the drug less for the resistance variants than for the sensitivity variants?

## Design
Within gene+drug pair. Gene identity, drug identity and gene prominence are held constant by
construction; only the variant and its curated direction change.

## Primary endpoint
Endorsement rate (recommendation of "Consider" or "Strongly recommend") for RESISTANT variants
minus the same for SENSITIVE variants, within gene-drug pair. Under correct behaviour this gap
should be strongly negative.

## Primary test
Linear probability model, endorse ~ label, with gene-drug pair fixed effects, cluster-robust SEs
by gene. Confirmatory analysis is run ONLY on the CONFIRM split.

## Splits (fixed here, seed 20260831, split by gene so no gene appears in both)
EXPLORE genes: ['ABL1', 'CDKN2A', 'EGFR', 'ERBB4', 'KIT', 'MET', 'MTOR', 'NTRK1', 'NTRK3', 'PIK3CA', 'RB1', 'RET', 'ROS1', 'TP53']
CONFIRM genes: ['ABCB1', 'ALK', 'BRAF', 'BRCA1', 'ERBB2', 'FLT3', 'KRAS', 'MAP2K1', 'NQO1', 'PDGFRA', 'PTEN', 'RARA', 'RRM1', 'TYMS']

## Declared exploratory (not confirmatory) analyses
- interaction with CIViC evidence level of the resistance finding
- interaction with gene prominence (NOTE: a prior prominence hypothesis on a different cohort
  failed held-out replication; treated as exploratory only)

---

## AMENDMENT 1 — 2026-08-31, after EXPLORE analysis, BEFORE any CONFIRM analysis

Change of primary endpoint.

Original primary endpoint was endorsement, defined as "Consider" or "Strongly recommend".
On inspecting the EXPLORE half only, "Consider" is judged not to be an unambiguous error for a
resistance variant: a clinician may still legitimately weigh a therapy in some contexts.

New primary endpoint: rate of "Strongly recommend" on a variant curated as RESISTANT.
This is unambiguously incorrect regardless of clinical context.

Broad endorsement is demoted to a secondary endpoint and will still be reported.

CONFIRM half remains untouched as of this amendment. No CONFIRM data has been analysed.
Reason for the change is interpretive (what counts as an error), not driven by CONFIRM results,
which have not been seen.
