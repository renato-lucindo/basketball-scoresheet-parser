# M1 Status

Status snapshot: 2026-10-03.

M1 — Trusted Evaluation Data is **complete**. The repository audit reports `ready`, and every exit criterion is satisfied for corpus `92f9e8d1124a2839b2029bc4e55d66309800629916c38e07d9031f2e4a726ba7`.

The first reproducible local corpus audit found:

- 51 catalogued FECABA documents;
- 5 pilot ground-truth files;
- 5 reviewed pilot ground-truth files with rosters, scoring events, period totals, final score, writer identity, team-foul boxes, and individual-foul cells visually reviewed;
- scoring events reconciled against the written period and final totals for every pilot;
- an explicit individual-foul review state that distinguishes a verified empty result from unfinished review;
- a cell-level observation area that separates verified labels, irreducible unresolved values, and pending review;
- 14 test-pilot individual-foul labels verified from ink color, cell order, and team-foul constraints; source-ambiguous cells remain explicit unresolved observations and are excluded from training labels;
- all 5 pilots are in `reviewed` state, with no pending core-field review;
- 4 anonymized writer groups recorded in the audit manifest;
- deterministic pilot split: 2 train, 2 validation, 1 test;
- no document/writer split leakage detected;
- corpus identity/fingerprinting operational;
- a reproducible 154-decision baseline bound to the completed corpus;
- baseline global accuracy of 13.0%, automation rate of 2.6%, and accepted error rate of 100% at threshold 1.0; no threshold currently satisfies the project quality gate.

## Exit evidence

The public audit snapshot records:

- `reviewed_ground_truth`: passed;
- `corpus_identified`: passed;
- `document_and_writer_isolation`: passed;
- `baseline_metrics_reproducible`: passed.

Reproduce the snapshot with:

```powershell
python -m sumula_reader dataset-predict-baseline --dataset-root datasets/fecaba --output baseline-predictions.jsonl --dpi 300
python -m sumula_reader dataset-baseline baseline-predictions.jsonl --dataset-root datasets/fecaba --output-dir evaluation/fecaba-pilot-v1
python -m sumula_reader dataset-audit --dataset-root datasets/fecaba --output-dir evaluation/fecaba-pilot-v1
```

The low baseline result is evidence for M2 work, rather than a completion gate for M1. M2 now owns the end-to-end FECABA parser and the explicit propagation of unavailable, review, and unresolved fields. See [Project Goals](project-goals.md).
