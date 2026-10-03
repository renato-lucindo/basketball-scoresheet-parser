# M1 Status

Status snapshot: 2026-10-03.

M1 — Trusted Evaluation Data is **in progress**.

The first reproducible local corpus audit found:

- 51 catalogued FECABA documents;
- 5 pilot ground-truth files;
- 5 pilot ground-truth files with rosters, period totals, final score, and writer identity visually transcribed;
- 1 pilot (the test document) with scoring events visually transcribed and period totals reconciled;
- an explicit individual-foul review state that distinguishes a verified empty result from unfinished review;
- a cell-level observation area that separates verified labels, irreducible unresolved values, and pending review;
- 14 test-pilot individual-foul labels verified from ink color, cell order, and team-foul constraints; 22 source-ambiguous cells are explicitly unresolved and excluded from training;
- the test pilot is reviewed; the remaining 4 pilots stay in `partial` review state while scoring and foul cells are verified;
- 4 anonymized writer groups recorded in the audit manifest;
- deterministic pilot split: 2 train, 2 validation, 1 test;
- no document/writer split leakage detected;
- corpus identity/fingerprinting operational;
- a reproducible 154-decision baseline bound to corpus `ffc6b5f68cfa4f8b4bd0fba34c9dd7a35bb01947a616c17f589f1123590c9c25`;
- baseline global accuracy of 13.0%, automation rate of 2.6%, and accepted error rate of 100% at threshold 1.0; no threshold currently satisfies the project quality gate.

## Current blocker

The next required work is scoring and foul verification for the remaining four pilots. Parser predictions may guide crop navigation but must not replace visual source review, because M1 exists to create independent evidence for later model evaluation.

After review metadata and complete labels exist, rerun:

```powershell
python -m sumula_reader dataset-audit --dataset-root datasets/fecaba
```

Then generate predictions for every reviewed test document and run the baseline procedure described in [M1 — Trusted Evaluation Data](m1-evaluation.md).
