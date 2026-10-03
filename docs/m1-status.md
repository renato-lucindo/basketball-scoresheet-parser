# M1 Status

Status snapshot: 2026-10-03.

M1 — Trusted Evaluation Data is **in progress**.

The first reproducible local corpus audit found:

- 51 catalogued FECABA documents;
- 5 pilot ground-truth files;
- 5 pilot ground-truth files with rosters, period totals, final score, and writer identity visually transcribed;
- 1 pilot (the test document) with scoring events visually transcribed and period totals reconciled;
- an explicit individual-foul review state that distinguishes a verified empty result from unfinished review;
- a pending-observation area that preserves legible foul marks without admitting them as training labels;
- 14 test-pilot individual-foul periods uniquely determined by ink color, cell order, and team-foul constraints; 10 labels from fully resolved rows are recorded, while 25 observations remain pending;
- all 5 pilots remain in `partial` review state while individual foul cells and the remaining scoring events are verified;
- 4 anonymized writer groups recorded in the audit manifest;
- deterministic pilot split: 2 train, 2 validation, 1 test;
- no document/writer split leakage detected;
- corpus identity/fingerprinting operational;
- reproducible baseline metrics not yet available because reviewed test ground truth does not yet exist.

## Current blocker

The next required work is cell-by-cell verification of individual fouls for the test pilot, followed by scoring and foul verification for the remaining four pilots. This must not be replaced by parser predictions, because M1 exists to create independent evidence for later model evaluation.

After review metadata and complete labels exist, rerun:

```powershell
python -m sumula_reader dataset-audit --dataset-root datasets/fecaba
```

Then generate predictions for every reviewed test document and run the baseline procedure described in [M1 — Trusted Evaluation Data](m1-evaluation.md).
