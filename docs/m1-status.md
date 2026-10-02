# M1 Status

Status snapshot: 2026-10-02.

M1 — Trusted Evaluation Data is **in progress**.

The first reproducible local corpus audit found:

- 51 catalogued FECABA documents;
- 5 pilot ground-truth files;
- 0 ground-truth files that currently satisfy the reviewed-and-complete contract;
- 0 known writers recorded in ground truth;
- deterministic pilot split: 2 train, 2 validation, 1 test;
- no document/writer split leakage detected;
- corpus identity/fingerprinting operational;
- reproducible baseline metrics not yet available because reviewed test ground truth does not yet exist.

## Current blocker

The next required work is human verification of the pilot ground truth against the source scoresheets. This must not be replaced by parser predictions, because M1 exists to create independent evidence for later model evaluation.

After review metadata and complete labels exist, rerun:

```powershell
python -m sumula_reader dataset-audit --dataset-root datasets/fecaba
```

Then generate predictions for every reviewed test document and run the baseline procedure described in [M1 — Trusted Evaluation Data](m1-evaluation.md).
