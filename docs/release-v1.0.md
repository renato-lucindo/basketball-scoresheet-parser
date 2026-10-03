# FECABA v1.0 Release Record

Release candidate date: 2026-10-03.

Package version: `1.0.0`.

Structured output schema: `1.0`.

## Supported contract

The release accepts single-page FECABA scoresheets as PDF, PNG, JPG, or JPEG through the public Python API or `scoresheet-parser analyze`. It emits a JSON-serializable `DocumentResult` covering teams, players, participation, starters, scoring by period, scoring events, individual fouls, team fouls, final scores, warnings, and a complete core-field state inventory.

Automatic roster and written-score recognizers produce candidates. Reviewed CLI/API context can provide accepted overrides. Missing, ambiguous, contradictory, or uncalibrated observations remain reviewable or unresolved.

## Evaluation identity

- Corpus SHA-256: `92f9e8d1124a2839b2029bc4e55d66309800629916c38e07d9031f2e4a726ba7`
- Reviewed ground-truth documents: 5
- Split definition: 2 train, 2 validation, 1 test
- Document leakage: none
- Writer leakage: none

## Held-out quality policy

No tested global threshold satisfies both the maximum 1% accepted-error rate and the report's 10% minimum automation target. The release policy is `review_all`, represented by an effective threshold above every valid confidence value.

On the one-document test slice with handwriting models:

- decisions: 186;
- global accuracy: 11.3%;
- candidate coverage: 64.5%;
- automation rate: 0%;
- review rate: 100%;
- accepted error rate: 0%;
- accepted decisions: 0.

On the three-document validation-plus-test generalization slice:

- decisions: 612;
- writers absent from training: 2;
- capture conditions: camera image and PDF scan;
- global accuracy: 13.2%;
- candidate coverage: 54.1%;
- automation rate: 0%;
- review rate: 100%;
- accepted error rate: 0%;
- accepted decisions: 0.

## Evidence

- [M1 trusted evaluation data](m1-status.md)
- [M2 complete parser](m2-status.md)
- [M3 measured reliability](m3-status.md)
- [M4 real-world generalization](m4-status.md)
- [M5 efficient review](m5-status.md)
- [Held-out deterministic metrics](../evaluation/fecaba-pilot-v1/baseline-metrics.json)
- [Held-out model metrics](../evaluation/fecaba-pilot-v1/handwriting-model-metrics.json)
- [Generalization deterministic metrics](../evaluation/fecaba-generalization-v1/baseline-metrics.json)
- [Generalization model metrics](../evaluation/fecaba-generalization-v1/handwriting-model-metrics.json)
- [Review-efficiency history](../evaluation/review-efficiency.json)

Model-backed reports embed SHA-256 fingerprints for `jersey.pt` and `foul.pt`.

## Reproduction gates

From a clean checkout:

```powershell
python -m pip install -e ".[dev,vision,ml,jev]"
python -m pytest -q
```

Dataset, review, and evaluation reproduction requires the private source corpus and follows [M1 Evaluation](m1-evaluation.md) and [Human Review Workflow](review-workflow.md). Public audit and metric artifacts preserve corpus and model identities without publishing personal or source-document data.

## Known limitations

- Automatic recognition quality is too low for silent acceptance.
- Camera images perform worse than PDF scans in the current evidence.
- Roster, scoring-jersey, written-score, and individual-foul recognition require substantial improvement.
- The evaluation corpus is small and does not justify claims beyond the documented FECABA conditions.
- Multi-page and non-FECABA forms are unsupported.
