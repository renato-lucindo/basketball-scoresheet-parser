# M3 Status

Status snapshot: 2026-10-03.

M3 — Measured Reliability is **in progress**.

## Current baseline

The reproducible deterministic baseline is bound to M1 corpus `92f9e8d1124a2839b2029bc4e55d66309800629916c38e07d9031f2e4a726ba7` and contains 154 held-out decisions from the reviewed test document.

At threshold `1.0`:

- global accuracy: 13.0%;
- automation rate: 2.6%;
- review rate: 97.4%;
- accepted error rate: 100%;
- accepted decisions: 4;
- accepted errors: 4.

No global threshold satisfies both the maximum 1% accepted-error rate and the minimum 10% automation rate used by the baseline report.

## Component metrics

The report now publishes the same quality-gate decision and complete metric set for each evaluated field type:

- final score;
- individual-foul period;
- individual-foul symbol;
- period score;
- scoring jersey;
- scoring period;
- scoring points;
- team fouls.

No component currently has a threshold that satisfies the quality gate. Candidates without a passing threshold must remain in review or unresolved.

## Writer slices

Known-writer and unknown-writer metrics are reported separately. The current single test document belongs to the known-writer slice, so the unknown-writer slice contains zero decisions. This absence is explicit and cannot be presented as evidence of unknown-writer reliability.

## Remaining M3 work

- Generate an evaluated prediction set with the local handwriting models while preserving the deterministic baseline.
- Measure automatic roster and written-score candidates against reviewed labels.
- Derive component thresholds only where held-out evidence satisfies the accepted-error gate.
- Keep unsupported components in review when no passing threshold exists.
- Add end-to-end completeness and decision-state metrics alongside recognition decisions.

## Reproduction

```powershell
python -m sumula_reader dataset-predict-baseline `
  --dataset-root datasets/fecaba `
  --output baseline-predictions.jsonl `
  --dpi 300

python -m sumula_reader dataset-baseline baseline-predictions.jsonl `
  --dataset-root datasets/fecaba `
  --output-dir evaluation/fecaba-pilot-v1
```

The generated report is committed at `evaluation/fecaba-pilot-v1/baseline-metrics.json`.
