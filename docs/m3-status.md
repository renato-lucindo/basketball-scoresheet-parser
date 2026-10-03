# M3 Status

Status snapshot: 2026-10-03.

M3 — Measured Reliability is **complete**.

## Evaluation scope

Both reports are bound to M1 corpus `92f9e8d1124a2839b2029bc4e55d66309800629916c38e07d9031f2e4a726ba7`. The held-out slice currently contains one reviewed known-writer document and 186 core-field decisions across 11 field types:

- roster jersey;
- scoring jersey, points, and period;
- individual-foul symbol and period;
- calculated and written period scores;
- calculated and written final scores;
- team fouls.

The report records global and per-component accepted error rate, automation rate, review rate, global accuracy, candidate coverage, and accepted/review/unresolved state counts. Known-writer and unknown-writer slices remain separate.

## Derived acceptance policy

No tested global or component threshold satisfies both the maximum 1% accepted-error rate and the report's minimum 10% automation rate. The evidence-derived policy is therefore `review_all`: the effective threshold is above the maximum confidence value, so no candidate can be accepted silently.

This policy produces:

- accepted error rate: 0%;
- automation rate: 0%;
- review rate: 100%;
- accepted decisions: 0;
- accepted errors: 0.

The safe fallback fixes an earlier reporting defect where a missing threshold was represented as `1.0`; predictions with confidence exactly `1.0` could then appear accepted despite failing the quality gate.

## Deterministic baseline

The deterministic parser has 10.8% global accuracy and 35.5% candidate coverage. It emits 66 review candidates and leaves 120 decisions unresolved.

The committed report is [`evaluation/fecaba-pilot-v1/baseline-metrics.json`](../evaluation/fecaba-pilot-v1/baseline-metrics.json).

## Handwriting-model baseline

The local handwriting models have 11.3% global accuracy and 64.5% candidate coverage. They emit 120 review candidates and leave 66 decisions unresolved. No model-backed candidate is automatically accepted.

The report embeds SHA-256 fingerprints for `jersey.pt` and `foul.pt`, so results cannot be confused with a different checkpoint. The committed report is [`evaluation/fecaba-pilot-v1/handwriting-model-metrics.json`](../evaluation/fecaba-pilot-v1/handwriting-model-metrics.json).

## Writer slices

The held-out document belongs to the known-writer slice. The unknown-writer slice contains zero decisions, which is reported explicitly and provides no unknown-writer reliability evidence. Closing that evidence gap belongs to M4.

## Reproduction

Generate and evaluate the deterministic baseline:

```powershell
python -m sumula_reader dataset-predict-baseline `
  --dataset-root datasets/fecaba `
  --output baseline-predictions.jsonl `
  --dpi 300

python -m sumula_reader dataset-baseline baseline-predictions.jsonl `
  --dataset-root datasets/fecaba `
  --output-dir evaluation/fecaba-pilot-v1
```

Generate model-backed predictions by adding `--handwriting-model-dir models/handwriting`. Model-backed evidence is valid only when the checkpoint hashes in the generated report match the recorded hashes.

## Exit-criterion evidence

- Component and end-to-end metrics come from the reviewed M1 held-out set.
- Threshold selection is computed from the held-out predictions and quality gate.
- The fallback policy accepts no unsupported decision and therefore keeps accepted error at 0%.
- Accepted error rate, automation rate, global accuracy, review rate, coverage, and decision states are reported together.

M4 must add supported real-world capture conditions and unseen-writer evidence while preserving this safety policy.
