# M4 Status

Status snapshot: 2026-10-03.

M4 — Real-world Generalization is **complete**.

## Evaluation scope

The generalization report evaluates the reviewed validation and test splits from M1. It covers three full real-world scoresheets, 612 core-field decisions, two writers absent from the training split, and two capture conditions:

- one camera image with 212 decisions;
- two PDF scans with 400 decisions.

The corpus remains identified by SHA-256 `92f9e8d1124a2839b2029bc4e55d66309800629916c38e07d9031f2e4a726ba7`. Document and writer isolation remain unchanged.

## Generalization results

The deterministic parser reaches 12.3% global accuracy and 31.0% candidate coverage. The handwriting-model pipeline reaches 13.2% global accuracy and 54.1% candidate coverage.

Model-backed results by capture condition:

| Capture condition | Decisions | Global accuracy | Candidate coverage | Accepted error | Automation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Camera image | 212 | 7.1% | 32.1% | 0% | 0% |
| PDF scan | 400 | 16.5% | 65.8% | 0% | 0% |

All 612 decisions belong to writers absent from the training split. This unseen-writer slice is reported separately and has 13.2% global accuracy with the handwriting models.

## Failure modes

The evidence identifies these generalization failures:

- camera-image candidate coverage is substantially lower than PDF-scan coverage;
- roster and scoring jersey recognition produces many candidates but very few correct values;
- written score recognition remains unreliable;
- individual-foul symbols and periods are mostly unresolved;
- calculated scores inherit missing or incorrect scoring-event observations.

The global acceptance policy remains `review_all`. Every emitted candidate is routed to review and every absent candidate is explicit as unresolved. No observed failure is silently accepted.

One scoring-period component threshold appears viable on the combined slice, but it was selected and measured on the same limited evidence. It is not promoted to the production policy. An independent calibration/evaluation cycle is required before automatic acceptance.

## Quality gate

Both deterministic and model-backed generalization reports have:

- accepted error rate: 0%;
- automation rate: 0%;
- review rate: 100%;
- accepted decisions: 0;
- accepted errors: 0.

The M3 accepted-error gate therefore continues to hold for camera images, PDF scans, and unseen writers.

## Evidence

- [`evaluation/fecaba-generalization-v1/baseline-metrics.json`](../evaluation/fecaba-generalization-v1/baseline-metrics.json)
- [`evaluation/fecaba-generalization-v1/handwriting-model-metrics.json`](../evaluation/fecaba-generalization-v1/handwriting-model-metrics.json)

The model-backed report embeds the checkpoint SHA-256 fingerprints.

## Reproduction

```powershell
python -m sumula_reader dataset-predict-baseline `
  --dataset-root datasets/fecaba `
  --output generalization-predictions.jsonl `
  --evaluation-splits validation test `
  --handwriting-model-dir models/handwriting

python -m sumula_reader dataset-baseline generalization-predictions.jsonl `
  --dataset-root datasets/fecaba `
  --output-dir evaluation/fecaba-generalization-v1 `
  --evaluation-splits validation test
```

## Exit-criterion evidence

- Evaluation uses full real-world camera and PDF inputs rather than development crops.
- Writers absent from training are identified and reported separately.
- Observed failure modes are documented and routed to review or unresolved.
- The M3 accepted-error gate holds across the defined generalization slice.
