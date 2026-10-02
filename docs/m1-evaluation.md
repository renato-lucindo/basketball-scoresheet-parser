# M1 — Trusted Evaluation Data

M1 establishes the evidence base used by every later recognition and automation claim. The milestone must not be considered complete from model results alone: the real FECABA ground truth, corpus identity, and split isolation must be auditable first.

## Ground-truth review contract

Ground-truth JSON files live under the local `datasets/fecaba/ground_truth/` directory. A document is considered reviewed only when its core fields are complete and it contains explicit review metadata:

```json
{
  "review": {
    "status": "reviewed",
    "reviewed_at": "2026-10-02",
    "method": "manual"
  }
}
```

`reviewed` and `verified` are accepted final states. Generated templates start as `pending`. Removing a pending warning without adding explicit review metadata does not make a document reviewed.

The reviewer must verify the source scoresheet itself. Parser predictions must not be copied into ground truth without visual verification.

## Audit the corpus

Run:

```powershell
python -m sumula_reader dataset-audit --dataset-root datasets/fecaba
```

The command writes two local artifacts under `datasets/fecaba/evaluation/`:

- `evaluation-manifest.jsonl`: one record per ground-truth document, containing source and ground-truth hashes, anonymized writer ID when available, deterministic split, review state, and completeness issues;
- `m1-audit.json`: corpus fingerprint, split counts, isolation result, review progress, and M1 exit-criterion status.

The corpus SHA-256 changes when the evaluated ground truth or its review state changes, making a baseline traceable to the exact evidence used.

## Split policy

Splits are deterministic. A known writer is the grouping unit, so all documents attributed to the same writer stay in one split. If writer identity is unavailable, the document is the grouping unit. `dataset-audit` validates that neither a document nor a known writer leaks across splits.

## Produce baseline metrics

Predictions for reviewed test documents use JSONL with one recognition decision per line:

```json
{"document_id":"game-001","expected":"12","predicted":"12","confidence":0.97,"writer_known":true}
```

Then run:

```powershell
python -m sumula_reader dataset-baseline predictions.jsonl --dataset-root datasets/fecaba
```

The command selects an acceptance threshold using the project quality gate, records global accuracy, automation rate, accepted error rate, review count, and known/unknown-writer slices, and writes `baseline-metrics.json`. The baseline stores the corpus SHA-256, so a later audit detects stale metrics after ground-truth changes.

## M1 completion

M1 is ready only when `dataset-audit` reports all exit criteria as true. Until then, the audit is evidence of remaining work rather than evidence that the milestone has passed.
