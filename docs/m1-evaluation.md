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

The evaluation split is frozen in the same file after the writer identity is reviewed:

```json
{"evaluation_split":"test"}
```

Valid values are `train`, `validation`, and `test`. Before this value is frozen, the tooling may derive a provisional split. A release-quality corpus must use explicit splits so adding a previously unknown writer cannot silently move a document between partitions.

`reviewed` and `verified` are accepted final states. Generated templates start as `pending`. Removing a pending warning without adding explicit review metadata does not make a document reviewed.

The reviewer must verify the source scoresheet itself. Parser predictions must not be copied into ground truth without visual verification.

Each team also carries an explicit `individual_fouls_reviewed` boolean. An empty `individual_fouls` object is conclusive only when that value is `true`; otherwise it means review is still pending. Cell-level entries under `individual_foul_observations` preserve the source slot and use one of three review states:

- `verified`: the symbol and its single period are confirmed and may be used as a training label;
- `unresolved`: the source was reviewed but does not uniquely determine the value; a reason is required and the cell remains excluded from training;
- `pending`: review work remains and the document cannot be considered complete.

This distinction allows reviewed ground truth to represent irreducible source ambiguity without converting it into a guessed label.

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

Explicit ground-truth splits are authoritative and stable. A known writer is the grouping unit, so every document attributed to that writer must use the same explicit split. If an explicit split is not yet available, the audit derives a provisional deterministic split from the writer or document. `dataset-audit` validates that neither a document nor a known writer leaks across splits.

## Produce baseline metrics

Generate predictions from the deterministic parser, without local handwriting-model artifacts:

```powershell
python -m sumula_reader dataset-predict-baseline --dataset-root datasets/fecaba
```

The command evaluates reviewed, complete test documents and writes one recognition decision per line. Each record has a stable field identity:

```json
{"document_id":"game-001","field_id":"scoring:A:1:jersey","field_type":"scoring_jersey","expected":"12","predicted":"12","confidence":0.97,"writer_known":true}
```

Then run:

```powershell
python -m sumula_reader dataset-baseline datasets/fecaba/evaluation/baseline-predictions.jsonl --dataset-root datasets/fecaba
```

The first command covers scoring points, periods, jersey numbers, period and final scores, team-foul boxes, and verified individual-foul cells. Explicitly unresolved ground-truth cells are excluded. The second command selects an acceptance threshold using the project quality gate, records global accuracy, automation rate, accepted error rate, review count, field-type counts, and known/unknown-writer slices, and writes `baseline-metrics.json`. The baseline stores the corpus SHA-256, so a later audit detects stale metrics after ground-truth changes.

## M1 completion

M1 is ready only when `dataset-audit` reports all exit criteria as true. Until then, the audit is evidence of remaining work rather than evidence that the milestone has passed.
