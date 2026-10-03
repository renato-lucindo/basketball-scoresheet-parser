# Human Review Workflow

This guide turns parser candidates into reviewed, reusable dataset records. The workflow uses Label Studio locally and rejects stale or contradictory exports before they can reach training data.

## Install review dependencies

```powershell
python -m pip install -e ".[vision,review]"
```

The examples assume an ingested dataset at `datasets/fecaba`. Raw sources, review media, and model checkpoints are intentionally excluded from Git.

## 1. Review document geometry

Prepare one Label Studio task per pilot document:

```powershell
scoresheet-parser dataset-review-prepare `
  --dataset-root datasets/fecaba `
  --stage geometry
```

The command writes:

- `datasets/fecaba/review/geometry.tasks.json`;
- `datasets/fecaba/review/geometry.labeling.xml`;
- normalized review images under `datasets/fecaba/review/media/geometry`.

Enable Label Studio local-file serving and start it with the review media directory as its document root:

```powershell
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=(Resolve-Path datasets/fecaba/review/media)
label-studio
```

Create a project, paste the generated XML into its labeling interface, and import the generated task JSON. After reviewing every region, export the project as Label Studio JSON and import it:

```powershell
scoresheet-parser dataset-review-import geometry-export.json `
  --dataset-root datasets/fecaba `
  --stage geometry
```

The importer binds every result to the source image hash and rejects duplicate documents, unknown documents, malformed regions, and source-version mismatches.

## 2. Build review crops

Build crops after geometry is reviewed:

```powershell
scoresheet-parser dataset-build `
  --dataset-root datasets/fecaba
```

This creates `datasets/fecaba/crops/manifest.jsonl` and crop images. Each crop carries stable identity, source-image hash, crop-image hash, split, field type, and parser context.

## 3. Review scoring, jerseys, and fouls

Prepare each stage independently:

```powershell
scoresheet-parser dataset-review-prepare --dataset-root datasets/fecaba --stage scoring
scoresheet-parser dataset-review-prepare --dataset-root datasets/fecaba --stage jerseys
scoresheet-parser dataset-review-prepare --dataset-root datasets/fecaba --stage fouls
```

For each stage, create or reuse a Label Studio project with the generated XML, import the task JSON, complete the decisions, export JSON, and import it back:

```powershell
scoresheet-parser dataset-review-import scoring-export.json --dataset-root datasets/fecaba --stage scoring
scoresheet-parser dataset-review-import jerseys-export.json --dataset-root datasets/fecaba --stage jerseys
scoresheet-parser dataset-review-import fouls-export.json --dataset-root datasets/fecaba --stage fouls
```

Review decisions have these meanings:

- `accepted`: the original crop and label are correct;
- `adjusted`: the reviewer changed the bounding box or final label;
- `erasure`: the cell contains a cancelled or erased mark;
- `illegible`: the source does not support a reliable label;
- `pending`: the task still needs a decision.

Accepted and adjusted records are written to `datasets/fecaba/crops/manifest.reviewed.jsonl` with the final label and reviewed bounding box. Training commands consume this reviewed manifest by default. Erasure, illegible, pending, stale, and unaudited records cannot become training labels.

## 4. Complete the second-review sample

Jersey and foul reviews assign a deterministic audit sample. Prepare and import the second review with `--audit-only`:

```powershell
scoresheet-parser dataset-review-prepare --dataset-root datasets/fecaba --stage jerseys --audit-only
scoresheet-parser dataset-review-import jerseys-audit-export.json --dataset-root datasets/fecaba --stage jerseys --audit-only

scoresheet-parser dataset-review-prepare --dataset-root datasets/fecaba --stage fouls --audit-only
scoresheet-parser dataset-review-import fouls-audit-export.json --dataset-root datasets/fecaba --stage fouls --audit-only
```

An audit result is recorded as passed or corrected. The dataset is not ready for training while the audit sample is incomplete.

## 5. Check readiness

```powershell
scoresheet-parser dataset-review-status --dataset-root datasets/fecaba
```

`ready_for_training` becomes true only when required stages have no pending or stale records, have trainable reviewed labels, pass the geometric-quality requirement, and complete their audit samples.

## Integrity rules

The importer fails before writing reusable data when it finds:

- a changed source-image hash;
- a changed or regenerated crop hash;
- duplicate tasks, crop IDs, documents, or regions;
- a crop from the wrong review stage;
- an unknown document or crop;
- invalid labels, decisions, dimensions, or bounding boxes;
- an item outside the deterministic audit sample.

Never edit hashes to bypass these checks. Regenerate the review tasks from the current dataset and review the changed evidence again.

## Measure the effect

After training or threshold changes, regenerate the held-out metrics described in [M3 Status](m3-status.md) and [M4 Status](m4-status.md). Record accepted error, automation, review, global accuracy, and candidate coverage together. A change may reduce review only when the accepted-error gate still passes on independent held-out evidence.
