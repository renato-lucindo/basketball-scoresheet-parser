# M2 Status

Status snapshot: 2026-10-03.

M2 — Complete FECABA Parser is **in progress**.

## Current pipeline

The public `analyze` command already accepts a FECABA image or PDF, normalizes it, extracts participation marks, scoring events, individual fouls, and team-foul indicators, reconciles derived totals, and serializes a `DocumentResult`.

The current command still requires both team rosters as input. Team names, written period scores, and written final scores can be supplied as explicit context while automatic extraction is developed. Handwritten jersey and foul recognition also requires optional local model files.

## Core-field contract

Every analysis result now includes `core_fields`, a machine-readable inventory for the M2 record. Each required field has an `accepted`, `review`, or `unresolved` status and an explanation when it is not accepted. Missing teams and unimplemented extraction paths therefore cannot appear as a clean accepted result.

The inventory covers, for each team:

- team name;
- player roster;
- participation;
- starters;
- period scoring;
- scoring events;
- individual fouls;
- team fouls;
- final score.

An unresolved core field propagates to the team and document status. Contradictory fields remain in review.

The `analyze` command accepts the optional context arguments `--team-a-name`, `--team-b-name`, `--period-scores-a`, `--period-scores-b`, `--final-score-a`, and `--final-score-b`. Period-score values contain exactly four comma-separated non-negative integers. Supplied values are reconciled against extracted scoring events and do not bypass review status when they disagree.

## Remaining M2 work

- Extract team names and rosters so the public command can run without undocumented manual transcription.
- Extract written period and final scores and reconcile them with scoring events.
- Define behavior when optional handwriting models are absent and keep every unavailable recognition result explicit.
- Add a representative end-to-end image/PDF fixture that asserts the complete JSON structure.
- Document the stable output contract and supported input expectations.

## Validation

Run the full suite from a clean development environment:

```powershell
python -m pytest -q
```

The initial M2 core-field slice passes 101 tests.
