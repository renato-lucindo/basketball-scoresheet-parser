# M2 Status

Status snapshot: 2026-10-03.

M2 — Complete FECABA Parser is **complete**.

## Current pipeline

The public `analyze` command already accepts a FECABA image or PDF, normalizes it, extracts participation marks, scoring events, individual fouls, and team-foul indicators, reconciles derived totals, and serializes a `DocumentResult`.

The command can recognize occupied roster rows with an optional handwriting model. Because M1 found no acceptance threshold that satisfies the quality gate, automatic jersey candidates remain in review and do not silently become players. Invalid and duplicate rows are unresolved. Callers can supply reviewed rosters as overrides. Team names, written period scores, and written final scores can also be supplied as explicit context while automatic extraction is developed.

The same safety rule now applies to the eight written regular-period score cells and both final-score cells. The parser crops and recognizes each cell, preserves the numeric candidate and confidence, and leaves it in review until a calibrated threshold satisfies the project quality gate. Reviewed period and final-score CLI values remain accepted overrides and are reconciled against scoring events.

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

## Exit evidence

- A FECABA image or PDF can run through the public CLI/API without required roster arguments or undocumented steps. Missing models or context produce explicit unresolved fields.
- The structured result accounts for all 18 required team-level core fields through accepted, review, or unresolved entries.
- Automatic roster and written-score candidates remain reviewable and separate from accepted reviewed overrides.
- Contradictions and missing evidence propagate through field, team, and document status.
- A generated representative image fixture exercises `analyze_path`, the real extraction pipeline, reconciliation, completeness assessment, and JSON serialization.

Automatic team-name recognition and acceptance-threshold calibration remain product improvements. They do not block M2 because the output contract explicitly represents unavailable fields; threshold measurement and calibration belong to M3.

## Validation

Run the full suite from a clean development environment:

```powershell
python -m pytest -q
```

The generated representative FECABA fixture now exercises the real normalized-image pipeline from participation and scoring marks through reconciliation, core-field assessment, and JSON serialization. It produces an accepted result only when all 18 core-field entries are accepted.

The current M2 branch passes 116 tests plus 3 parameterized subtests.
