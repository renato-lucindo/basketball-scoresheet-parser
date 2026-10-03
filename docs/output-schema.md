# Structured Output Contract

The parser returns a JSON-serializable `DocumentResult`. Schema version `0.1` is the active pre-v1.0 contract.

## Decision states

Every decision that can be uncertain uses one of three states:

- `accepted`: the parser has enough evidence to use the value automatically;
- `review`: a value exists, but validation or recognition evidence requires human review;
- `unresolved`: the required value is unavailable or cannot be determined safely.

The document status reflects the most severe core-field state. An unresolved core field makes the document unresolved. Otherwise, a review field or validation warning makes the document require review.

## Top-level result

```json
{
  "schema_version": "0.1",
  "metadata": {},
  "teams": {
    "A": {},
    "B": {}
  },
  "core_fields": [],
  "status": "accepted",
  "warnings": []
}
```

`metadata` identifies the template, game, competition, category, date, and available writer profile. `teams` contains the extracted game record. `warnings` contains reconciliation findings intended for review.

## Team result

Each team contains:

- `side` and `name`;
- `players`, including participation, starter state, fouls, and calculated scoring totals;
- `periods`, with calculated and written scores kept separately;
- `scoring_events`, including running score, jersey, shot type, points, confidence, and decision status;
- `team_fouls` for each period;
- `calculated_score` and `written_final_score` kept separately;
- team-level `status` and `warnings`.

The parser never overwrites a written score with a calculated value. Reconciliation compares both values and reports a contradiction for review.

## Core-field inventory

`core_fields` makes M2 completeness machine-readable. It contains one entry for every required field on both teams:

```json
{
  "path": "teams.A.final_score",
  "status": "unresolved",
  "reason": "written final score was not extracted"
}
```

The required paths cover team name, players, participation, starters, period scoring, scoring events, individual fouls, team fouls, and final score. Consumers should inspect this inventory before treating a result as complete.

## CLI output

Run the parser with a handwriting model:

```powershell
scoresheet-parser analyze game.pdf `
  --handwriting-model-dir models/handwriting `
  --output result.json
```

Roster recognition preserves a row-level observation for every occupied row. The current M1 evidence does not justify an automatic acceptance threshold, so model candidates remain in review and do not become players. Invalid and duplicate predictions are unresolved. `--roster-a` and `--roster-b` provide reviewed overrides.

Until automatic header and written-score extraction is complete, callers may also provide `--team-a-name`, `--team-b-name`, `--period-scores-a`, `--period-scores-b`, `--final-score-a`, and `--final-score-b`. These values enter the normal reconciliation path and do not bypass review checks.

Use `scoresheet-parser schema` to print a complete serialized skeleton, including all unresolved core fields for an empty document.
