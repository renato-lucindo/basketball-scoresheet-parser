from __future__ import annotations

from dataclasses import asdict
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from .dataset import assign_document_split, validate_split_isolation
from .evaluation import (
    PredictionRecord,
    choose_acceptance_threshold,
    compare_known_unknown_writers,
    evaluate_threshold,
)


REVIEWED_STATUSES = {"reviewed", "verified"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        records.append(item)
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _anonymous_writer_id(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    digest = hashlib.sha256(f"writer:{value.strip().casefold()}".encode()).hexdigest()
    return f"writer_{digest[:12]}"


def _review_status(payload: dict[str, Any]) -> str:
    review = payload.get("review")
    if isinstance(review, dict):
        status = review.get("status")
        if isinstance(status, str) and status.strip():
            return status.strip().lower()
    warnings = payload.get("warnings")
    if isinstance(warnings, list) and any(
        isinstance(item, str) and "pendente" in item.casefold() for item in warnings
    ):
        return "pending"
    return "unverified"


def _individual_foul_observation_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    teams = payload.get("teams")
    if not isinstance(teams, dict):
        return {}
    for side in ("A", "B"):
        team = teams.get(side)
        if not isinstance(team, dict):
            continue
        observations = team.get("individual_foul_observations")
        if not isinstance(observations, dict):
            continue
        for entries in observations.values():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, dict):
                    counts[str(entry.get("review_state", "pending"))] += 1
    return dict(sorted(counts.items()))


def _ground_truth_issues(
    payload: dict[str, Any], catalog_record: dict[str, Any]
) -> list[str]:
    issues: list[str] = []
    document_id = catalog_record.get("document_id")
    if payload.get("game_id") != document_id:
        issues.append("game_id_mismatch")
    if payload.get("source_file") != catalog_record.get("source_path"):
        issues.append("source_file_mismatch")

    review_status = _review_status(payload)
    review = payload.get("review")
    if review_status in REVIEWED_STATUSES:
        if not isinstance(review, dict):
            issues.append("review_metadata_missing")
        else:
            if not isinstance(review.get("reviewed_at"), str) or not review["reviewed_at"].strip():
                issues.append("reviewed_at_missing")
            if not isinstance(review.get("method"), str) or not review["method"].strip():
                issues.append("review_method_missing")
    warnings = payload.get("warnings")
    if isinstance(warnings, list) and any(
        isinstance(item, str) and "pendente" in item.casefold() for item in warnings
    ):
        issues.append("pending_review_warning_present")

    teams = payload.get("teams")
    if not isinstance(teams, dict):
        return issues + ["teams_missing"]
    for side in ("A", "B"):
        team = teams.get(side)
        if not isinstance(team, dict):
            issues.append(f"team_{side}_missing")
            continue
        roster = team.get("roster")
        if not isinstance(roster, list) or not roster:
            issues.append(f"team_{side}_roster_missing")
        team_fouls = team.get("team_fouls")
        if not isinstance(team_fouls, dict) or any(
            not isinstance(team_fouls.get(f"Q{period}"), int)
            for period in range(1, 5)
        ):
            issues.append(f"team_{side}_team_fouls_incomplete")
        if team.get("individual_fouls_reviewed") is not True:
            issues.append(f"team_{side}_individual_fouls_unreviewed")
        observations = team.get("individual_foul_observations", {})
        if not isinstance(observations, dict):
            issues.append(f"team_{side}_individual_foul_observations_invalid")
        else:
            states = [
                entry.get("review_state", "pending")
                for entries in observations.values()
                if isinstance(entries, list)
                for entry in entries
                if isinstance(entry, dict)
            ]
            if "pending" in states:
                issues.append(f"team_{side}_individual_foul_observations_pending")

    period_scores = payload.get("period_scores")
    for side in ("A", "B"):
        side_scores = period_scores.get(side) if isinstance(period_scores, dict) else None
        if not isinstance(side_scores, dict) or any(
            not isinstance(side_scores.get(f"Q{period}"), int) for period in range(1, 5)
        ):
            issues.append(f"team_{side}_period_scores_incomplete")

    final_score = payload.get("final_score")
    if not isinstance(final_score, dict) or any(
        not isinstance(final_score.get(side), int) for side in ("A", "B")
    ):
        issues.append("final_score_incomplete")

    scoring = payload.get("scoring")
    if not isinstance(scoring, list):
        issues.append("scoring_missing")
        scoring = []

    if isinstance(final_score, dict) and any(
        isinstance(final_score.get(side), int) and final_score[side] > 0
        for side in ("A", "B")
    ) and not scoring:
        issues.append("scoring_empty")

    if isinstance(period_scores, dict) and isinstance(final_score, dict):
        for side in ("A", "B"):
            side_scores = period_scores.get(side)
            final = final_score.get(side)
            if isinstance(side_scores, dict) and all(
                isinstance(side_scores.get(f"Q{period}"), int)
                for period in range(1, 5)
            ) and isinstance(final, int):
                if sum(side_scores[f"Q{period}"] for period in range(1, 5)) != final:
                    issues.append(f"team_{side}_period_total_mismatch")

    scoring_totals: dict[tuple[str, str], int] = {}
    for event in scoring:
        if not isinstance(event, dict):
            continue
        team = event.get("team")
        period = event.get("period")
        points = event.get("points")
        if team in {"A", "B"} and period in {"Q1", "Q2", "Q3", "Q4"} and isinstance(points, int):
            key = (team, period)
            scoring_totals[key] = scoring_totals.get(key, 0) + points
    if scoring and isinstance(period_scores, dict):
        for side in ("A", "B"):
            side_scores = period_scores.get(side)
            if not isinstance(side_scores, dict):
                continue
            for period in range(1, 5):
                period_name = f"Q{period}"
                written = side_scores.get(period_name)
                if isinstance(written, int) and scoring_totals.get((side, period_name), 0) != written:
                    issues.append(f"team_{side}_{period_name}_scoring_mismatch")
    return issues


def audit_fecaba_dataset(
    dataset_root: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(dataset_root)
    catalog_path = root / "catalog.jsonl"
    if not catalog_path.exists():
        raise FileNotFoundError(f"Dataset catalog not found: {catalog_path}")

    catalog = _read_jsonl(catalog_path)
    by_document: dict[str, dict[str, Any]] = {}
    duplicate_documents: list[str] = []
    for record in catalog:
        document_id = record.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("Every catalog record must have a document_id")
        if document_id in by_document:
            duplicate_documents.append(document_id)
        by_document[document_id] = record

    evaluation_records: list[dict[str, Any]] = []
    ground_truth_root = root / "ground_truth"
    orphan_ground_truth: list[str] = []
    for path in sorted(ground_truth_root.glob("*.json")):
        document_id = path.stem
        catalog_record = by_document.get(document_id)
        if catalog_record is None:
            orphan_ground_truth.append(document_id)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Ground truth must be a JSON object: {path}")
        writer_id = _anonymous_writer_id(payload.get("writer_name"))
        review_status = _review_status(payload)
        issues = _ground_truth_issues(payload, catalog_record)
        explicit_split = payload.get("evaluation_split")
        if explicit_split not in {None, "train", "validation", "test"}:
            issues.append("evaluation_split_invalid")
            explicit_split = None
        evaluation_records.append(
            {
                "document_id": document_id,
                "source_sha256": catalog_record.get("sha256"),
                "ground_truth_sha256": _sha256_file(path),
                "writer_id": writer_id,
                "split": explicit_split
                or assign_document_split(document_id, writer_id=writer_id),
                "split_source": "explicit" if explicit_split else "derived",
                "review_status": review_status,
                "individual_foul_observation_counts": (
                    _individual_foul_observation_counts(payload)
                ),
                "ground_truth_complete": not issues,
                "issues": issues,
            }
        )

    isolation_error: str | None = None
    try:
        validate_split_isolation(evaluation_records)
    except ValueError as exc:
        isolation_error = str(exc)

    canonical = json.dumps(
        evaluation_records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    corpus_sha256 = hashlib.sha256(canonical).hexdigest()

    reviewed = [
        item
        for item in evaluation_records
        if item["review_status"] in REVIEWED_STATUSES
        and item["ground_truth_complete"]
    ]
    known_writers = {item["writer_id"] for item in evaluation_records if item["writer_id"]}
    review_status_counts = dict(
        sorted(Counter(item["review_status"] for item in evaluation_records).items())
    )
    split_counts = {name: 0 for name in ("train", "validation", "test")}
    for item in evaluation_records:
        split_counts[item["split"]] += 1

    artifacts = Path(output_dir) if output_dir is not None else root / "evaluation"
    baseline_path = artifacts / "baseline-metrics.json"
    baseline_current = False
    if baseline_path.exists():
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            baseline_current = baseline.get("corpus_sha256") == corpus_sha256
        except (OSError, json.JSONDecodeError, AttributeError):
            baseline_current = False

    exit_criteria = {
        "reviewed_ground_truth": bool(evaluation_records)
        and len(reviewed) == len(evaluation_records),
        "corpus_identified": bool(evaluation_records) and not duplicate_documents,
        "document_and_writer_isolation": isolation_error is None,
        "baseline_metrics_reproducible": baseline_current,
    }
    report: dict[str, Any] = {
        "milestone": "M1",
        "status": "ready" if all(exit_criteria.values()) else "in_progress",
        "corpus_sha256": corpus_sha256,
        "catalog_documents": len(catalog),
        "ground_truth_documents": len(evaluation_records),
        "reviewed_ground_truth_documents": len(reviewed),
        "known_writers": len(known_writers),
        "review_status_counts": review_status_counts,
        "split_counts": split_counts,
        "duplicate_documents": sorted(set(duplicate_documents)),
        "orphan_ground_truth": orphan_ground_truth,
        "isolation_error": isolation_error,
        "exit_criteria": exit_criteria,
    }

    artifacts.mkdir(parents=True, exist_ok=True)
    manifest_path = artifacts / "evaluation-manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for item in evaluation_records:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    (artifacts / "m1-audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_baseline_metrics(
    dataset_root: str | Path,
    predictions: str | Path,
    *,
    output_dir: str | Path | None = None,
    max_accepted_error_rate: float = 0.01,
    min_automation_rate: float = 0.10,
) -> dict[str, Any]:
    root = Path(dataset_root)
    artifacts = Path(output_dir) if output_dir is not None else root / "evaluation"
    audit = audit_fecaba_dataset(root, output_dir=artifacts)
    manifest = _read_jsonl(artifacts / "evaluation-manifest.jsonl")
    eligible_test_documents = {
        item["document_id"]
        for item in manifest
        if item.get("split") == "test"
        and item.get("review_status") in REVIEWED_STATUSES
        and item.get("ground_truth_complete") is True
    }
    if not eligible_test_documents:
        raise ValueError("No reviewed and complete test documents are available")

    rows = _read_jsonl(Path(predictions))
    records: list[PredictionRecord] = []
    records_by_field_type: dict[str, list[PredictionRecord]] = {}
    field_ids: set[str] = set()
    field_type_counts: Counter[str] = Counter()
    for row in rows:
        document_id = row.get("document_id")
        if document_id not in eligible_test_documents:
            raise ValueError(
                f"Prediction references a document outside the reviewed test set: {document_id}"
            )
        expected = row.get("expected")
        predicted = row.get("predicted")
        confidence = row.get("confidence")
        if not isinstance(expected, str) or not isinstance(confidence, (int, float)):
            raise ValueError("Predictions require string expected and numeric confidence")
        if predicted is not None and not isinstance(predicted, str):
            raise ValueError("predicted must be a string or null")
        field_id = row.get("field_id")
        if not isinstance(field_id, str) or not field_id.strip():
            raise ValueError("Predictions require a non-empty field_id")
        unique_field_id = f"{document_id}:{field_id}"
        if unique_field_id in field_ids:
            raise ValueError(f"Duplicate prediction field_id: {unique_field_id}")
        field_ids.add(unique_field_id)
        field_type = row.get("field_type")
        if not isinstance(field_type, str) or not field_type.strip():
            raise ValueError("Predictions require a non-empty field_type")
        field_type_counts[field_type] += 1
        record = PredictionRecord(
                expected=expected,
                predicted=predicted,
                confidence=float(confidence),
                document_id=document_id,
                writer_known=bool(row.get("writer_known", False)),
            )
        records.append(record)
        records_by_field_type.setdefault(field_type, []).append(record)
    if not records:
        raise ValueError("Prediction file is empty")

    covered_documents = {record.document_id for record in records}
    missing_documents = sorted(eligible_test_documents - covered_documents)
    if missing_documents:
        raise ValueError(
            "Predictions do not cover every reviewed test document: "
            + ", ".join(missing_documents)
        )

    selected = choose_acceptance_threshold(
        records,
        max_accepted_error_rate=max_accepted_error_rate,
        min_automation_rate=min_automation_rate,
    )
    threshold = selected.threshold if selected is not None else 1.0
    metrics = evaluate_threshold(records, threshold)
    writer_metrics = compare_known_unknown_writers(records, threshold)
    field_type_metrics: dict[str, dict[str, object]] = {}
    for field_type, field_records in sorted(records_by_field_type.items()):
        field_selected = choose_acceptance_threshold(
            field_records,
            max_accepted_error_rate=max_accepted_error_rate,
            min_automation_rate=min_automation_rate,
        )
        field_threshold = (
            field_selected.threshold if field_selected is not None else 1.0
        )
        field_type_metrics[field_type] = {
            "quality_gate": {
                "max_accepted_error_rate": max_accepted_error_rate,
                "min_automation_rate": min_automation_rate,
                "threshold_found": field_selected is not None,
            },
            "metrics": asdict(evaluate_threshold(field_records, field_threshold)),
        }
    result = {
        "corpus_sha256": audit["corpus_sha256"],
        "prediction_records": len(records),
        "field_type_counts": dict(sorted(field_type_counts.items())),
        "test_documents": sorted(covered_documents),
        "eligible_test_documents": sorted(eligible_test_documents),
        "quality_gate": {
            "max_accepted_error_rate": max_accepted_error_rate,
            "min_automation_rate": min_automation_rate,
            "threshold_found": selected is not None,
        },
        "metrics": asdict(metrics),
        "writer_metrics": {
            key: asdict(value) for key, value in writer_metrics.items()
        },
        "field_type_metrics": field_type_metrics,
    }
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "baseline-metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _prediction_row(
    *,
    document_id: str,
    field_id: str,
    field_type: str,
    expected: object,
    predicted: object,
    confidence: float | None,
    writer_known: bool,
) -> dict[str, object]:
    return {
        "document_id": document_id,
        "field_id": field_id,
        "field_type": field_type,
        "expected": str(expected),
        "predicted": None if predicted is None else str(predicted),
        "confidence": max(0.0, min(1.0, float(confidence or 0.0))),
        "writer_known": writer_known,
    }


def generate_fecaba_baseline_predictions(
    dataset_root: str | Path,
    *,
    output: str | Path,
    dpi: int = 300,
) -> dict[str, object]:
    """Run the deterministic parser and compare it with reviewed test labels."""
    from .fecaba_dataset import _foul_labels
    from .pipeline import AnalysisContext, analyze_path

    root = Path(dataset_root)
    catalog = {
        item["document_id"]: item
        for item in _read_jsonl(root / "catalog.jsonl")
    }
    audit_dir = Path(output).parent
    audit_fecaba_dataset(root, output_dir=audit_dir)
    manifest = _read_jsonl(audit_dir / "evaluation-manifest.jsonl")
    eligible = [
        item
        for item in manifest
        if item.get("split") == "test"
        and item.get("review_status") in REVIEWED_STATUSES
        and item.get("ground_truth_complete") is True
    ]
    if not eligible:
        raise ValueError("No reviewed and complete test documents are available")

    rows: list[dict[str, object]] = []
    for manifest_item in eligible:
        document_id = str(manifest_item["document_id"])
        ground_truth = json.loads(
            (root / "ground_truth" / f"{document_id}.json").read_text(
                encoding="utf-8"
            )
        )
        record = catalog[document_id]
        rosters = {
            side: [int(number) for number in ground_truth["teams"][side]["roster"]]
            for side in ("A", "B")
        }
        writer_id = manifest_item.get("writer_id")
        result = analyze_path(
            root / record["source_path"],
            context=AnalysisContext(
                rosters=rosters,
                writer_id=str(writer_id) if writer_id else None,
                writer_known=writer_id is not None,
            ),
            dpi=dpi,
        )
        writer_known = writer_id is not None

        for side in ("A", "B"):
            team = result.teams[side]
            predicted_scores = {
                event.running_score: event for event in team.scoring_events
            }
            for expected_event in ground_truth["scoring"]:
                if expected_event["team"] != side:
                    continue
                running_score = int(expected_event["team_score_after"])
                predicted_event = predicted_scores.get(running_score)
                prefix = f"scoring:{side}:{running_score}"
                rows.append(
                    _prediction_row(
                        document_id=document_id,
                        field_id=f"{prefix}:points",
                        field_type="scoring_points",
                        expected=expected_event["points"],
                        predicted=(
                            predicted_event.points if predicted_event is not None else None
                        ),
                        confidence=(
                            predicted_event.confidence
                            if predicted_event is not None
                            else 0.0
                        ),
                        writer_known=writer_known,
                    )
                )
                rows.append(
                    _prediction_row(
                        document_id=document_id,
                        field_id=f"{prefix}:period",
                        field_type="scoring_period",
                        expected=expected_event["period"],
                        predicted=(
                            f"Q{predicted_event.period}"
                            if predicted_event is not None
                            and predicted_event.period is not None
                            else None
                        ),
                        confidence=(
                            predicted_event.confidence
                            if predicted_event is not None
                            else 0.0
                        ),
                        writer_known=writer_known,
                    )
                )
                if expected_event.get("jersey") is not None:
                    rows.append(
                        _prediction_row(
                            document_id=document_id,
                            field_id=f"{prefix}:jersey",
                            field_type="scoring_jersey",
                            expected=expected_event["jersey"],
                            predicted=(
                                predicted_event.jersey
                                if predicted_event is not None
                                else None
                            ),
                            confidence=(
                                predicted_event.jersey_confidence
                                if predicted_event is not None
                                else 0.0
                            ),
                            writer_known=writer_known,
                        )
                    )

            period_results = {period.number: period for period in team.periods}
            for period in range(1, 5):
                predicted_period = period_results.get(period)
                rows.append(
                    _prediction_row(
                        document_id=document_id,
                        field_id=f"period_score:{side}:Q{period}",
                        field_type="period_score",
                        expected=ground_truth["period_scores"][side][f"Q{period}"],
                        predicted=(
                            predicted_period.score
                            if predicted_period is not None
                            else None
                        ),
                        confidence=(
                            predicted_period.confidence
                            if predicted_period is not None
                            else 0.0
                        ),
                        writer_known=writer_known,
                    )
                )
            scoring_confidences = [
                float(event.confidence or 0.0) for event in team.scoring_events
            ]
            rows.append(
                _prediction_row(
                    document_id=document_id,
                    field_id=f"final_score:{side}",
                    field_type="final_score",
                    expected=ground_truth["final_score"][side],
                    predicted=team.calculated_score,
                    confidence=(min(scoring_confidences) if scoring_confidences else 0.0),
                    writer_known=writer_known,
                )
            )

            predicted_team_fouls = {
                item.period: item for item in team.team_fouls
            }
            for period in range(1, 5):
                predicted_foul = predicted_team_fouls.get(period)
                rows.append(
                    _prediction_row(
                        document_id=document_id,
                        field_id=f"team_fouls:{side}:Q{period}",
                        field_type="team_fouls",
                        expected=ground_truth["teams"][side]["team_fouls"][f"Q{period}"],
                        predicted=(
                            predicted_foul.x_count
                            if predicted_foul is not None
                            else None
                        ),
                        confidence=(
                            predicted_foul.confidence
                            if predicted_foul is not None
                            else 0.0
                        ),
                        writer_known=writer_known,
                    )
                )

            players = {player.jersey: player for player in team.players}
            for jersey in rosters[side]:
                labels = _foul_labels(ground_truth, side, jersey)
                player = players.get(jersey)
                predicted_fouls = (
                    {foul.slot: foul for foul in player.fouls}
                    if player is not None
                    else {}
                )
                for slot, (period_name, symbol) in labels.items():
                    predicted_foul = predicted_fouls.get(slot)
                    prefix = f"individual_foul:{side}:{jersey}:{slot}"
                    rows.append(
                        _prediction_row(
                            document_id=document_id,
                            field_id=f"{prefix}:symbol",
                            field_type="individual_foul_symbol",
                            expected=symbol,
                            predicted=(
                                predicted_foul.raw_symbol
                                if predicted_foul is not None
                                else None
                            ),
                            confidence=(
                                predicted_foul.confidence
                                if predicted_foul is not None
                                else 0.0
                            ),
                            writer_known=writer_known,
                        )
                    )
                    rows.append(
                        _prediction_row(
                            document_id=document_id,
                            field_id=f"{prefix}:period",
                            field_type="individual_foul_period",
                            expected=period_name,
                            predicted=(
                                f"Q{predicted_foul.period}"
                                if predicted_foul is not None
                                and predicted_foul.period is not None
                                else None
                            ),
                            confidence=(
                                predicted_foul.confidence
                                if predicted_foul is not None
                                else 0.0
                            ),
                            writer_known=writer_known,
                        )
                    )

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "documents": len(eligible),
        "prediction_records": len(rows),
        "output": str(target),
        "parser_mode": "deterministic_without_handwriting_models",
    }
