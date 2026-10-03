from __future__ import annotations

import argparse
import json
from pathlib import Path

from .completeness import assess_core_fields
from .imaging import load_document, normalize_document, save_debug_bundle
from .label_studio_review import REVIEW_STAGES
from .models import DocumentResult
from .pipeline import AnalysisContext, analyze_path
from .template import TEMPLATES, get_template


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sumula-reader")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("schema", help="Print the current structured output skeleton")

    debug = sub.add_parser(
        "debug",
        help="Normalize a scoresheet and save template overlays and crops",
    )
    debug.add_argument("input", type=Path)
    debug.add_argument("--output", type=Path, default=Path("debug"))
    debug.add_argument(
        "--template",
        default="fecaba_v1",
        choices=sorted(TEMPLATES),
    )
    debug.add_argument("--dpi", type=int, default=250)
    debug.add_argument(
        "--strict-alignment",
        action="store_true",
        help="Fail when the outer scoresheet border cannot be detected",
    )

    analyze = sub.add_parser(
        "analyze",
        help="Run the FECABA pipeline and return structured JSON",
    )
    analyze.add_argument("input", type=Path)
    analyze.add_argument(
        "--roster-a",
        help="Comma-separated team A roster; overrides automatic recognition",
    )
    analyze.add_argument(
        "--roster-b",
        help="Comma-separated team B roster; overrides automatic recognition",
    )
    analyze.add_argument("--team-a-name")
    analyze.add_argument("--team-b-name")
    analyze.add_argument(
        "--period-scores-a",
        help="Four comma-separated written period scores for team A",
    )
    analyze.add_argument(
        "--period-scores-b",
        help="Four comma-separated written period scores for team B",
    )
    analyze.add_argument("--final-score-a", type=int)
    analyze.add_argument("--final-score-b", type=int)
    analyze.add_argument("--writer-id")
    analyze.add_argument(
        "--handwriting-model-dir",
        type=Path,
        help="Directory containing jersey.pt and/or foul.pt",
    )
    analyze.add_argument("--output", type=Path)
    analyze.add_argument(
        "--template",
        default="fecaba_v1",
        choices=sorted(TEMPLATES),
    )
    analyze.add_argument("--dpi", type=int, default=250)
    analyze.add_argument(
        "--jev",
        action="store_true",
        help="Use Jev/TypeSafe only for ambiguous visual decisions",
    )
    analyze.add_argument(
        "--jev-model",
        help="TypeSafe model (default: jev-latest or TYPESAFE_MODEL)",
    )

    train_digits = sub.add_parser(
        "train-digits",
        help="Train the handwritten-digit baseline on EMNIST",
    )
    train_digits.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/external/emnist"),
    )
    train_digits.add_argument(
        "--output",
        type=Path,
        default=Path("models/digit-cnn-v1.pt"),
    )
    train_digits.add_argument("--epochs", type=int, default=5)
    train_digits.add_argument("--batch-size", type=int, default=128)
    train_digits.add_argument("--learning-rate", type=float, default=1e-3)
    train_digits.add_argument("--seed", type=int, default=20260930)
    train_digits.add_argument("--max-train-samples", type=int)
    train_digits.add_argument("--max-test-samples", type=int)

    dataset_ingest = sub.add_parser(
        "dataset-ingest",
        help="Extract, catalogue, and validate scoresheets from a ZIP archive",
    )
    dataset_ingest.add_argument("archive", type=Path)
    dataset_ingest.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/fecaba"),
    )
    dataset_ingest.add_argument("--dpi", type=int, default=150)
    dataset_ingest.add_argument("--pilot-count", type=int, default=5)

    dataset_build = sub.add_parser(
        "dataset-build",
        help="Build crops and a manifest from the catalogue and ground truth",
    )
    dataset_build.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/fecaba"),
    )
    dataset_build.add_argument("--output", type=Path)
    dataset_build.add_argument("--dpi", type=int, default=250)
    dataset_build.add_argument(
        "--all-documents",
        action="store_true",
        help="Process all ground truth instead of only the pilot batch",
    )

    dataset_audit = sub.add_parser(
        "dataset-audit",
        help="Audits M1 ground truth, corpus fingerprint, and split isolation",
    )
    dataset_audit.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/fecaba"),
    )
    dataset_audit.add_argument("--output-dir", type=Path)

    dataset_baseline = sub.add_parser(
        "dataset-baseline",
        help="Builds reproducible M1 baseline metrics from prediction JSONL",
    )
    dataset_baseline.add_argument("predictions", type=Path)
    dataset_baseline.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/fecaba"),
    )
    dataset_baseline.add_argument("--output-dir", type=Path)

    dataset_predictions = sub.add_parser(
        "dataset-predict-baseline",
        help="Generates parser predictions for reviewed M1 test data",
    )
    dataset_predictions.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/fecaba"),
    )
    dataset_predictions.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/fecaba/evaluation/baseline-predictions.jsonl"),
    )
    dataset_predictions.add_argument("--dpi", type=int, default=300)
    dataset_predictions.add_argument(
        "--handwriting-model-dir",
        type=Path,
        help="Directory containing jersey.pt and/or foul.pt",
    )
    dataset_baseline.add_argument(
        "--max-accepted-error-rate", type=float, default=0.01
    )
    dataset_baseline.add_argument("--min-automation-rate", type=float, default=0.10)

    review_prepare = sub.add_parser(
        "dataset-review-prepare",
        help="Prepare local Label Studio tasks for assisted review",
    )
    review_prepare.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/fecaba"),
    )
    review_prepare.add_argument(
        "--stage",
        required=True,
        choices=REVIEW_STAGES,
    )
    review_prepare.add_argument("--manifest", type=Path)
    review_prepare.add_argument("--dpi", type=int, default=250)
    review_prepare.add_argument(
        "--audit-only",
        action="store_true",
        help="Prepare only the 10%% sample marked for a second review",
    )

    review_import = sub.add_parser(
        "dataset-review-import",
        help="Import a Label Studio JSON export and validate the review",
    )
    review_import.add_argument("export", type=Path)
    review_import.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/fecaba"),
    )
    review_import.add_argument(
        "--stage",
        required=True,
        choices=REVIEW_STAGES,
    )
    review_import.add_argument("--manifest", type=Path)
    review_import.add_argument(
        "--audit-only",
        action="store_true",
        help="Import the audit sample's second review",
    )

    review_status_parser = sub.add_parser(
        "dataset-review-status",
        help="Show assisted-review progress",
    )
    review_status_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/fecaba"),
    )

    for name, help_text, default_output in (
        ("train-jerseys", "Train the jersey-number recognizer", "models/handwriting/jersey.pt"),
        ("train-fouls", "Train the foul-symbol recognizer", "models/handwriting/foul.pt"),
    ):
        training = sub.add_parser(name, help=help_text)
        training.add_argument(
            "--data-dir",
            type=Path,
            default=Path("datasets/external/emnist"),
        )
        training.add_argument("--output", type=Path, default=Path(default_output))
        training.add_argument(
            "--manifest",
            type=Path,
            default=Path("datasets/fecaba/crops/manifest.reviewed.jsonl"),
        )
        training.add_argument("--epochs", type=int, default=5)
        training.add_argument("--batch-size", type=int, default=128)
        training.add_argument("--learning-rate", type=float, default=1e-3)
        training.add_argument("--synthetic-train-samples", type=int, default=20_000)
        training.add_argument("--synthetic-test-samples", type=int, default=4_000)
        training.add_argument("--seed", type=int, default=20261001)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "schema":
        print(
            json.dumps(
                assess_core_fields(DocumentResult()).to_dict(),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "debug":
        template = get_template(args.template)
        image = load_document(args.input, dpi=args.dpi)
        document = normalize_document(
            image,
            template,
            strict=args.strict_alignment,
        )
        files = save_debug_bundle(document, template, args.output)
        print(
            json.dumps(
                {
                    "template": template.template_id,
                    "perspective_warp": document.used_perspective_warp,
                    "normalization_method": document.method,
                    "output": str(args.output),
                    "files_written": len(files),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "train-digits":
        from .handwriting_ml import DigitTrainingConfig, train_emnist_digits

        result = train_emnist_digits(
            DigitTrainingConfig(
                data_dir=args.data_dir,
                output=args.output,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
                max_train_samples=args.max_train_samples,
                max_test_samples=args.max_test_samples,
            )
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "dataset-ingest":
        from .fecaba_dataset import ingest_scoresheet_archive

        result = ingest_scoresheet_archive(
            args.archive,
            output_root=args.output,
            dpi=args.dpi,
            pilot_count=args.pilot_count,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "dataset-build":
        from .fecaba_dataset import build_fecaba_crops

        result = build_fecaba_crops(
            args.dataset_root,
            output_root=args.output,
            selected_only=not args.all_documents,
            dpi=args.dpi,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "dataset-audit":
        from .m1_audit import audit_fecaba_dataset

        result = audit_fecaba_dataset(
            args.dataset_root,
            output_dir=args.output_dir,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "dataset-predict-baseline":
        from .m1_audit import generate_fecaba_baseline_predictions

        handwriting = None
        if args.handwriting_model_dir is not None:
            from .torch_recognition import TorchHandwritingRecognizer

            handwriting = TorchHandwritingRecognizer(args.handwriting_model_dir)
        result = generate_fecaba_baseline_predictions(
            args.dataset_root,
            output=args.output,
            dpi=args.dpi,
            handwriting=handwriting,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "dataset-baseline":
        from .m1_audit import build_baseline_metrics

        result = build_baseline_metrics(
            args.dataset_root,
            args.predictions,
            output_dir=args.output_dir,
            max_accepted_error_rate=args.max_accepted_error_rate,
            min_automation_rate=args.min_automation_rate,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "dataset-review-prepare":
        from .label_studio_review import prepare_review

        result = prepare_review(
            args.dataset_root,
            stage=args.stage,
            manifest=args.manifest,
            dpi=args.dpi,
            audit_only=args.audit_only,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "dataset-review-import":
        from .label_studio_review import import_review

        result = import_review(
            args.dataset_root,
            args.export,
            stage=args.stage,
            manifest=args.manifest,
            audit_only=args.audit_only,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "dataset-review-status":
        from .label_studio_review import review_status

        print(
            json.dumps(
                review_status(args.dataset_root),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command in {"train-jerseys", "train-fouls"}:
        from .torch_recognition import (
            SequenceTrainingConfig,
            train_foul_symbols,
            train_jersey_sequences,
        )

        config = SequenceTrainingConfig(
            data_dir=args.data_dir,
            output=args.output,
            manifest=args.manifest if args.manifest.exists() else None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            synthetic_train_samples=args.synthetic_train_samples,
            synthetic_test_samples=args.synthetic_test_samples,
            seed=args.seed,
        )
        trainer = (
            train_jersey_sequences
            if args.command == "train-jerseys"
            else train_foul_symbols
        )
        print(json.dumps(trainer(config), ensure_ascii=False, indent=2))
        return

    if args.command == "analyze":
        template = get_template(args.template)
        handwriting = None
        if args.handwriting_model_dir is not None:
            from .torch_recognition import TorchHandwritingRecognizer

            handwriting = TorchHandwritingRecognizer(args.handwriting_model_dir)
        decision_engine = None
        if args.jev:
            from .jev import JevDecisionEngine, TypeSafeAPIError

            try:
                decision_engine = JevDecisionEngine.from_env_or_prompt(
                    model=args.jev_model,
                )
            except TypeSafeAPIError as exc:
                raise SystemExit(str(exc)) from exc
        context = AnalysisContext(
            rosters={
                side: _parse_roster(value)
                for side, value in (("A", args.roster_a), ("B", args.roster_b))
                if value is not None
            },
            team_names={
                side: name.strip()
                for side, name in (
                    ("A", args.team_a_name),
                    ("B", args.team_b_name),
                )
                if name is not None and name.strip()
            },
            period_scores={
                side: _parse_period_scores(value)
                for side, value in (
                    ("A", args.period_scores_a),
                    ("B", args.period_scores_b),
                )
                if value is not None
            },
            final_scores={
                side: value
                for side, value in (
                    ("A", args.final_score_a),
                    ("B", args.final_score_b),
                )
                if value is not None
            },
            writer_id=args.writer_id,
            writer_known=args.writer_id is not None,
        )
        result = analyze_path(
            args.input,
            context=context,
            template=template,
            handwriting=handwriting,
            decision_engine=decision_engine,
            dpi=args.dpi,
        )
        payload = json.dumps(
            result.to_dict(),
            ensure_ascii=False,
            indent=2,
        )
        if args.output is None:
            print(payload)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
            print(str(args.output))


def _parse_roster(value: str) -> list[int]:
    numbers: list[int] = []
    seen: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            number = int(item)
        except ValueError as exc:
            raise SystemExit(f"Invalid jersey number: {item}") from exc
        if not 0 <= number <= 99:
            raise SystemExit(f"Jersey number outside the 0..99 range: {number}")
        if number in seen:
            raise SystemExit(f"Duplicate jersey number: {number}")
        seen.add(number)
        numbers.append(number)
    if not numbers:
        raise SystemExit("The roster cannot be empty")
    if len(numbers) > 12:
        raise SystemExit("The FECABA template supports at most 12 players")
    return numbers


def _parse_period_scores(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",")]
    if len(items) != 4:
        raise SystemExit("Period scores must contain exactly four comma-separated values")
    try:
        scores = [int(item) for item in items]
    except ValueError as exc:
        raise SystemExit("Period scores must be non-negative integers") from exc
    if any(score < 0 for score in scores):
        raise SystemExit("Period scores must be non-negative integers")
    return scores


if __name__ == "__main__":
    main()
