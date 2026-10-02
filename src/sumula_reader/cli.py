from __future__ import annotations

import argparse
import json
from pathlib import Path

from .imaging import load_document, normalize_document, save_debug_bundle
from .label_studio_review import REVIEW_STAGES
from .models import DocumentResult
from .pipeline import AnalysisContext, analyze_path
from .template import TEMPLATES, get_template


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sumula-reader")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("schema", help="Mostra o esqueleto JSON do MVP Core")

    debug = sub.add_parser(
        "debug",
        help="Normaliza uma sumula e salva overlay/crops das regioes do template",
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
        help="Falha se a borda externa da sumula nao puder ser detectada",
    )

    analyze = sub.add_parser(
        "analyze",
        help="Executa o pipeline atual do MVP e retorna JSON estruturado",
    )
    analyze.add_argument("input", type=Path)
    analyze.add_argument("--roster-a", required=True)
    analyze.add_argument("--roster-b", required=True)
    analyze.add_argument("--writer-id")
    analyze.add_argument(
        "--handwriting-model-dir",
        type=Path,
        help="Diretorio contendo jersey.pt e/ou foul.pt",
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
        help="Usa Jev/TypeSafe apenas para decisoes visuais ambiguas",
    )
    analyze.add_argument(
        "--jev-model",
        help="Modelo TypeSafe (padrao: jev-latest ou TYPESAFE_MODEL)",
    )

    train_digits = sub.add_parser(
        "train-digits",
        help="Treina o baseline de digitos manuscritos no EMNIST",
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
        help="Extrai, cataloga e valida sumulas de um arquivo ZIP",
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
        help="Gera crops e manifest a partir do catalogo e dos gabaritos",
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
        help="Processa todos os gabaritos em vez de apenas o lote piloto",
    )

    review_prepare = sub.add_parser(
        "dataset-review-prepare",
        help="Prepara tarefas locais do Label Studio para revisao assistida",
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
        help="Prepara somente a amostra de 10%% marcada para segunda revisao",
    )

    review_import = sub.add_parser(
        "dataset-review-import",
        help="Importa um export JSON do Label Studio e valida a revisao",
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
        help="Importa a segunda revisao da amostra de auditoria",
    )

    review_status_parser = sub.add_parser(
        "dataset-review-status",
        help="Mostra o progresso da revisao assistida",
    )
    review_status_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/fecaba"),
    )

    for name, help_text, default_output in (
        ("train-jerseys", "Treina o reconhecedor de camisas", "models/handwriting/jersey.pt"),
        ("train-fouls", "Treina o reconhecedor de simbolos de falta", "models/handwriting/foul.pt"),
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
        print(json.dumps(DocumentResult().to_dict(), ensure_ascii=False, indent=2))
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
                "A": _parse_roster(args.roster_a),
                "B": _parse_roster(args.roster_b),
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
            raise SystemExit(f"Numero de camisa invalido: {item}") from exc
        if not 0 <= number <= 99:
            raise SystemExit(f"Numero de camisa fora do intervalo 0..99: {number}")
        if number in seen:
            raise SystemExit(f"Numero de camisa duplicado: {number}")
        seen.add(number)
        numbers.append(number)
    if not numbers:
        raise SystemExit("O roster nao pode ser vazio")
    if len(numbers) > 12:
        raise SystemExit("O template FECABA suporta no maximo 12 jogadores")
    return numbers


if __name__ == "__main__":
    main()
