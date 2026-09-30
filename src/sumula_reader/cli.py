from __future__ import annotations

import argparse
import json
from pathlib import Path

from .imaging import load_document, normalize_document, save_debug_bundle
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
    analyze.add_argument("--output", type=Path)
    analyze.add_argument(
        "--template",
        default="fecaba_v1",
        choices=sorted(TEMPLATES),
    )
    analyze.add_argument("--dpi", type=int, default=250)
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

    if args.command == "analyze":
        template = get_template(args.template)
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
