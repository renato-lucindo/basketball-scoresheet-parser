from __future__ import annotations

import argparse
import json
from pathlib import Path

from .imaging import load_document, normalize_document, save_debug_bundle
from .models import DocumentResult
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


if __name__ == "__main__":
    main()
