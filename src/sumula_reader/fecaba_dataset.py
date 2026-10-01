from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import unicodedata
import zipfile

import numpy as np
from PIL import Image

from .dataset import assign_document_split, validate_split_isolation
from .fouls import (
    PLAYER_FOUL_GRID,
    _period_for_foul,
    detect_half_separator,
)
from .imaging import load_document, normalize_document
from .imaging import crop_region
from .scoring import (
    DEFAULT_GRID,
    InkColor,
    _iter_scoring_crops,
    _looks_like_closure_stroke,
    colored_ink_masks,
    detect_scoring_grid,
)
from .template import FECABA_V1


SUPPORTED_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png"}


@dataclass(slots=True)
class IngestedDocument:
    document_id: str
    original_name: str
    source_path: str
    category: str
    extension: str
    sha256: str
    size_bytes: int
    page_count: int | None
    image_width: int | None
    image_height: int | None
    alignment_method: str | None
    used_perspective_warp: bool | None
    status: str
    error: str | None = None
    selected_pilot: bool = False


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_value).strip("-").lower()
    return slug or "documento"


def _safe_archive_parts(name: str) -> tuple[str, str]:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
        raise ValueError(f"Caminho inseguro ou sem categoria no ZIP: {name!r}")
    return path.parts[-2], path.name


def _pdf_page_count(path: Path) -> int:
    import pymupdf

    with pymupdf.open(path) as document:
        return document.page_count


def _empty_ground_truth(record: IngestedDocument) -> dict[str, object]:
    periods = {f"Q{period}": None for period in range(1, 5)}
    return {
        "game_id": record.document_id,
        "source_file": record.source_path,
        "writer_name": None,
        "teams": {
            side: {
                "roster": [],
                "team_fouls": dict(periods),
                "individual_fouls": {},
            }
            for side in ("A", "B")
        },
        "scoring": [],
        "period_scores": {
            side: dict(periods) for side in ("A", "B")
        },
        "final_score": {"A": None, "B": None},
        "warnings": ["Gabarito pendente de preenchimento manual."],
    }


def _select_pilots(
    records: list[IngestedDocument],
    *,
    limit: int,
) -> list[IngestedDocument]:
    usable = [record for record in records if record.status != "unreadable"]
    categories = sorted({record.category for record in usable})
    jpeg_categories = sorted(
        {
            record.category
            for record in usable
            if record.extension in {".jpg", ".jpeg"}
        }
    )
    jpeg_category = jpeg_categories[0] if jpeg_categories else None
    selected: list[IngestedDocument] = []
    for category in categories:
        candidates = [record for record in usable if record.category == category]
        candidates.sort(
            key=lambda record: (
                record.status != "ready",
                not (
                    category == jpeg_category
                    and record.extension in {".jpg", ".jpeg"}
                ),
                record.document_id,
            )
        )
        if candidates:
            candidates[0].selected_pilot = True
            selected.append(candidates[0])
        if len(selected) >= limit:
            break
    return selected


def ingest_scoresheet_archive(
    archive: str | Path,
    *,
    output_root: str | Path,
    dpi: int = 150,
    pilot_count: int = 5,
) -> dict[str, object]:
    source_zip = Path(archive)
    root = Path(output_root)
    source_root = root / "source"
    ground_truth_root = root / "ground_truth"
    source_root.mkdir(parents=True, exist_ok=True)
    ground_truth_root.mkdir(parents=True, exist_ok=True)
    records: list[IngestedDocument] = []

    with zipfile.ZipFile(source_zip) as zipped:
        entries = [entry for entry in zipped.infolist() if not entry.is_dir()]
        for entry in entries:
            suffix = PurePosixPath(entry.filename).suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                continue
            category_raw, filename = _safe_archive_parts(entry.filename)
            category = _slug(category_raw)
            payload = zipped.read(entry)
            digest = hashlib.sha256(payload).hexdigest()
            document_id = (
                f"{_slug(category_raw)}-{_slug(Path(filename).stem)}-{digest[:8]}"
            )
            relative_path = Path("source") / category / f"{document_id}{suffix}"
            target = root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                target.write_bytes(payload)

            record = IngestedDocument(
                document_id=document_id,
                original_name=entry.filename,
                source_path=relative_path.as_posix(),
                category=category,
                extension=suffix,
                sha256=digest,
                size_bytes=len(payload),
                page_count=None,
                image_width=None,
                image_height=None,
                alignment_method=None,
                used_perspective_warp=None,
                status="unreadable",
            )
            try:
                record.page_count = _pdf_page_count(target) if suffix == ".pdf" else 1
                image = load_document(target, dpi=dpi)
                record.image_height, record.image_width = image.shape[:2]
                normalized = normalize_document(image, FECABA_V1)
                record.alignment_method = normalized.method
                record.used_perspective_warp = normalized.used_perspective_warp
                record.status = (
                    "ready"
                    if normalized.used_perspective_warp
                    else "alignment_review"
                )
            except Exception as exc:
                record.error = f"{type(exc).__name__}: {exc}"
            records.append(record)

    records.sort(key=lambda record: record.document_id)
    selected = _select_pilots(records, limit=pilot_count)
    catalog = root / "catalog.jsonl"
    with catalog.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    for record in selected:
        ground_truth = ground_truth_root / f"{record.document_id}.json"
        if not ground_truth.exists():
            ground_truth.write_text(
                json.dumps(_empty_ground_truth(record), ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )

    statuses: dict[str, int] = {}
    for record in records:
        statuses[record.status] = statuses.get(record.status, 0) + 1
    summary = {
        "archive": str(source_zip),
        "output_root": str(root),
        "documents": len(records),
        "statuses": statuses,
        "categories": sorted({record.category for record in records}),
        "pilot_documents": [record.document_id for record in selected],
        "catalog": str(catalog),
    }
    (root / "ingest-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_ground_truth(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Gabarito invalido em {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Gabarito deve ser um objeto JSON: {path}")
    teams = payload.get("teams")
    if not isinstance(teams, dict) or set(teams) != {"A", "B"}:
        raise ValueError(f"Gabarito deve conter exatamente as equipes A e B: {path}")
    scoring = payload.get("scoring")
    if not isinstance(scoring, list):
        raise ValueError(f"Campo scoring deve ser uma lista: {path}")

    for side in ("A", "B"):
        team = teams[side]
        if not isinstance(team, dict) or not isinstance(team.get("roster"), list):
            raise ValueError(f"Roster da equipe {side} deve ser uma lista: {path}")
        roster = {int(number) for number in team["roster"]}
        for jersey in roster:
            if jersey < 0 or jersey > 99:
                raise ValueError(f"Camisa fora de 0..99 na equipe {side}: {jersey}")
        individual = team.get("individual_fouls")
        if not isinstance(individual, dict):
            raise ValueError(f"individual_fouls da equipe {side} deve ser objeto")
        for jersey_text in individual:
            if roster and int(jersey_text) not in roster:
                raise ValueError(
                    f"Falta atribuida a camisa {jersey_text} fora do roster {side}"
                )

    for index, event in enumerate(scoring):
        if not isinstance(event, dict):
            raise ValueError(f"Evento scoring[{index}] deve ser objeto")
        side = event.get("team")
        if side not in {"A", "B"}:
            raise ValueError(f"Equipe invalida em scoring[{index}]")
        points = event.get("points")
        if points not in {1, 2, 3}:
            raise ValueError(f"Pontos invalidos em scoring[{index}]")
        jersey = event.get("jersey")
        roster = {int(number) for number in teams[side]["roster"]}
        if jersey is not None and roster and int(jersey) not in roster:
            raise ValueError(
                f"Camisa {jersey} de scoring[{index}] fora do roster {side}"
            )
    return payload


def _anonymous_writer_id(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    digest = hashlib.sha256(f"writer:{value.strip().casefold()}".encode()).hexdigest()
    return f"writer_{digest[:12]}"


def _save_crop(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def _scoring_labels(ground_truth: dict[str, object]) -> dict[tuple[str, int], dict]:
    result: dict[tuple[str, int], dict] = {}
    for event in ground_truth["scoring"]:
        score = event.get("team_score_after")
        if isinstance(score, int):
            result[(event["team"], score)] = event
    return result


def _foul_labels(
    ground_truth: dict[str, object],
    side: str,
    jersey: int,
) -> list[tuple[str, str]]:
    team = ground_truth["teams"][side]
    periods = team.get("individual_fouls", {}).get(str(jersey), {})
    labels: list[tuple[str, str]] = []
    for period in range(1, 5):
        period_name = f"Q{period}"
        for symbol in periods.get(period_name, []):
            labels.append((period_name, str(symbol).upper().replace(" ", "")))
    return labels


def build_fecaba_crops(
    dataset_root: str | Path,
    *,
    output_root: str | Path | None = None,
    selected_only: bool = True,
    dpi: int = 250,
) -> dict[str, object]:
    root = Path(dataset_root)
    crops_root = Path(output_root) if output_root is not None else root / "crops"
    catalog_path = root / "catalog.jsonl"
    records = [
        json.loads(line)
        for line in catalog_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if selected_only:
        records = [record for record in records if record["selected_pilot"]]

    manifest: list[dict[str, object]] = []
    processed_documents = 0
    for document in records:
        document_id = document["document_id"]
        ground_truth_path = root / "ground_truth" / f"{document_id}.json"
        if not ground_truth_path.exists():
            continue
        ground_truth = _load_ground_truth(ground_truth_path)
        writer_id = _anonymous_writer_id(ground_truth.get("writer_name"))
        source = root / document["source_path"]
        normalized = normalize_document(load_document(source, dpi=dpi), FECABA_V1)
        split = assign_document_split(document_id, writer_id=writer_id)
        scoring_labels = _scoring_labels(ground_truth)

        table = crop_region(normalized.image, FECABA_V1.region("scoring_table"))
        detected_grid = detect_scoring_grid(table, grid=DEFAULT_GRID)
        for cell, _, jersey_crop, grid_aligned in _iter_scoring_crops(
            table, detected_grid, DEFAULT_GRID
        ):
            red_mask, blue_mask = colored_ink_masks(jersey_crop)
            ink_ratio = float((red_mask | blue_mask).mean())
            if ink_ratio < 0.025 or _looks_like_closure_stroke(jersey_crop):
                continue
            expected = scoring_labels.get((cell.team, cell.running_score))
            label = expected.get("jersey") if expected is not None else None
            relative = (
                Path("jerseys")
                / split
                / document_id
                / f"{cell.team}-{cell.running_score:03d}.png"
            )
            _save_crop(jersey_crop, crops_root / relative)
            manifest.append(
                {
                    "document_id": document_id,
                    "writer_id": writer_id,
                    "field_type": "jersey",
                    "label": str(label) if label is not None else None,
                    "crop_path": relative.as_posix(),
                    "split": split,
                    "team": cell.team,
                    "period": expected.get("period") if expected else None,
                    "running_score": cell.running_score,
                    "grid_aligned": grid_aligned,
                    "status": "labeled" if label is not None else "unlabeled",
                }
            )

        for side in ("A", "B"):
            team = ground_truth["teams"][side]
            roster = [int(number) for number in team["roster"]]
            block = crop_region(
                normalized.image,
                FECABA_V1.region(f"team_{side.lower()}_player_fouls"),
            )
            grid = PLAYER_FOUL_GRID
            data_top = round(block.shape[0] * grid.header_fraction)
            data = block[data_top:]
            row_height = data.shape[0] / grid.roster_rows
            for row_index in range(grid.roster_rows):
                top = round(row_index * row_height)
                bottom = round((row_index + 1) * row_height)
                row = data[top:bottom]
                jersey = roster[row_index] if row_index < len(roster) else None
                expected_fouls = (
                    _foul_labels(ground_truth, side, jersey)
                    if jersey is not None
                    else []
                )
                separator = detect_half_separator(row, slots=grid.slots)
                height, width = row.shape[:2]
                slot_width = width / grid.slots
                event_index = 0
                for slot_index in range(grid.slots):
                    left = round(slot_index * slot_width)
                    right = round((slot_index + 1) * slot_width)
                    margin_x = round((right - left) * grid.inner_margin)
                    margin_y = round(height * grid.inner_margin)
                    cell_crop = row[
                        margin_y : max(margin_y + 1, height - margin_y),
                        left + margin_x : max(left + margin_x + 1, right - margin_x),
                    ]
                    red_mask, blue_mask = colored_ink_masks(cell_crop)
                    if float((red_mask | blue_mask).mean()) < 0.006:
                        continue
                    red_count = int(red_mask.sum())
                    blue_count = int(blue_mask.sum())
                    color = (
                        InkColor.RED
                        if red_count > blue_count
                        else InkColor.BLUE
                        if blue_count > red_count
                        else InkColor.UNKNOWN
                    )
                    period_number = _period_for_foul(
                        slot=slot_index + 1,
                        first_half_slots=separator.first_half_slots,
                        color=color,
                    )
                    expected = (
                        expected_fouls[event_index]
                        if event_index < len(expected_fouls)
                        else None
                    )
                    event_index += 1
                    label = expected[1] if expected is not None else None
                    relative = (
                        Path("fouls")
                        / split
                        / document_id
                        / f"{side}-row{row_index + 1:02d}-slot{slot_index + 1}.png"
                    )
                    _save_crop(cell_crop, crops_root / relative)
                    manifest.append(
                        {
                            "document_id": document_id,
                            "writer_id": writer_id,
                            "field_type": "foul_symbol",
                            "label": label,
                            "crop_path": relative.as_posix(),
                            "split": split,
                            "team": side,
                            "period": (
                                expected[0]
                                if expected is not None
                                else f"Q{period_number}" if period_number else None
                            ),
                            "jersey": jersey,
                            "row": row_index + 1,
                            "slot": slot_index + 1,
                            "ink_color": color.value,
                            "status": "labeled" if label is not None else "unlabeled",
                        }
                    )
        processed_documents += 1

    validate_split_isolation(manifest)
    crops_root.mkdir(parents=True, exist_ok=True)
    manifest_path = crops_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        "documents": processed_documents,
        "crops": len(manifest),
        "labeled": sum(item["status"] == "labeled" for item in manifest),
        "unlabeled": sum(item["status"] == "unlabeled" for item in manifest),
        "split_isolation": "ok",
        "manifest": str(manifest_path),
    }
    (crops_root / "build-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
