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
    classify_foul_terminal,
    clean_player_foul_cell,
    detect_half_separator,
    detect_player_foul_grid,
)
from .imaging import load_document, normalize_document
from .imaging import crop_region
from .label_studio_review import load_geometry_overrides, reviewed_region, stable_crop_id
from .scoring import (
    DEFAULT_GRID,
    InkColor,
    ScoreMarkKind,
    _iter_scoring_crops,
    _looks_like_closure_stroke,
    classify_score_mark,
    colored_ink_masks,
    detect_jersey_circle,
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
        "evaluation_split": None,
        "review": {
            "status": "pending",
            "reviewed_at": None,
            "method": "manual",
        },
        "teams": {
            side: {
                "roster": [],
                "team_fouls": dict(periods),
                "individual_fouls": {},
                "individual_fouls_reviewed": False,
                "individual_foul_observations": {},
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
    evaluation_split = payload.get("evaluation_split")
    if evaluation_split not in {None, "train", "validation", "test"}:
        raise ValueError(
            f"evaluation_split deve ser train, validation, test ou null: {path}"
        )

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
        reviewed = team.get("individual_fouls_reviewed")
        if reviewed not in {True, False}:
            raise ValueError(
                f"individual_fouls_reviewed for team {side} must be a boolean: {path}"
            )
        observations = team.get("individual_foul_observations", {})
        if not isinstance(observations, dict):
            raise ValueError(
                f"individual_foul_observations for team {side} must be an object: {path}"
            )
        for jersey_text, entries in observations.items():
            try:
                jersey = int(jersey_text)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid observed-foul jersey for team {side}: {jersey_text!r}"
                ) from exc
            if roster and jersey not in roster:
                raise ValueError(
                    f"Observed foul assigned to jersey {jersey} outside roster {side}"
                )
            if not isinstance(entries, list):
                raise ValueError(
                    f"Observed fouls for team {side}, jersey {jersey} must be a list"
                )
            seen_slots: set[int] = set()
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"Observed foul for team {side}, jersey {jersey} must be an object"
                    )
                slot = entry.get("slot")
                if not isinstance(slot, int) or slot < 1 or slot > 5:
                    raise ValueError(
                        f"Observed foul slot for team {side}, jersey {jersey} must be 1..5"
                    )
                if slot in seen_slots:
                    raise ValueError(
                        f"Duplicate observed foul slot for team {side}, jersey {jersey}: {slot}"
                    )
                seen_slots.add(slot)
                symbol = entry.get("symbol")
                if symbol is not None and (
                    not isinstance(symbol, str) or not symbol.strip()
                ):
                    raise ValueError(
                        f"Observed foul symbol for team {side}, jersey {jersey} must be text or null"
                    )
                candidates = entry.get("period_candidates")
                if not isinstance(candidates, list) or not candidates or any(
                    period not in {"Q1", "Q2", "Q3", "Q4"}
                    for period in candidates
                ):
                    raise ValueError(
                        f"Observed foul period_candidates for team {side}, jersey {jersey} are invalid"
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


def _review_metadata(
    item: dict[str, object],
    crop: np.ndarray,
    *,
    source_image_hash: str,
    bbox_original: list[int] | None = None,
) -> dict[str, object]:
    height, width = crop.shape[:2]
    item["crop_id"] = stable_crop_id(item)
    item["source_image_hash"] = source_image_hash
    item["crop_image_hash"] = hashlib.sha256(
        np.ascontiguousarray(crop).tobytes()
    ).hexdigest()
    item.setdefault("review_state", "pending")
    item["bbox_original"] = bbox_original or [0, 0, width, height]
    if item.get("field_type") == "scoring_event":
        item["position"] = f"score:{item.get('team')}:{item.get('running_score')}"
    else:
        item["position"] = f"foul:{item.get('team')}:row{item.get('row')}:slot{item.get('slot')}"
    return item


def _combine_scoring_candidate(
    jersey_crop: np.ndarray,
    score_crop: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    jersey = np.asarray(jersey_crop, dtype=np.uint8)
    score = np.asarray(score_crop, dtype=np.uint8)
    height = max(jersey.shape[0], score.shape[0])
    gap = max(3, round(height * 0.08))
    width = jersey.shape[1] + gap + score.shape[1]
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    jersey_top = (height - jersey.shape[0]) // 2
    score_top = (height - score.shape[0]) // 2
    canvas[jersey_top : jersey_top + jersey.shape[0], : jersey.shape[1]] = jersey
    score_left = jersey.shape[1] + gap
    canvas[
        score_top : score_top + score.shape[0],
        score_left : score_left + score.shape[1],
    ] = score
    return canvas, [0, jersey_top, jersey.shape[1], jersey_top + jersey.shape[0]]


def _advance_scoring_period(
    state: tuple[int, InkColor | None],
    color: InkColor,
) -> tuple[int | None, tuple[int, InkColor | None]]:
    period, previous_color = state
    if color is InkColor.UNKNOWN:
        return None, state
    if previous_color is None:
        period = 1 if color is InkColor.RED else 2
        return period, (period, color)
    if color is previous_color:
        return period, state
    possible = (1, 3) if color is InkColor.RED else (2, 4)
    later = [candidate for candidate in possible if candidate > period]
    if not later:
        return None, state
    period = later[0]
    return period, (period, color)


def _terminal_crop_item(
    *,
    document_id: str,
    writer_id: str | None,
    relative: Path,
    split: str,
    side: str,
    jersey: int | None,
    row: int,
    slot: int,
    predicted_label: str,
) -> dict[str, object]:
    return {
        "document_id": document_id,
        "writer_id": writer_id,
        "field_type": "foul_terminal",
        "label": None,
        "predicted_label": predicted_label,
        "crop_path": relative.as_posix(),
        "split": split,
        "team": side,
        "period": None,
        "jersey": jersey,
        "row": row,
        "slot": slot,
        "status": "unlabeled",
        "grid_aligned": True,
        "model_trainable": False,
    }


def _reviewed_trainable_ids(
    crops_root: Path,
    manifest: list[dict[str, object]],
) -> set[str]:
    reviewed_path = crops_root / "manifest.reviewed.jsonl"
    if not reviewed_path.exists():
        return set()
    automatic = {
        str(item.get("crop_id")): item
        for item in manifest
        if item.get("crop_id") is not None
    }
    result: set[str] = set()
    for line in reviewed_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        crop_id = str(item.get("crop_id") or "")
        current = automatic.get(crop_id)
        if current is None:
            continue
        if current.get("model_trainable") is False:
            continue
        if item.get("review_state") not in {"accepted", "adjusted"}:
            continue
        if item.get("label") is None:
            continue
        if item.get("source_image_hash") != current.get("source_image_hash"):
            continue
        if item.get("crop_image_hash") != current.get("crop_image_hash"):
            continue
        result.add(crop_id)
    return result


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
    geometry_overrides = load_geometry_overrides(root)
    for document in records:
        document_id = document["document_id"]
        ground_truth_path = root / "ground_truth" / f"{document_id}.json"
        if not ground_truth_path.exists():
            continue
        ground_truth = _load_ground_truth(ground_truth_path)
        writer_id = _anonymous_writer_id(ground_truth.get("writer_name"))
        source = root / document["source_path"]
        normalized = normalize_document(load_document(source, dpi=dpi), FECABA_V1)
        split = ground_truth.get("evaluation_split") or assign_document_split(
            document_id,
            writer_id=writer_id,
        )
        scoring_labels = _scoring_labels(ground_truth)

        scoring_region = reviewed_region(geometry_overrides, document_id, "scoring_table")
        if scoring_region is not None:
            table = crop_region(normalized.image, scoring_region)
            detected_grid = detect_scoring_grid(table, grid=DEFAULT_GRID)
            period_state: dict[str, tuple[int, InkColor | None]] = {
                "A": (0, None),
                "B": (0, None),
            }
            scoring_cells = (
                _iter_scoring_crops(table, detected_grid, DEFAULT_GRID)
                if detected_grid.valid
                else ()
            )
            for cell, score_crop, jersey_crop, _ in scoring_cells:
                red_mask, blue_mask = colored_ink_masks(jersey_crop)
                ink_ratio = float((red_mask | blue_mask).mean())
                if ink_ratio < 0.025 or _looks_like_closure_stroke(jersey_crop):
                    continue
                mark = classify_score_mark(score_crop)
                circle = detect_jersey_circle(jersey_crop)
                ink_color = mark.ink_color
                if ink_color is InkColor.UNKNOWN:
                    red_count = int(red_mask.sum())
                    blue_count = int(blue_mask.sum())
                    ink_color = (
                        InkColor.RED
                        if red_count > blue_count
                        else InkColor.BLUE
                        if blue_count > red_count
                        else InkColor.UNKNOWN
                    )
                predicted_period, period_state[cell.team] = _advance_scoring_period(
                    period_state[cell.team], ink_color
                )
                if mark.kind is ScoreMarkKind.FREE_THROW:
                    predicted_type = "free_throw"
                elif mark.kind is ScoreMarkKind.FIELD_GOAL and circle.detected:
                    predicted_type = "three_point"
                elif mark.kind is ScoreMarkKind.FIELD_GOAL:
                    predicted_type = "two_point"
                else:
                    predicted_type = "ambiguous"
                expected = scoring_labels.get((cell.team, cell.running_score))
                label = expected.get("jersey") if expected is not None else None
                candidate, jersey_bbox = _combine_scoring_candidate(
                    jersey_crop,
                    score_crop,
                )
                relative = (
                    Path("scoring_events")
                    / split
                    / document_id
                    / f"{cell.team}-{cell.running_score:03d}.png"
                )
                _save_crop(candidate, crops_root / relative)
                item: dict[str, object] = {
                    "document_id": document_id,
                    "writer_id": writer_id,
                    "field_type": "scoring_event",
                    "label": str(label) if label is not None else None,
                    "crop_path": relative.as_posix(),
                    "split": split,
                    "team": cell.team,
                    "period": (
                        expected.get("period")
                        if expected is not None and expected.get("period") is not None
                        else f"Q{predicted_period}" if predicted_period else None
                    ),
                    "running_score": cell.running_score,
                    "predicted_type": predicted_type,
                    "ink_color": ink_color.value,
                    "grid_aligned": True,
                    "status": "labeled" if label is not None else "unlabeled",
                }
                manifest.append(
                    _review_metadata(
                        item,
                        candidate,
                        source_image_hash=str(document["sha256"]),
                        bbox_original=jersey_bbox,
                    )
                )

        for side in ("A", "B"):
            team = ground_truth["teams"][side]
            roster = [int(number) for number in team["roster"]]
            foul_region = reviewed_region(
                geometry_overrides,
                document_id,
                f"team_{side.lower()}_player_fouls",
            )
            if foul_region is None:
                continue
            block = crop_region(normalized.image, foul_region)
            grid = PLAYER_FOUL_GRID
            detected_foul_grid = detect_player_foul_grid(block, side=side, grid=grid)
            if not detected_foul_grid.valid:
                continue
            for row_index in range(grid.roster_rows):
                top = detected_foul_grid.y_lines[row_index]
                bottom = detected_foul_grid.y_lines[row_index + 1]
                row = block[top:bottom]
                jersey = roster[row_index] if row_index < len(roster) else None
                expected_fouls = (
                    _foul_labels(ground_truth, side, jersey)
                    if jersey is not None
                    else []
                )
                separator = detect_half_separator(row, slots=grid.slots)
                height = row.shape[0]
                event_index = 0
                for slot_index in range(grid.slots):
                    left = detected_foul_grid.x_lines[slot_index]
                    right = detected_foul_grid.x_lines[slot_index + 1]
                    margin_x = round((right - left) * grid.inner_margin)
                    margin_y = round(height * grid.inner_margin)
                    raw_cell = row[
                        margin_y : max(margin_y + 1, height - margin_y),
                        left + margin_x : max(left + margin_x + 1, right - margin_x),
                    ]
                    terminal = classify_foul_terminal(raw_cell)
                    if terminal is not None:
                        relative = (
                            Path("foul_terminals")
                            / split
                            / document_id
                            / f"{side}-row{row_index + 1:02d}-slot{slot_index + 1}.png"
                        )
                        _save_crop(raw_cell, crops_root / relative)
                        terminal_item = _terminal_crop_item(
                            document_id=document_id,
                            writer_id=writer_id,
                            relative=relative,
                            split=split,
                            side=side,
                            jersey=jersey,
                            row=row_index + 1,
                            slot=slot_index + 1,
                            predicted_label=(
                                "F"
                                if terminal.kind.value == "disqualification"
                                else "stroke"
                            ),
                        )
                        terminal_item["ink_color"] = terminal.ink_color.value
                        manifest.append(
                            _review_metadata(
                                terminal_item,
                                raw_cell,
                                source_image_hash=str(document["sha256"]),
                            )
                        )
                        break
                    cell_crop = clean_player_foul_cell(raw_cell)
                    red_mask, blue_mask = colored_ink_masks(cell_crop)
                    if float((red_mask | blue_mask).mean()) < grid.min_ink_ratio:
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
                    item = {
                        "document_id": document_id,
                        "writer_id": writer_id,
                        "field_type": "foul_event",
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
                        "grid_aligned": True,
                        "model_trainable": True,
                        "status": "labeled" if label is not None else "unlabeled",
                    }
                    manifest.append(
                        _review_metadata(
                            item,
                            cell_crop,
                            source_image_hash=str(document["sha256"]),
                        )
                    )
        processed_documents += 1

    validate_split_isolation(manifest)
    crops_root.mkdir(parents=True, exist_ok=True)
    manifest_path = crops_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    automatic_labeled_ids = {
        str(item["crop_id"])
        for item in manifest
        if item["status"] == "labeled" and item.get("crop_id") is not None
    }
    reviewed_labeled_ids = _reviewed_trainable_ids(crops_root, manifest)
    labeled_ids = automatic_labeled_ids | reviewed_labeled_ids
    summary = {
        "documents": processed_documents,
        "crops": len(manifest),
        "labeled": len(labeled_ids),
        "automatic_labeled": len(automatic_labeled_ids),
        "reviewed_labeled": len(reviewed_labeled_ids),
        "unlabeled": len(manifest) - len(labeled_ids),
        "split_isolation": "ok",
        "manifest": str(manifest_path),
    }
    (crops_root / "build-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
