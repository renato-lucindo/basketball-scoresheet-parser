from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any
from urllib.parse import quote

import numpy as np
from PIL import Image

from .imaging import load_document, normalize_document
from .recognition import FOUL_LABELS
from .template import FECABA_V1, NormalizedRect


GEOMETRY_REGIONS = (
    "scoring_table",
    "team_a_player_fouls",
    "team_b_player_fouls",
    "team_a_team_fouls",
    "team_b_team_fouls",
    "team_a_roster",
    "team_b_roster",
    "period_scores",
    "final_score",
    "apontador",
)
REVIEW_STAGES = ("geometry", "scoring", "jerseys", "fouls")
CROP_DECISIONS = {
    "accepted",
    "adjusted",
    "rejected_empty",
    "rejected_grid",
    "rejected_wrong_crop",
    "erasure",
    "illegible",
}
TRAINABLE_REVIEW_STATES = {"accepted", "adjusted"}


def _stage_field_types(stage: str) -> tuple[str, ...]:
    if stage in {"scoring", "jerseys"}:
        return ("scoring_event", "jersey")
    if stage == "fouls":
        return ("foul_event", "foul_terminal", "foul_symbol")
    return tuple()


def stable_crop_id(item: dict[str, Any]) -> str:
    field_type = str(item.get("field_type") or "")
    document_id = str(item.get("document_id") or "")
    if field_type in {"jersey", "scoring_event"}:
        position = f"{item.get('team')}:{item.get('running_score')}"
    elif field_type in {"foul_symbol", "foul_event", "foul_terminal"}:
        position = f"{item.get('team')}:{item.get('row')}:{item.get('slot')}"
    else:
        position = str(item.get("crop_path") or "")
    payload = f"{document_id}|{field_type}|{position}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def prepare_review(
    dataset_root: str | Path,
    *,
    stage: str,
    manifest: str | Path | None = None,
    dpi: int = 250,
    audit_only: bool = False,
) -> dict[str, Any]:
    root = Path(dataset_root)
    if stage not in REVIEW_STAGES:
        raise ValueError(f"Invalid review stage: {stage}")
    review_root = root / "review"
    review_root.mkdir(parents=True, exist_ok=True)
    if stage == "geometry":
        if audit_only:
            raise ValueError("--audit-only is supported only for jerseys or fouls")
        tasks = _prepare_geometry(root, review_root, dpi=dpi)
        config = _geometry_labeling_config()
    else:
        manifest_path = (
            Path(manifest)
            if manifest is not None
            else root
            / "crops"
            / ("manifest.reviewed.jsonl" if audit_only else "manifest.jsonl")
        )
        tasks = _prepare_crops(
            root,
            review_root,
            stage=stage,
            manifest=manifest_path,
            audit_only=audit_only,
        )
        config = _crop_labeling_config(stage)

    suffix = ".audit" if audit_only else ""
    tasks_path = review_root / f"{stage}{suffix}.tasks.json"
    config_path = review_root / f"{stage}{suffix}.labeling.xml"
    tasks_path.write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    config_path.write_text(config.rstrip() + "\n", encoding="utf-8")
    summary = {
        "stage": stage,
        "audit_only": audit_only,
        "tasks": len(tasks),
        "tasks_file": str(tasks_path),
        "labeling_config": str(config_path),
        "media_root": str(review_root / "media"),
    }
    (review_root / f"{stage}{suffix}.prepare-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def import_review(
    dataset_root: str | Path,
    export_path: str | Path,
    *,
    stage: str,
    manifest: str | Path | None = None,
    audit_only: bool = False,
) -> dict[str, Any]:
    root = Path(dataset_root)
    if stage not in REVIEW_STAGES:
        raise ValueError(f"Invalid review stage: {stage}")
    exported = json.loads(Path(export_path).read_text(encoding="utf-8"))
    if not isinstance(exported, list):
        raise ValueError("The Label Studio export must be a task list")
    if stage == "geometry":
        if audit_only:
            raise ValueError("--audit-only is supported only for jerseys or fouls")
        return _import_geometry(root, exported)
    if audit_only:
        reviewed_manifest = (
            Path(manifest)
            if manifest is not None
            else root / "crops" / "manifest.reviewed.jsonl"
        )
        return _import_audit(root, exported, stage=stage, manifest=reviewed_manifest)
    manifest_path = (
        Path(manifest) if manifest is not None else root / "crops" / "manifest.jsonl"
    )
    return _import_crops(root, exported, stage=stage, manifest=manifest_path)


def review_status(dataset_root: str | Path) -> dict[str, Any]:
    root = Path(dataset_root)
    catalog = _catalog_by_id(root)
    pilot_documents = [
        record for record in catalog.values() if bool(record.get("selected_pilot"))
    ]
    geometry = _load_json(root / "review" / "geometry.reviewed.json", default={})
    documents = geometry.get("documents", {}) if isinstance(geometry, dict) else {}
    reviewed_regions = 0
    unusable_regions = 0
    for document in pilot_documents:
        entry = documents.get(document["document_id"], {})
        regions = entry.get("regions", {}) if isinstance(entry, dict) else {}
        reviewed_regions += sum(name in regions for name in GEOMETRY_REGIONS)
        unusable_regions += sum(
            isinstance(regions.get(name), dict)
            and regions[name].get("state") == "unusable"
            for name in GEOMETRY_REGIONS
        )
    expected_regions = len(pilot_documents) * len(GEOMETRY_REGIONS)

    manifest_path = root / "crops" / "manifest.jsonl"
    reviewed_path = root / "crops" / "manifest.reviewed.jsonl"
    automatic = _read_jsonl(manifest_path) if manifest_path.exists() else []
    reviewed = _read_jsonl(reviewed_path) if reviewed_path.exists() else []
    reviewed_by_id = {
        str(item.get("crop_id")): item
        for item in reviewed
        if item.get("crop_id") is not None
    }
    crops: dict[str, Any] = {}
    for stage, field_types in (
        ("jerseys", ("jersey", "scoring_event")),
        ("fouls", ("foul_symbol", "foul_event", "foul_terminal")),
    ):
        candidates = [item for item in automatic if item.get("field_type") in field_types]
        states: dict[str, int] = {}
        valid = 0
        audit_required = 0
        audit_completed = 0
        for item in candidates:
            crop_id = str(item.get("crop_id") or stable_crop_id(item))
            reviewed_item = reviewed_by_id.get(crop_id)
            stale = False
            if reviewed_item:
                reviewed_source_hash = reviewed_item.get("source_image_hash")
                automatic_source_hash = item.get("source_image_hash")
                reviewed_crop_hash = reviewed_item.get("crop_image_hash")
                automatic_crop_hash = item.get("crop_image_hash")
                stale = bool(
                    reviewed_source_hash is not None
                    and automatic_source_hash is not None
                    and reviewed_source_hash != automatic_source_hash
                ) or bool(
                    reviewed_crop_hash is not None
                    and automatic_crop_hash is not None
                    and reviewed_crop_hash != automatic_crop_hash
                )
            state = "stale" if stale else (
                str(reviewed_item.get("review_state"))
                if reviewed_item and reviewed_item.get("review_state")
                else "pending"
            )
            states[state] = states.get(state, 0) + 1
            if reviewed_item and not stale and _is_trainable_review(reviewed_item):
                valid += 1
                if bool(reviewed_item.get("audit_required")):
                    audit_required += 1
                    audit_completed += int(
                        reviewed_item.get("audit_state") in {"passed", "corrected"}
                    )
        decided = len(candidates) - states.get("pending", 0) - states.get("stale", 0)
        geometrically_correct = sum(
            states.get(state, 0)
            for state in ("accepted", "adjusted", "erasure", "illegible")
        )
        geometric_rate = geometrically_correct / decided if decided else 0.0
        crops[stage] = {
            "candidates": len(candidates),
            "decided": decided,
            "pending": states.get("pending", 0),
            "stale": states.get("stale", 0),
            "completion": decided / len(candidates) if candidates else 1.0,
            "trainable": valid,
            "states": states,
            "audit_required": audit_required,
            "audit_completed": audit_completed,
            "audit_pending": audit_required - audit_completed,
            "geometrically_correct": geometrically_correct,
            "geometric_correct_rate": geometric_rate,
            "geometric_quality_met": (
                decided == len(candidates)
                and states.get("stale", 0) == 0
                and geometric_rate >= 0.98
            ),
        }
    crops["scoring"] = dict(crops["jerseys"])
    return {
        "geometry": {
            "documents": len(pilot_documents),
            "expected_regions": expected_regions,
            "reviewed_regions": reviewed_regions,
            "pending_regions": max(0, expected_regions - reviewed_regions),
            "completion": reviewed_regions / expected_regions if expected_regions else 1.0,
            "unusable_regions": unusable_regions,
        },
        "crops": crops,
        "ready_for_training": all(
            stage["pending"] == 0
            and stage["stale"] == 0
            and stage["trainable"] > 0
            and stage["audit_pending"] == 0
            and stage["geometric_quality_met"]
            for stage in crops.values()
        ),
        "reviewed_manifest": str(reviewed_path),
    }


def _prepare_geometry(root: Path, review_root: Path, *, dpi: int) -> list[dict[str, Any]]:
    catalog = _catalog_by_id(root)
    media_root = review_root / "media" / "geometry"
    media_root.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []
    for record in sorted(catalog.values(), key=lambda item: item["document_id"]):
        if not record.get("selected_pilot"):
            continue
        document_id = str(record["document_id"])
        normalized = normalize_document(
            load_document(root / str(record["source_path"]), dpi=dpi), FECABA_V1
        )
        media_path = media_root / f"{document_id}.png"
        Image.fromarray(np.asarray(normalized.image, dtype=np.uint8)).save(media_path)
        height, width = normalized.image.shape[:2]
        results: list[dict[str, Any]] = []
        for region_name in GEOMETRY_REGIONS:
            rect = FECABA_V1.region(region_name)
            result_id = _result_id(document_id, region_name)
            results.append(
                _rectangle_result(
                    result_id,
                    from_name="region",
                    label=region_name,
                    rect=(rect.x, rect.y, rect.width, rect.height),
                    width=width,
                    height=height,
                )
            )
            results.append(
                {
                    "id": result_id,
                    "type": "choices",
                    "from_name": "geometry_state",
                    "to_name": "image",
                    "value": {"choices": ["usable"]},
                }
            )
        tasks.append(
            {
                "data": {
                    "image": _media_url(media_path.relative_to(root)),
                    "document_id": document_id,
                    "source_image_hash": record["sha256"],
                    "image_width": width,
                    "image_height": height,
                },
                "predictions": [
                    {
                        "model_version": f"template:{FECABA_V1.template_id}",
                        "score": 1.0,
                        "result": results,
                    }
                ],
            }
        )
    return tasks


def _prepare_crops(
    root: Path,
    review_root: Path,
    *,
    stage: str,
    manifest: Path,
    audit_only: bool,
) -> list[dict[str, Any]]:
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    field_types = _stage_field_types(stage)
    catalog = _catalog_by_id(root)
    media_root = review_root / "media" / stage
    media_root.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in _read_jsonl(manifest):
        if item.get("field_type") not in field_types:
            continue
        if audit_only and not (
            item.get("review_state") in TRAINABLE_REVIEW_STATES
            and bool(item.get("audit_required"))
            and item.get("audit_state") not in {"passed", "corrected"}
        ):
            continue
        crop_id = str(item.get("crop_id") or stable_crop_id(item))
        if crop_id in seen:
            raise ValueError(f"Duplicate crop_id in manifest: {crop_id}")
        seen.add(crop_id)
        source_crop = manifest.parent / str(item["crop_path"])
        if not source_crop.exists():
            raise FileNotFoundError(source_crop)
        with Image.open(source_crop) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            actual_crop_hash = hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()
        media_path = media_root / f"{crop_id}{source_crop.suffix.lower() or '.png'}"
        shutil.copyfile(source_crop, media_path)
        document_id = str(item["document_id"])
        catalog_hash = str(catalog[document_id]["sha256"])
        source_hash = str(item.get("source_image_hash") or catalog_hash)
        if source_hash != catalog_hash:
            raise ValueError(f"Hash de origem divergente no crop {crop_id}")
        expected_crop_hash = item.get("crop_image_hash")
        if expected_crop_hash is not None and str(expected_crop_hash) != actual_crop_hash:
            raise ValueError(f"Crop {crop_id} diverge do manifesto automatico")
        original_bbox = (
            item.get("bbox_revised")
            if audit_only and item.get("bbox_revised") is not None
            else item.get("bbox_original")
        ) or [0, 0, width, height]
        normalized_bbox = _pixel_bbox_to_normalized(original_bbox, width, height)
        result_id = _result_id(crop_id, "candidate")
        results = [
            _rectangle_result(
                result_id,
                from_name="bbox",
                label="candidate",
                rect=normalized_bbox,
                width=width,
                height=height,
            )
        ]
        original_label = item.get("label")
        if original_label is not None:
            results.extend(
                [
                    {
                        "id": result_id,
                        "type": "textarea",
                        "from_name": "final_label",
                        "to_name": "image",
                        "value": {"text": [str(original_label)]},
                    },
                    {
                        "id": result_id,
                        "type": "choices",
                        "from_name": "decision",
                        "to_name": "image",
                        "value": {"choices": ["accepted"]},
                    },
                ]
            )
        tasks.append(
            {
                "data": {
                    "image": _media_url(media_path.relative_to(root)),
                    "crop_id": crop_id,
                    "document_id": document_id,
                    "writer_id": item.get("writer_id"),
                    "field_type": item.get("field_type"),
                    "source_image_hash": source_hash,
                    "crop_image_hash": actual_crop_hash,
                    "original_label": original_label,
                    "team": item.get("team"),
                    "period": item.get("period"),
                    "predicted_type": item.get("predicted_type"),
                    "predicted_label": item.get("predicted_label"),
                    "position": item.get("position") or _position_text(item),
                    "audit_only": audit_only,
                    "image_width": width,
                    "image_height": height,
                },
                "predictions": [
                    {"model_version": "dataset-build", "score": 1.0, "result": results}
                ],
            }
        )
    return tasks


def _import_geometry(root: Path, exported: list[dict[str, Any]]) -> dict[str, Any]:
    catalog = _catalog_by_id(root)
    target = root / "review" / "geometry.reviewed.json"
    current = _load_json(target, default={"version": 1, "documents": {}})
    if not isinstance(current, dict):
        current = {"version": 1, "documents": {}}
    documents = current.setdefault("documents", {})
    seen_documents: set[str] = set()
    imported_regions = 0
    pending_tasks = 0
    for task in exported:
        data = _task_data(task)
        document_id = _resolve_geometry_document(root, task, data, catalog)
        if document_id in seen_documents:
            raise ValueError(f"Duplicate document in export: {document_id}")
        seen_documents.add(document_id)
        annotation = _latest_annotation(task)
        if annotation is None:
            pending_tasks += 1
            continue
        results = annotation.get("result", [])
        choices = _choices_by_region(results, from_name="geometry_state")
        annotation_regions: dict[str, Any] = {}
        for result in results:
            if result.get("type") != "rectanglelabels" or result.get("from_name") != "region":
                continue
            value = result.get("value", {})
            labels = value.get("rectanglelabels", [])
            if not isinstance(labels, list) or len(labels) != 1:
                raise ValueError(f"Region without a unique label in document {document_id}")
            region_name = str(labels[0])
            if region_name not in GEOMETRY_REGIONS:
                raise ValueError(f"Unknown region: {region_name}")
            if region_name in annotation_regions:
                raise ValueError(f"Duplicate region in {document_id}: {region_name}")
            revised = _normalized_bbox_from_result(result)
            state_values = choices.get(str(result.get("id")), ["usable"])
            if len(state_values) != 1 or state_values[0] not in {"usable", "unusable"}:
                raise ValueError(
                    f"Invalid geometry state in {document_id}/{region_name}"
                )
            original = FECABA_V1.region(region_name)
            annotation_regions[region_name] = {
                "state": state_values[0],
                "bbox_original": [original.x, original.y, original.width, original.height],
                "bbox_revised": list(revised),
            }
        if not annotation_regions:
            pending_tasks += 1
            continue
        previous = documents.get(document_id, {})
        previous_regions = (
            dict(previous.get("regions", {})) if isinstance(previous, dict) else {}
        )
        previous_regions.update(annotation_regions)
        documents[document_id] = {
            "source_image_hash": catalog[document_id]["sha256"],
            "regions": previous_regions,
        }
        imported_regions += len(annotation_regions)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "stage": "geometry",
        "tasks": len(exported),
        "imported_regions": imported_regions,
        "pending_tasks": pending_tasks,
        "geometry_file": str(target),
    }


def _import_crops(
    root: Path,
    exported: list[dict[str, Any]],
    *,
    stage: str,
    manifest: Path,
) -> dict[str, Any]:
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    catalog = _catalog_by_id(root)
    automatic = _read_jsonl(manifest)
    base_by_id: dict[str, dict[str, Any]] = {}
    for item in automatic:
        crop_id = str(item.get("crop_id") or stable_crop_id(item))
        if crop_id in base_by_id:
            raise ValueError(f"Duplicate crop_id in manifest: {crop_id}")
        enriched = dict(item)
        enriched["crop_id"] = crop_id
        enriched.setdefault("source_image_hash", catalog[str(item["document_id"])]["sha256"])
        base_by_id[crop_id] = enriched
    reviewed_path = manifest.parent / "manifest.reviewed.jsonl"
    reviewed_by_id = {
        str(item["crop_id"]): item
        for item in (_read_jsonl(reviewed_path) if reviewed_path.exists() else [])
        if item.get("crop_id")
    }
    seen: set[str] = set()
    imported = 0
    pending = 0
    stage_fields = _stage_field_types(stage)
    for task in exported:
        data = _task_data(task)
        crop_id = str(data.get("crop_id") or "")
        if not crop_id or crop_id not in base_by_id:
            raise ValueError(f"Unknown crop_id in export: {crop_id!r}")
        if crop_id in seen:
            raise ValueError(f"Duplicate crop_id in export: {crop_id}")
        seen.add(crop_id)
        base = base_by_id[crop_id]
        expected_field = str(base.get("field_type") or "")
        if expected_field not in stage_fields:
            raise ValueError(f"Crop {crop_id} does not belong to stage {stage}")
        document = catalog[str(base["document_id"])]
        _validate_source_hash(data, document)
        if str(base.get("source_image_hash")) != str(document["sha256"]):
            raise ValueError(f"Scoresheet version mismatch for {crop_id}")
        actual_crop_hash = _crop_file_hash(manifest.parent / str(base["crop_path"]))
        expected_crop_hash = str(base.get("crop_image_hash") or actual_crop_hash)
        if str(data.get("crop_image_hash") or "") != expected_crop_hash:
            raise ValueError(f"Crop {crop_id} belongs to a different crop version")
        if actual_crop_hash != expected_crop_hash:
            raise ValueError(f"Crop {crop_id} was regenerated after review preparation")
        base["crop_image_hash"] = expected_crop_hash
        annotation = _latest_annotation(task)
        if annotation is None:
            pending += 1
            continue
        result = _parse_crop_annotation(annotation.get("result", []), crop_id=crop_id)
        decision = result["decision"]
        if decision is None:
            pending += 1
            continue
        label_final = result["label"]
        if decision in TRAINABLE_REVIEW_STATES:
            if base.get("model_trainable") is False:
                label_final = str(base.get("predicted_label") or label_final or "")
            else:
                label_final = _validate_final_label(expected_field, label_final, crop_id)
        width = int(data.get("image_width") or 0)
        height = int(data.get("image_height") or 0)
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions for crop {crop_id}")
        revised_bbox = _normalized_to_pixel_bbox(result["bbox"], width, height)
        original_bbox = base.get("bbox_original") or [0, 0, width, height]
        _validate_pixel_bbox(original_bbox, width, height, crop_id)
        _validate_pixel_bbox(revised_bbox, width, height, crop_id)
        original_label = base.get("label")
        if decision == "accepted" and (
            [int(value) for value in original_bbox] != revised_bbox
            or str(original_label) != str(label_final)
        ):
            decision = "adjusted"
        reviewed = dict(base)
        reviewed.update(
            {
                "crop_id": crop_id,
                "bbox_original": [int(value) for value in original_bbox],
                "bbox_revised": revised_bbox,
                "label_original": original_label,
                "label_final": label_final if decision in TRAINABLE_REVIEW_STATES else None,
                "label": label_final if decision in TRAINABLE_REVIEW_STATES else None,
                "review_state": decision,
                "status": "labeled" if decision in TRAINABLE_REVIEW_STATES else "excluded",
                "audit_required": False,
                "audit_state": None,
            }
        )
        reviewed_by_id[crop_id] = reviewed
        imported += 1

    merged: list[dict[str, Any]] = []
    for crop_id, base in base_by_id.items():
        existing = reviewed_by_id.get(crop_id)
        if existing is None:
            existing = dict(base)
            existing.update(
                {
                    "bbox_original": existing.get("bbox_original"),
                    "bbox_revised": None,
                    "label_original": existing.get("label"),
                    "label_final": None,
                    "review_state": "pending",
                    "audit_required": False,
                }
            )
        merged.append(existing)
    for field_type in stage_fields:
        _assign_audit_sample(merged, field_type)
    _write_jsonl(reviewed_path, merged)
    trainable = sum(_is_trainable_review(item) for item in merged)
    return {
        "stage": stage,
        "tasks": len(exported),
        "imported": imported,
        "pending_tasks": pending,
        "trainable": trainable,
        "reviewed_manifest": str(reviewed_path),
    }


def _import_audit(
    root: Path,
    exported: list[dict[str, Any]],
    *,
    stage: str,
    manifest: Path,
) -> dict[str, Any]:
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    catalog = _catalog_by_id(root)
    items = _read_jsonl(manifest)
    by_id = {
        str(item["crop_id"]): item
        for item in items
        if item.get("crop_id") is not None
    }
    stage_fields = _stage_field_types(stage)
    seen: set[str] = set()
    completed = 0
    corrected = 0
    pending = 0
    for task in exported:
        data = _task_data(task)
        crop_id = str(data.get("crop_id") or "")
        if not crop_id or crop_id not in by_id:
            raise ValueError(f"Unknown crop_id in audit export: {crop_id!r}")
        if crop_id in seen:
            raise ValueError(f"Duplicate crop_id in audit export: {crop_id}")
        seen.add(crop_id)
        item = by_id[crop_id]
        expected_field = str(item.get("field_type") or "")
        if expected_field not in stage_fields:
            raise ValueError(f"Crop {crop_id} does not belong to stage {stage}")
        if not bool(item.get("audit_required")):
            raise ValueError(f"Crop {crop_id} does not belong to the audit sample")
        document = catalog[str(item["document_id"])]
        _validate_source_hash(data, document)
        actual_crop_hash = _crop_file_hash(manifest.parent / str(item["crop_path"]))
        expected_crop_hash = str(item.get("crop_image_hash") or actual_crop_hash)
        if str(data.get("crop_image_hash") or "") != expected_crop_hash:
            raise ValueError(f"Crop {crop_id} belongs to a different crop version")
        if actual_crop_hash != expected_crop_hash:
            raise ValueError(f"Crop {crop_id} was regenerated after the first review")
        annotation = _latest_annotation(task)
        if annotation is None:
            pending += 1
            continue
        result = _parse_crop_annotation(annotation.get("result", []), crop_id=crop_id)
        decision = result["decision"]
        if decision is None:
            pending += 1
            continue
        label_final = result["label"]
        if decision in TRAINABLE_REVIEW_STATES:
            label_final = _validate_final_label(expected_field, label_final, crop_id)
        width = int(data.get("image_width") or 0)
        height = int(data.get("image_height") or 0)
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions for crop {crop_id}")
        revised_bbox = _normalized_to_pixel_bbox(result["bbox"], width, height)
        first_bbox = item.get("bbox_revised") or item.get("bbox_original")
        if first_bbox is None:
            first_bbox = [0, 0, width, height]
        _validate_pixel_bbox(first_bbox, width, height, crop_id)
        first_bbox_values = [int(value) for value in first_bbox]
        first_label = item.get("label")
        changed = (
            first_bbox_values != revised_bbox
            or str(first_label) != str(label_final)
            or decision not in TRAINABLE_REVIEW_STATES
        )
        if decision == "accepted" and changed:
            decision = "adjusted"
        item["bbox_revised"] = revised_bbox
        item["label_final"] = (
            label_final if decision in TRAINABLE_REVIEW_STATES else None
        )
        item["label"] = label_final if decision in TRAINABLE_REVIEW_STATES else None
        item["review_state"] = decision
        item["status"] = "labeled" if decision in TRAINABLE_REVIEW_STATES else "excluded"
        item["audit_state"] = "corrected" if changed else "passed"
        completed += 1
        corrected += int(changed)
    _write_jsonl(manifest, items)
    return {
        "stage": stage,
        "audit_only": True,
        "tasks": len(exported),
        "completed": completed,
        "corrected": corrected,
        "pending_tasks": pending,
        "reviewed_manifest": str(manifest),
    }


def _assign_audit_sample(items: list[dict[str, Any]], field_type: str) -> None:
    candidates = [
        item
        for item in items
        if item.get("field_type") == field_type and _is_trainable_review(item)
    ]
    sample_size = (len(candidates) + 9) // 10
    ranked = sorted(
        candidates,
        key=lambda item: hashlib.sha1(str(item["crop_id"]).encode("utf-8")).hexdigest(),
    )
    selected = {str(item["crop_id"]) for item in ranked[:sample_size]}
    for item in items:
        if item.get("field_type") != field_type:
            continue
        crop_id = str(item.get("crop_id") or "")
        required = crop_id in selected
        previous_required = bool(item.get("audit_required"))
        previous_state = item.get("audit_state")
        item["audit_required"] = required
        if not required:
            item["audit_state"] = None
        elif previous_required and previous_state in {"passed", "corrected"}:
            item["audit_state"] = previous_state
        else:
            item["audit_state"] = "pending"


def _parse_crop_annotation(results: list[dict[str, Any]], *, crop_id: str) -> dict[str, Any]:
    boxes = [
        result
        for result in results
        if result.get("type") == "rectanglelabels" and result.get("from_name") == "bbox"
    ]
    if len(boxes) != 1:
        raise ValueError(f"Crop {crop_id} must contain exactly one bounding box")
    box = boxes[0]
    region_id = str(box.get("id") or "")
    bbox = _normalized_bbox_from_result(box)
    choices = _choices_by_region(results, from_name="decision").get(region_id, [])
    if len(choices) > 1:
        raise ValueError(f"Crop {crop_id} possui mais de uma decisao")
    decision = str(choices[0]) if choices else None
    if decision is not None and decision not in CROP_DECISIONS:
        raise ValueError(f"Invalid decision for {crop_id}: {decision}")
    texts = [
        result
        for result in results
        if str(result.get("id") or "") == region_id
        and result.get("type") == "textarea"
        and result.get("from_name") == "final_label"
    ]
    if len(texts) > 1:
        raise ValueError(f"Crop {crop_id} has more than one final label")
    label: str | None = None
    if texts:
        values = texts[0].get("value", {}).get("text", [])
        if isinstance(values, list) and values:
            label = str(values[0]).strip() or None
    return {"bbox": bbox, "decision": decision, "label": label}


def _validate_final_label(field_type: str, label: str | None, crop_id: str) -> str:
    if label is None:
        raise ValueError(f"Accepted or adjusted crop {crop_id} requires a label")
    if field_type in {"jersey", "scoring_event"}:
        label = label.strip()
        if not label.isdigit() or not 0 <= int(label) <= 99:
            raise ValueError(f"Invalid jersey number in {crop_id}: {label!r}")
        return label
    normalized = label.upper().replace(" ", "")
    canonical = {value.upper(): value for value in FOUL_LABELS}
    if normalized not in canonical:
        allowed = ", ".join(FOUL_LABELS)
        raise ValueError(f"Invalid foul in {crop_id}: {label!r}. Allowed: {allowed}")
    return canonical[normalized]


def _catalog_by_id(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "catalog.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    records = _read_jsonl(path)
    return {str(record["document_id"]): record for record in records}


def _task_data(task: dict[str, Any]) -> dict[str, Any]:
    data = task.get("data")
    if not isinstance(data, dict):
        raise ValueError("Label Studio task is missing its data object")
    return data


def _resolve_geometry_document(
    root: Path,
    task: dict[str, Any],
    data: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
) -> str:
    document_id = str(data.get("document_id") or "")
    if document_id:
        if document_id not in catalog:
            raise ValueError(f"Unknown document in export: {document_id!r}")
        _validate_source_hash(data, catalog[document_id])
        return document_id

    upload_name = Path(str(task.get("file_upload") or "")).name
    if not upload_name:
        raise ValueError("Geometry export has neither document_id nor file_upload")

    candidates = [
        candidate_id
        for candidate_id, record in catalog.items()
        if bool(record.get("selected_pilot"))
        and (
            upload_name == f"{candidate_id}.png"
            or upload_name.endswith(f"-{candidate_id}.png")
        )
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Scoresheet could not be identified uniquely from file_upload: {upload_name!r}"
        )
    document_id = candidates[0]

    expected_media = root / "review" / "media" / "geometry" / f"{document_id}.png"
    if not expected_media.exists():
        raise FileNotFoundError(expected_media)
    upload_root = root / "review" / "label-studio-data" / "media" / "upload"
    uploaded = list(upload_root.rglob(upload_name)) if upload_root.exists() else []
    if len(uploaded) != 1:
        raise ValueError(
            f"Label Studio upload could not be resolved uniquely: {upload_name!r}"
        )
    expected_hash = hashlib.sha256(expected_media.read_bytes()).hexdigest()
    uploaded_hash = hashlib.sha256(uploaded[0].read_bytes()).hexdigest()
    if uploaded_hash != expected_hash:
        raise ValueError(f"Label Studio image differs from {document_id} geometry")

    return document_id


def _validate_source_hash(data: dict[str, Any], catalog_record: dict[str, Any]) -> None:
    expected = str(catalog_record["sha256"])
    received = str(data.get("source_image_hash") or "")
    if received != expected:
        raise ValueError(
            f"Scoresheet version mismatch for {catalog_record['document_id']}: "
            f"esperado {expected[:12]}, recebido {received[:12] or '<vazio>'}"
        )


def _latest_annotation(task: dict[str, Any]) -> dict[str, Any] | None:
    annotations = task.get("annotations")
    if not isinstance(annotations, list):
        return None
    valid = [
        item
        for item in annotations
        if isinstance(item, dict) and not bool(item.get("was_cancelled"))
    ]
    return valid[-1] if valid else None


def _choices_by_region(
    results: list[dict[str, Any]], *, from_name: str
) -> dict[str, list[str]]:
    choices: dict[str, list[str]] = {}
    for result in results:
        if result.get("type") != "choices" or result.get("from_name") != from_name:
            continue
        region_id = str(result.get("id") or "")
        values = result.get("value", {}).get("choices", [])
        if not isinstance(values, list):
            raise ValueError(f"Invalid choices value in {from_name}")
        if region_id in choices:
            raise ValueError(f"Duplicate choices value for region {region_id}")
        choices[region_id] = [str(value) for value in values]
    return choices


def _rectangle_result(
    result_id: str,
    *,
    from_name: str,
    label: str,
    rect: tuple[float, float, float, float],
    width: int,
    height: int,
) -> dict[str, Any]:
    x, y, rect_width, rect_height = rect
    _validate_normalized_bbox(rect)
    return {
        "id": result_id,
        "type": "rectanglelabels",
        "from_name": from_name,
        "to_name": "image",
        "original_width": width,
        "original_height": height,
        "image_rotation": 0,
        "value": {
            "x": x * 100,
            "y": y * 100,
            "width": rect_width * 100,
            "height": rect_height * 100,
            "rotation": 0,
            "rectanglelabels": [label],
        },
    }


def _normalized_bbox_from_result(result: dict[str, Any]) -> tuple[float, float, float, float]:
    value = result.get("value", {})
    try:
        x = float(value["x"]) / 100.0
        y = float(value["y"]) / 100.0
        width = float(value["width"]) / 100.0
        height = float(value["height"]) / 100.0
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Incomplete Label Studio bounding box") from exc
    epsilon = 1e-9
    if -epsilon <= x < 0:
        x = 0.0
    if -epsilon <= y < 0:
        y = 0.0
    if 1.0 < x + width <= 1.0 + epsilon:
        width = 1.0 - x
    if 1.0 < y + height <= 1.0 + epsilon:
        height = 1.0 - y
    rect = (x, y, width, height)
    _validate_normalized_bbox(rect)
    return rect


def _validate_normalized_bbox(rect: tuple[float, float, float, float]) -> None:
    x, y, width, height = rect
    if width <= 0 or height <= 0:
        raise ValueError("Bounding box width and height must be positive")
    if x < 0 or y < 0 or x + width > 1.000001 or y + height > 1.000001:
        raise ValueError("Caixa ultrapassa os limites da imagem")


def _pixel_bbox_to_normalized(
    bbox: Any, width: int, height: int
) -> tuple[float, float, float, float]:
    _validate_pixel_bbox(bbox, width, height, "manifest")
    left, top, right, bottom = [int(value) for value in bbox]
    return left / width, top / height, (right - left) / width, (bottom - top) / height


def _normalized_to_pixel_bbox(
    rect: tuple[float, float, float, float], width: int, height: int
) -> list[int]:
    x, y, rect_width, rect_height = rect
    left = round(x * width)
    top = round(y * height)
    right = round((x + rect_width) * width)
    bottom = round((y + rect_height) * height)
    result = [left, top, right, bottom]
    _validate_pixel_bbox(result, width, height, "annotation")
    return result


def _validate_pixel_bbox(bbox: Any, width: int, height: int, crop_id: str) -> None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"Invalid bounding box in {crop_id}")
    try:
        left, top, right, bottom = [int(value) for value in bbox]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid bounding box in {crop_id}") from exc
    if left < 0 or top < 0 or right > width or bottom > height:
        raise ValueError(f"bbox fora da imagem em {crop_id}")
    if right <= left or bottom <= top:
        raise ValueError(f"bbox vazia em {crop_id}")


def _media_url(relative: Path) -> str:
    return "/data/local-files/?d=" + quote(relative.as_posix(), safe="/")


def _result_id(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


def _position_text(item: dict[str, Any]) -> str:
    if item.get("field_type") in {"jersey", "scoring_event"}:
        return f"score:{item.get('team')}:{item.get('running_score')}"
    return f"foul:{item.get('team')}:row{item.get('row')}:slot{item.get('slot')}"


def _is_trainable_review(item: dict[str, Any]) -> bool:
    return (
        item.get("model_trainable") is not False
        and item.get("review_state") in TRAINABLE_REVIEW_STATES
        and item.get("label") is not None
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _crop_file_hash(path: Path) -> str:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def _load_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_geometry_overrides(dataset_root: str | Path) -> dict[str, Any]:
    payload = _load_json(Path(dataset_root) / "review" / "geometry.reviewed.json", default={})
    if not isinstance(payload, dict):
        return {}
    documents = payload.get("documents", {})
    return documents if isinstance(documents, dict) else {}


def reviewed_region(
    overrides: dict[str, Any], document_id: str, region_name: str
) -> NormalizedRect | None:
    document = overrides.get(document_id)
    if not isinstance(document, dict):
        return FECABA_V1.region(region_name)
    regions = document.get("regions", {})
    entry = regions.get(region_name) if isinstance(regions, dict) else None
    if not isinstance(entry, dict):
        return FECABA_V1.region(region_name)
    if entry.get("state") == "unusable":
        return None
    bbox = entry.get("bbox_revised")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return FECABA_V1.region(region_name)
    rect = tuple(float(value) for value in bbox)
    _validate_normalized_bbox(rect)  # type: ignore[arg-type]
    return NormalizedRect(*rect)


def _geometry_labeling_config() -> str:
    labels = "\n".join(f'      <Label value="{name}" />' for name in GEOMETRY_REGIONS)
    return f"""<View>
  <Image name="image" value="$image" />
  <RectangleLabels name="region" toName="image">
{labels}
  </RectangleLabels>
  <Choices name="geometry_state" toName="image" perRegion="true" choice="single" showInline="true">
    <Choice value="usable" />
    <Choice value="unusable" />
  </Choices>
</View>"""


def _crop_labeling_config(stage: str) -> str:
    vocabulary = "0-99" if stage in {"jerseys", "scoring"} else ", ".join(FOUL_LABELS)
    return f"""<View>
  <Header value="Crop: $crop_id | Document: $document_id | Position: $position | Period: $period | Predicted: $predicted_type $predicted_label" />
  <Image name="image" value="$image" />
  <RectangleLabels name="bbox" toName="image">
    <Label value="candidate" />
  </RectangleLabels>
  <Choices name="decision" toName="image" perRegion="true" choice="single" showInline="true">
    <Choice value="accepted" />
    <Choice value="adjusted" />
    <Choice value="rejected_empty" />
    <Choice value="rejected_grid" />
    <Choice value="rejected_wrong_crop" />
    <Choice value="erasure" />
    <Choice value="illegible" />
  </Choices>
  <TextArea name="final_label" toName="image" perRegion="true" editable="true" rows="1" />
  <Header value="Rotulos permitidos: {vocabulary}" />
</View>"""
