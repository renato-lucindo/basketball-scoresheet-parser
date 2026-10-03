from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(slots=True)
class DatasetRecord:
    document_id: str
    writer_id: str | None
    field_type: str
    label: str
    crop_path: str
    team: str | None = None
    period: int | None = None
    split: str | None = None
    source_file: str | None = None
    running_score: int | None = None
    event_index: int | None = None


def assign_document_split(
    document_id: str,
    *,
    writer_id: str | None = None,
    seed: str = "sumula-reader-v1",
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> str:
    """Split by document or scorer, never by crop.

    When the scorer is known, that identity becomes the grouping unit so every
    scoresheet written by the same person remains in one split. Without a
    ``writer_id``, the historical document-based behavior is preserved.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train + validation must leave room for test")

    split_key = document_id if writer_id is None else f"writer:{writer_id}"
    digest = hashlib.sha256(f"{seed}:{split_key}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64)
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + validation_ratio:
        return "validation"
    return "test"


def validate_split_isolation(records: list[dict[str, object]]) -> None:
    """Fail if a document or scorer appears in more than one split."""
    document_splits: dict[str, set[str]] = {}
    writer_splits: dict[str, set[str]] = {}
    for record in records:
        split = record.get("split")
        if not isinstance(split, str):
            continue
        document_id = record.get("document_id")
        if isinstance(document_id, str):
            document_splits.setdefault(document_id, set()).add(split)
        writer_id = record.get("writer_id")
        if isinstance(writer_id, str) and writer_id:
            writer_splits.setdefault(writer_id, set()).add(split)

    leaked_documents = sorted(
        document_id
        for document_id, splits in document_splits.items()
        if len(splits) > 1
    )
    leaked_writers = sorted(
        writer_id for writer_id, splits in writer_splits.items() if len(splits) > 1
    )
    if leaked_documents or leaked_writers:
        details: list[str] = []
        if leaked_documents:
            details.append("documents=" + ",".join(leaked_documents))
        if leaked_writers:
            details.append("scorers=" + ",".join(leaked_writers))
        raise ValueError("Split leakage: " + "; ".join(details))


def save_annotated_crop(
    image: np.ndarray,
    *,
    root: str | Path,
    record: DatasetRecord,
) -> Path:
    """Save a crop and JSONL record without requiring the scorer's real name."""
    base = Path(root)
    crop_path = base / record.crop_path
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(crop_path)

    manifest = base / "manifest.jsonl"
    payload = asdict(record)
    payload["crop_path"] = crop_path.relative_to(base).as_posix()
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return crop_path
