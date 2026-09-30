from __future__ import annotations

from dataclasses import asdict, dataclass
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


def save_annotated_crop(
    image: np.ndarray,
    *,
    root: str | Path,
    record: DatasetRecord,
) -> Path:
    """Salva crop + JSONL sem exigir nome real do apontador."""
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
