from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw

from sumula_reader.label_studio_review import (
    GEOMETRY_REGIONS,
    import_review,
    load_geometry_overrides,
    prepare_review,
    review_status,
    reviewed_region,
    _validate_final_label,
)


def _write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")


def _make_dataset(root: Path) -> tuple[str, str]:
    document_id = "pilot-001"
    source = root / "source" / "sub15" / "pilot.png"
    source.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (900, 1300), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 880, 1280), outline="black", width=8)
    image.save(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    _write_jsonl(
        root / "catalog.jsonl",
        [
            {
                "document_id": document_id,
                "source_path": source.relative_to(root).as_posix(),
                "sha256": digest,
                "selected_pilot": True,
            }
        ],
    )
    return document_id, digest


class LabelStudioReviewTests(unittest.TestCase):
    def test_invalid_review_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid jersey number"):
            _validate_final_label("jersey", "100", "crop-1")
        with self.assertRaisesRegex(ValueError, "Invalid foul"):
            _validate_final_label("foul_symbol", "unknown", "crop-2")

    def test_geometry_prepare_and_import_create_reviewed_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document_id, _ = _make_dataset(root)

            summary = prepare_review(root, stage="geometry", dpi=100)

            self.assertEqual(summary["tasks"], 1)
            tasks = json.loads((root / "review" / "geometry.tasks.json").read_text())
            task = tasks[0]
            rectangles = [
                result
                for result in task["predictions"][0]["result"]
                if result["type"] == "rectanglelabels"
            ]
            self.assertEqual(len(rectangles), len(GEOMETRY_REGIONS))
            self.assertTrue(task["data"]["image"].startswith("/data/local-files/?d="))
            self.assertTrue(
                (root / "review" / "media" / "geometry" / f"{document_id}.png").exists()
            )

            export = [dict(task)]
            annotation_results = json.loads(json.dumps(task["predictions"][0]["result"]))
            scoring = next(
                result
                for result in annotation_results
                if result.get("type") == "rectanglelabels"
                and result["value"]["rectanglelabels"] == ["scoring_table"]
            )
            scoring["value"]["y"] += 0.25
            export[0]["annotations"] = [{"result": annotation_results}]
            export_path = root / "geometry-export.json"
            export_path.write_text(json.dumps(export), encoding="utf-8")

            imported = import_review(root, export_path, stage="geometry")

            self.assertEqual(imported["imported_regions"], len(GEOMETRY_REGIONS))
            overrides = load_geometry_overrides(root)
            revised = reviewed_region(overrides, document_id, "scoring_table")
            self.assertIsNotNone(revised)
            self.assertGreater(revised.y, 0.151)

    def test_crop_review_generates_separate_trainable_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document_id, digest = _make_dataset(root)
            crop = root / "crops" / "jerseys" / "train" / document_id / "A-002.png"
            crop.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (40, 30), "white").save(crop)
            _write_jsonl(
                root / "crops" / "manifest.jsonl",
                [
                    {
                        "document_id": document_id,
                        "writer_id": None,
                        "field_type": "jersey",
                        "label": None,
                        "crop_path": crop.relative_to(root / "crops").as_posix(),
                        "split": "train",
                        "team": "A",
                        "period": "Q1",
                        "running_score": 2,
                        "status": "unlabeled",
                        "source_image_hash": digest,
                    }
                ],
            )

            summary = prepare_review(root, stage="jerseys")
            self.assertEqual(summary["tasks"], 1)
            tasks = json.loads((root / "review" / "jerseys.tasks.json").read_text())
            task = tasks[0]
            results = json.loads(json.dumps(task["predictions"][0]["result"]))
            region_id = next(
                result["id"] for result in results if result["type"] == "rectanglelabels"
            )
            results.append(
                {
                    "id": region_id,
                    "type": "choices",
                    "from_name": "decision",
                    "to_name": "image",
                    "value": {"choices": ["accepted"]},
                }
            )
            results.append(
                {
                    "id": region_id,
                    "type": "textarea",
                    "from_name": "final_label",
                    "to_name": "image",
                    "value": {"text": ["12"]},
                }
            )
            task["annotations"] = [{"result": results}]
            export = root / "jerseys-export.json"
            export.write_text(json.dumps(tasks), encoding="utf-8")

            imported = import_review(root, export, stage="jerseys")

            self.assertEqual(imported["trainable"], 1)
            reviewed = [
                json.loads(line)
                for line in (root / "crops" / "manifest.reviewed.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(reviewed[0]["label"], "12")
            self.assertEqual(reviewed[0]["review_state"], "adjusted")
            self.assertEqual(reviewed[0]["bbox_revised"], [0, 0, 40, 30])

            status = review_status(root)
            self.assertEqual(status["crops"]["jerseys"]["pending"], 0)
            self.assertEqual(status["crops"]["jerseys"]["trainable"], 1)
            self.assertEqual(status["crops"]["jerseys"]["audit_required"], 1)
            self.assertEqual(status["crops"]["jerseys"]["audit_pending"], 1)
            self.assertTrue(status["crops"]["jerseys"]["geometric_quality_met"])

            audit_summary = prepare_review(root, stage="jerseys", audit_only=True)
            self.assertEqual(audit_summary["tasks"], 1)
            audit_tasks = json.loads(
                (root / "review" / "jerseys.audit.tasks.json").read_text()
            )
            audit_tasks[0]["annotations"] = [
                {"result": audit_tasks[0]["predictions"][0]["result"]}
            ]
            audit_export = root / "jerseys-audit-export.json"
            audit_export.write_text(json.dumps(audit_tasks), encoding="utf-8")
            audit_import = import_review(
                root,
                audit_export,
                stage="jerseys",
                audit_only=True,
            )
            self.assertEqual(audit_import["completed"], 1)
            self.assertEqual(audit_import["corrected"], 0)
            status = review_status(root)
            self.assertEqual(status["crops"]["jerseys"]["audit_pending"], 0)
            self.assertEqual(status["crops"]["jerseys"]["audit_completed"], 1)

    def test_scoring_stage_reviews_combined_grid_event(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document_id, digest = _make_dataset(root)
            crop = (
                root
                / "crops"
                / "scoring_events"
                / "train"
                / document_id
                / "A-002.png"
            )
            crop.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (80, 30), "white").save(crop)
            _write_jsonl(
                root / "crops" / "manifest.jsonl",
                [
                    {
                        "document_id": document_id,
                        "writer_id": None,
                        "field_type": "scoring_event",
                        "label": None,
                        "crop_path": crop.relative_to(root / "crops").as_posix(),
                        "split": "train",
                        "team": "A",
                        "period": "Q1",
                        "running_score": 2,
                        "predicted_type": "two_point",
                        "status": "unlabeled",
                        "review_state": "pending",
                        "bbox_original": [0, 0, 40, 30],
                        "source_image_hash": digest,
                    }
                ],
            )

            summary = prepare_review(root, stage="scoring")

            self.assertEqual(summary["tasks"], 1)
            task = json.loads((root / "review" / "scoring.tasks.json").read_text())[0]
            self.assertEqual(task["data"]["field_type"], "scoring_event")
            self.assertEqual(task["data"]["predicted_type"], "two_point")
            bbox = next(
                result
                for result in task["predictions"][0]["result"]
                if result["type"] == "rectanglelabels"
            )
            self.assertEqual(bbox["value"]["width"], 50.0)

    def test_import_rejects_stale_source_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            prepare_review(root, stage="geometry", dpi=100)
            tasks = json.loads((root / "review" / "geometry.tasks.json").read_text())
            tasks[0]["data"]["source_image_hash"] = "0" * 64
            tasks[0]["annotations"] = [
                {"result": tasks[0]["predictions"][0]["result"]}
            ]
            export = root / "stale.json"
            export.write_text(json.dumps(tasks), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Scoresheet version mismatch"):
                import_review(root, export, stage="geometry")

    def test_import_rejects_duplicate_review_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_dataset(root)
            prepare_review(root, stage="geometry", dpi=100)
            tasks = json.loads((root / "review" / "geometry.tasks.json").read_text())
            task = tasks[0]
            task["annotations"] = [
                {"result": task["predictions"][0]["result"]}
            ]
            export = root / "duplicate.json"
            export.write_text(
                json.dumps([task, json.loads(json.dumps(task))]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Duplicate document in export"):
                import_review(root, export, stage="geometry")

    def test_geometry_import_accepts_label_studio_uploaded_png_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document_id, _ = _make_dataset(root)
            prepare_review(root, stage="geometry", dpi=100)
            tasks = json.loads((root / "review" / "geometry.tasks.json").read_text())
            task = tasks[0]
            annotation_results = json.loads(json.dumps(task["predictions"][0]["result"]))
            period_scores = next(
                result
                for result in annotation_results
                if result.get("type") == "rectanglelabels"
                and result["value"]["rectanglelabels"] == ["period_scores"]
            )
            period_scores["value"]["x"] = -1e-10
            task["annotations"] = [{"result": annotation_results}]

            upload_name = f"deadbeef-{document_id}.png"
            upload = (
                root
                / "review"
                / "label-studio-data"
                / "media"
                / "upload"
                / "1"
                / upload_name
            )
            upload.parent.mkdir(parents=True, exist_ok=True)
            source_media = root / "review" / "media" / "geometry" / f"{document_id}.png"
            upload.write_bytes(source_media.read_bytes())
            task["file_upload"] = upload_name
            task["data"] = {"image": f"/data/upload/1/{upload_name}"}

            export = root / "uploaded-geometry-export.json"
            export.write_text(json.dumps(tasks), encoding="utf-8")
            imported = import_review(root, export, stage="geometry")

            self.assertEqual(imported["imported_regions"], len(GEOMETRY_REGIONS))
            self.assertEqual(review_status(root)["geometry"]["pending_regions"], 0)


if __name__ == "__main__":
    unittest.main()
