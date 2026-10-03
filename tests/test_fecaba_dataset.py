from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image, ImageDraw

from sumula_reader.fecaba_dataset import (
    _foul_labels,
    build_fecaba_crops,
    ingest_scoresheet_archive,
)


def _jpeg_scoresheet() -> bytes:
    image = Image.new("RGB", (900, 1300), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 880, 1280), outline="black", width=8)
    payload = BytesIO()
    image.save(payload, format="JPEG")
    return payload.getvalue()


class FecabaDatasetTests(unittest.TestCase):
    def test_verified_foul_observation_keeps_its_cell_slot(self):
        ground_truth = {
            "teams": {
                "A": {
                    "individual_fouls": {},
                    "individual_foul_observations": {
                        "5": [
                            {
                                "slot": 3,
                                "symbol": "P2",
                                "period_candidates": ["Q3"],
                                "review_state": "verified",
                            }
                        ]
                    },
                }
            }
        }

        self.assertEqual(_foul_labels(ground_truth, "A", 5), {3: ("Q3", "P2")})

    def test_ingest_catalogs_documents_and_creates_pilot_ground_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            archive = base / "sumulas.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("Sumulas/Sub14/Jogo A.jpeg", _jpeg_scoresheet())
                zipped.writestr("Sumulas/Sub15/Jogo B.jpeg", _jpeg_scoresheet())

            summary = ingest_scoresheet_archive(
                archive,
                output_root=base / "dataset",
                pilot_count=2,
            )

            self.assertEqual(summary["documents"], 2)
            self.assertEqual(len(summary["pilot_documents"]), 2)
            catalog = (base / "dataset" / "catalog.jsonl").read_text(
                encoding="utf-8"
            )
            records = [json.loads(line) for line in catalog.splitlines()]
            self.assertTrue(all(record["sha256"] for record in records))
            self.assertTrue(all(record["selected_pilot"] for record in records))
            for document_id in summary["pilot_documents"]:
                ground_truth = (
                    base / "dataset" / "ground_truth" / f"{document_id}.json"
                )
                self.assertTrue(ground_truth.exists())
                payload = json.loads(ground_truth.read_text(encoding="utf-8"))
                self.assertFalse(
                    payload["teams"]["A"]["individual_fouls_reviewed"]
                )
                self.assertEqual(
                    payload["teams"]["A"]["individual_foul_observations"], {}
                )

            build = build_fecaba_crops(base / "dataset")
            self.assertEqual(build["documents"], 2)
            self.assertTrue((base / "dataset" / "crops" / "manifest.jsonl").exists())

    def test_ingest_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            archive = base / "sumulas.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("../fora.jpeg", _jpeg_scoresheet())

            with self.assertRaisesRegex(ValueError, "Unsafe"):
                ingest_scoresheet_archive(archive, output_root=base / "dataset")


if __name__ == "__main__":
    unittest.main()
