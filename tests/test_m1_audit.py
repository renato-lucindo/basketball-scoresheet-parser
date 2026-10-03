from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from sumula_reader.dataset import assign_document_split
from sumula_reader.m1_audit import audit_fecaba_dataset, build_baseline_metrics


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _ground_truth(
    document_id: str,
    source_path: str,
    *,
    reviewed: bool,
    writer_name: str | None = "Writer A",
    evaluation_split: str | None = None,
) -> dict:
    periods = {f"Q{period}": 0 for period in range(1, 5)}
    return {
        "game_id": document_id,
        "source_file": source_path,
        "writer_name": writer_name,
        "evaluation_split": evaluation_split,
        "review": {
            "status": "reviewed" if reviewed else "pending",
            "reviewed_at": "2026-10-02" if reviewed else None,
            "method": "manual",
        },
        "teams": {
            "A": {"roster": [4, 5], "team_fouls": periods, "individual_fouls": {}},
            "B": {"roster": [6, 7], "team_fouls": periods, "individual_fouls": {}},
        },
        "scoring": [],
        "period_scores": {"A": periods, "B": periods},
        "final_score": {"A": 0, "B": 0},
        "warnings": [],
    }


class M1AuditTests(unittest.TestCase):
    def _dataset(
        self,
        root: Path,
        *,
        reviewed: bool,
        document_id: str = "game-001",
        writer_name: str | None = "Writer A",
        evaluation_split: str | None = None,
    ) -> str:
        source_path = "source/u15/game-001.pdf"
        _write_jsonl(
            root / "catalog.jsonl",
            [
                {
                    "document_id": document_id,
                    "source_path": source_path,
                    "sha256": "a" * 64,
                    "selected_pilot": True,
                }
            ],
        )
        ground_truth = root / "ground_truth" / f"{document_id}.json"
        ground_truth.parent.mkdir(parents=True, exist_ok=True)
        ground_truth.write_text(
            json.dumps(
                _ground_truth(
                    document_id,
                    source_path,
                    reviewed=reviewed,
                    writer_name=writer_name,
                    evaluation_split=evaluation_split,
                )
            ),
            encoding="utf-8",
        )
        return document_id

    def test_audit_keeps_pending_ground_truth_out_of_reviewed_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._dataset(root, reviewed=False)

            report = audit_fecaba_dataset(root)

            self.assertEqual(report["ground_truth_documents"], 1)
            self.assertEqual(report["reviewed_ground_truth_documents"], 0)
            self.assertEqual(report["review_status_counts"], {"pending": 1})
            self.assertFalse(report["exit_criteria"]["reviewed_ground_truth"])
            self.assertEqual(report["status"], "in_progress")
            self.assertTrue((root / "evaluation" / "evaluation-manifest.jsonl").exists())

    def test_audit_fingerprint_is_stable_and_writer_split_is_isolated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._dataset(root, reviewed=True)

            first = audit_fecaba_dataset(root)
            second = audit_fecaba_dataset(root)

            self.assertEqual(first["corpus_sha256"], second["corpus_sha256"])
            self.assertEqual(first["known_writers"], 1)
            self.assertEqual(first["review_status_counts"], {"reviewed": 1})
            self.assertTrue(first["exit_criteria"]["document_and_writer_isolation"])

    def test_reviewed_status_requires_review_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document_id = self._dataset(root, reviewed=True)
            path = root / "ground_truth" / f"{document_id}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["review"]["reviewed_at"] = None
            path.write_text(json.dumps(payload), encoding="utf-8")

            report = audit_fecaba_dataset(root)

            self.assertEqual(report["reviewed_ground_truth_documents"], 0)
            manifest = [
                json.loads(line)
                for line in (root / "evaluation" / "evaluation-manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertIn("reviewed_at_missing", manifest[0]["issues"])

    def test_explicit_split_stays_stable_when_writer_becomes_known(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._dataset(
                root,
                reviewed=True,
                writer_name="Writer A",
                evaluation_split="test",
            )

            report = audit_fecaba_dataset(root)
            manifest = [
                json.loads(line)
                for line in (root / "evaluation" / "evaluation-manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            self.assertEqual(report["split_counts"]["test"], 1)
            self.assertEqual(manifest[0]["split_source"], "explicit")

    def test_baseline_is_bound_to_reviewed_test_corpus(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document_id = next(
                f"game-{index:03d}"
                for index in range(1, 1000)
                if assign_document_split(f"game-{index:03d}") == "test"
            )
            self._dataset(
                root,
                reviewed=True,
                document_id=document_id,
                writer_name=None,
            )
            audit = audit_fecaba_dataset(root)
            predictions = root / "predictions.jsonl"
            _write_jsonl(
                predictions,
                [
                    {
                        "document_id": document_id,
                        "expected": "12",
                        "predicted": "12",
                        "confidence": 0.99,
                        "writer_known": True,
                    }
                ],
            )

            result = build_baseline_metrics(root, predictions)

            self.assertEqual(result["corpus_sha256"], audit["corpus_sha256"])
            self.assertEqual(result["metrics"]["accepted_error_rate"], 0.0)
            updated = audit_fecaba_dataset(root)
            self.assertTrue(updated["exit_criteria"]["baseline_metrics_reproducible"])

    def test_baseline_rejects_partial_test_document_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            test_ids = [
                f"game-{index:03d}"
                for index in range(1, 1000)
                if assign_document_split(f"game-{index:03d}") == "test"
            ][:2]
            rows = []
            for document_id in test_ids:
                source_path = f"source/u15/{document_id}.pdf"
                rows.append(
                    {
                        "document_id": document_id,
                        "source_path": source_path,
                        "sha256": document_id.ljust(64, "0")[:64],
                        "selected_pilot": True,
                    }
                )
                ground_truth = root / "ground_truth" / f"{document_id}.json"
                ground_truth.parent.mkdir(parents=True, exist_ok=True)
                ground_truth.write_text(
                    json.dumps(
                        _ground_truth(
                            document_id,
                            source_path,
                            reviewed=True,
                            writer_name=None,
                        )
                    ),
                    encoding="utf-8",
                )
            _write_jsonl(root / "catalog.jsonl", rows)
            predictions = root / "predictions.jsonl"
            _write_jsonl(
                predictions,
                [
                    {
                        "document_id": test_ids[0],
                        "expected": "12",
                        "predicted": "12",
                        "confidence": 0.99,
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "do not cover every reviewed test document"):
                build_baseline_metrics(root, predictions)


if __name__ == "__main__":
    unittest.main()
