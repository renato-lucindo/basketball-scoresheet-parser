import unittest

from sumula_reader.dataset import assign_document_split, validate_split_isolation


class DatasetTests(unittest.TestCase):
    def test_same_document_always_uses_same_split(self):
        first = assign_document_split("game_2026_001")
        second = assign_document_split("game_2026_001")
        self.assertEqual(first, second)

    def test_split_is_one_of_expected_values(self):
        self.assertIn(
            assign_document_split("game_2026_002"),
            {"train", "validation", "test"},
        )

    def test_same_writer_stays_in_same_split_across_documents(self):
        first = assign_document_split("game_2026_003", writer_id="writer_abc")
        second = assign_document_split("game_2026_999", writer_id="writer_abc")
        self.assertEqual(first, second)

    def test_split_validation_rejects_writer_leak(self):
        records = [
            {
                "document_id": "game_1",
                "writer_id": "writer_abc",
                "split": "train",
            },
            {
                "document_id": "game_2",
                "writer_id": "writer_abc",
                "split": "test",
            },
        ]
        with self.assertRaisesRegex(ValueError, "scorers=writer_abc"):
            validate_split_isolation(records)


if __name__ == "__main__":
    unittest.main()
