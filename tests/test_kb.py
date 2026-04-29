from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kbvqa.kb import KnowledgeBase, KnowledgeBaseError


class KnowledgeBaseTests(unittest.TestCase):
    def test_load_json_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kb.json"
            path.write_text(
                '[{"id":"apple","title":"Apple","text":"A red fruit.","metadata":{"source":"x"}}]',
                encoding="utf-8",
            )

            kb = KnowledgeBase.load(path)

            self.assertEqual(len(kb), 1)
            entry = kb.get("apple")
            self.assertIsNotNone(entry)
            self.assertEqual(entry.title, "Apple")
            self.assertEqual(entry.metadata["source"], "x")

    def test_load_csv_with_extra_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kb.csv"
            path.write_text(
                'id,title,text,metadata,source\n'
                'bike,Bicycle,"Two wheels","{""kind"": ""vehicle""}",sample\n',
                encoding="utf-8",
            )

            kb = KnowledgeBase.load(path)
            entry = kb.get("bike")

            self.assertIsNotNone(entry)
            self.assertEqual(entry.metadata["kind"], "vehicle")
            self.assertEqual(entry.metadata["source"], "sample")

    def test_missing_required_field_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kb.json"
            path.write_text('[{"id":"x","title":"Missing text"}]', encoding="utf-8")

            with self.assertRaises(KnowledgeBaseError):
                KnowledgeBase.load(path)


if __name__ == "__main__":
    unittest.main()
