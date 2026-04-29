from __future__ import annotations

import unittest

from kbvqa.kb import KnowledgeBase
from kbvqa.retriever import KBRetriever


class RetrieverTests(unittest.TestCase):
    def test_search_orders_by_tfidf_score(self):
        kb = KnowledgeBase.from_dicts(
            [
                {
                    "id": "apple",
                    "title": "Apple",
                    "text": "A red edible fruit used in pies.",
                },
                {
                    "id": "traffic",
                    "title": "Traffic Light",
                    "text": "A signal with red yellow and green lights.",
                },
            ]
        )
        retriever = KBRetriever(kb)

        results = retriever.search("red fruit pie", top_k=2)

        self.assertEqual(results[0].id, "apple")
        self.assertGreater(results[0].score, results[1].score)

    def test_empty_query_returns_empty_list(self):
        kb = KnowledgeBase.from_dicts(
            [{"id": "x", "title": "X", "text": "Some text"}]
        )

        self.assertEqual(KBRetriever(kb).search("   "), [])


if __name__ == "__main__":
    unittest.main()
