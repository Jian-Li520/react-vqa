"""Small local TF-IDF retriever for knowledge evidence."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .kb import KBEntry, KnowledgeBase


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


@dataclass(frozen=True)
class Evidence:
    """A retrieved knowledge entry with a retrieval score."""

    id: str
    title: str
    text: str
    score: float
    metadata: dict[str, Any]

    @classmethod
    def from_entry(cls, entry: KBEntry, score: float) -> "Evidence":
        return cls(
            id=entry.id,
            title=entry.title,
            text=entry.text,
            score=score,
            metadata=entry.metadata,
        )

    def to_dict(self, include_text: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "score": self.score,
        }
        if include_text:
            payload["text"] = self.text
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


class KBRetriever:
    """A dependency-light TF-IDF retriever over local KB entries."""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
        self._entries = list(knowledge_base)
        self._doc_terms = [
            Counter(_tokenize(f"{entry.title} {entry.title} {entry.text}"))
            for entry in self._entries
        ]
        self._idf = self._compute_idf(self._doc_terms)

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        if top_k <= 0:
            return []
        query_terms = Counter(_tokenize(query))
        if not query_terms:
            return []

        scores = []
        for index, doc_terms in enumerate(self._doc_terms):
            score = self._score(query_terms, doc_terms)
            if score > 0:
                scores.append((index, score))

        scores.sort(key=lambda item: (-item[1], item[0]))
        return [
            Evidence.from_entry(self._entries[index], round(score, 6))
            for index, score in scores[:top_k]
        ]

    @staticmethod
    def _compute_idf(doc_terms: list[Counter[str]]) -> dict[str, float]:
        doc_count = len(doc_terms)
        document_frequency: Counter[str] = Counter()
        for terms in doc_terms:
            document_frequency.update(terms.keys())
        return {
            term: math.log((1 + doc_count) / (1 + frequency)) + 1.0
            for term, frequency in document_frequency.items()
        }

    def _score(self, query_terms: Counter[str], doc_terms: Counter[str]) -> float:
        score = 0.0
        doc_length = sum(doc_terms.values()) or 1
        query_length = sum(query_terms.values()) or 1
        for term, query_tf in query_terms.items():
            doc_tf = doc_terms.get(term, 0)
            if doc_tf == 0:
                continue
            idf = self._idf.get(term, 1.0)
            score += (query_tf / query_length) * (doc_tf / doc_length) * (idf * idf)
        return score


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]
