"""Local knowledge base loading and validation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class KnowledgeBaseError(ValueError):
    """Raised when a knowledge base cannot be loaded or validated."""


@dataclass(frozen=True)
class KBEntry:
    """One local knowledge base document."""

    id: str
    title: str
    text: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "text": self.text,
            "metadata": self.metadata,
        }


class KnowledgeBase:
    """In-memory local knowledge base."""

    def __init__(self, entries: Iterable[KBEntry]):
        self._entries = list(entries)
        if not self._entries:
            raise KnowledgeBaseError("Knowledge base must contain at least one entry.")
        self._by_id = {entry.id: entry for entry in self._entries}
        if len(self._by_id) != len(self._entries):
            raise KnowledgeBaseError("Knowledge base contains duplicate entry ids.")

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeBase":
        kb_path = Path(path)
        if not kb_path.exists():
            raise KnowledgeBaseError(f"Knowledge base file does not exist: {kb_path}")
        suffix = kb_path.suffix.lower()
        if suffix == ".json":
            return cls(_load_json_entries(kb_path))
        if suffix == ".csv":
            return cls(_load_csv_entries(kb_path))
        raise KnowledgeBaseError(
            f"Unsupported knowledge base format '{suffix}'. Use .json or .csv."
        )

    @classmethod
    def from_dicts(cls, rows: Iterable[dict[str, Any]]) -> "KnowledgeBase":
        return cls(_coerce_entry(row, index) for index, row in enumerate(rows, start=1))

    def __iter__(self):
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, entry_id: str) -> KBEntry | None:
        return self._by_id.get(entry_id)


def _load_json_entries(path: Path) -> list[KBEntry]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = None
        for key in ("entries", "items", "documents"):
            candidate = data.get(key)
            if isinstance(candidate, list):
                rows = candidate
                break
        if rows is None:
            raise KnowledgeBaseError(
                "JSON knowledge base must be a list or contain entries/items/documents."
            )
    else:
        raise KnowledgeBaseError("JSON knowledge base must be an object or list.")
    return [_coerce_entry(row, index) for index, row in enumerate(rows, start=1)]


def _load_csv_entries(path: Path) -> list[KBEntry]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise KnowledgeBaseError("CSV knowledge base is missing a header row.")
        missing = {"id", "title", "text"} - set(reader.fieldnames)
        if missing:
            raise KnowledgeBaseError(
                f"CSV knowledge base is missing required columns: {sorted(missing)}"
            )
        return [_coerce_entry(row, index) for index, row in enumerate(reader, start=1)]


def _coerce_entry(row: Any, index: int) -> KBEntry:
    if not isinstance(row, dict):
        raise KnowledgeBaseError(f"Entry #{index} must be an object.")

    missing = [key for key in ("id", "title", "text") if key not in row]
    if missing:
        raise KnowledgeBaseError(f"Entry #{index} is missing required fields: {missing}")

    entry_id = str(row["id"]).strip()
    title = str(row["title"]).strip()
    text = str(row["text"]).strip()
    if not entry_id or not title or not text:
        raise KnowledgeBaseError(
            f"Entry #{index} fields id, title, and text must be non-empty."
        )

    metadata = _coerce_metadata(row.get("metadata"), index)
    extra = {
        str(key): value
        for key, value in row.items()
        if key not in {"id", "title", "text", "metadata"} and value not in (None, "")
    }
    if extra:
        metadata = {**extra, **metadata}

    return KBEntry(id=entry_id, title=title, text=text, metadata=metadata)


def _coerce_metadata(value: Any, index: int) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise KnowledgeBaseError(
                f"Entry #{index} metadata must be a JSON object string."
            ) from exc
        if not isinstance(parsed, dict):
            raise KnowledgeBaseError(f"Entry #{index} metadata must be an object.")
        return parsed
    raise KnowledgeBaseError(f"Entry #{index} metadata must be an object.")
