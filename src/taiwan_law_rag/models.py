from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Article:
    number: str
    text: str
    section: str = ""


@dataclass(frozen=True)
class Law:
    name: str
    nature: str
    category: str
    url: str
    modified_date: str
    effective_date: str
    effective_note: str
    abolished: bool
    history: str
    preamble: str
    articles: tuple[Article, ...]
    dataset: str


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    law_name: str
    article_number: str
    section: str
    url: str
    modified_date: str
    nature: str
    category: str
    dataset: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

