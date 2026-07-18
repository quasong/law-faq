from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path

from .models import Chunk
from .store import SearchResult


QUERY_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "解僱": ("終止勞動契約", "資遣", "非有法定情事不得預告"),
    "車禍": ("交通事故",),
    "告人": ("告訴", "起訴"),
}

LAW_ALIASES: dict[str, str] = {
    "勞基法": "勞動基準法",
    "個資法": "個人資料保護法",
    "消保法": "消費者保護法",
    "刑訴": "刑事訴訟法",
    "民訴": "民事訴訟法",
}


def _fingerprint(chunks: list[Chunk]) -> str:
    digest = hashlib.sha256(b"fts5-bigram-v1")
    for chunk in chunks:
        digest.update(chunk.id.encode("utf-8"))
        digest.update(chunk.text.encode("utf-8"))
    return digest.hexdigest()[:16]


def build_fts_index(chunks: list[Chunk], index_dir: Path) -> None:
    """Build a fast, embedding-free Chinese full-text index."""
    if not chunks:
        raise ValueError("沒有可建立索引的法條")
    index_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(chunks)
    db_name = f"fts-{fingerprint}.sqlite3"
    db_path = index_dir / db_name
    info_path = index_dir / "index.json"

    if info_path.exists() and db_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("mode") == "fast" and info.get("fingerprint") == fingerprint:
            print(f"快速索引內容未變更，沿用既有 {len(chunks)} 個法條片段")
            return

    partial_path = index_dir / f"fts-{fingerprint}.partial.sqlite3"
    partial_path.unlink(missing_ok=True)
    connection = sqlite3.connect(partial_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA cache_size=-65536")
        connection.execute(
            """
            CREATE VIRTUAL TABLE chunks USING fts5(
                tokens,
                text UNINDEXED,
                law_name UNINDEXED,
                article_number UNINDEXED,
                section UNINDEXED,
                id UNINDEXED,
                url UNINDEXED,
                modified_date UNINDEXED,
                nature UNINDEXED,
                category UNINDEXED,
                dataset UNINDEXED,
                tokenize='unicode61'
            )
            """
        )
        rows = (
            (
                _bigram_tokens(chunk.text),
                chunk.text,
                chunk.law_name,
                chunk.article_number,
                chunk.section,
                chunk.id,
                chunk.url,
                chunk.modified_date,
                chunk.nature,
                chunk.category,
                chunk.dataset,
            )
            for chunk in chunks
        )
        connection.executemany(
            """
            INSERT INTO chunks(
                tokens, text, law_name, article_number, section, id, url,
                modified_date, nature, category, dataset
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()
        connection.execute("INSERT INTO chunks(chunks) VALUES('optimize')")
        connection.commit()
    except sqlite3.OperationalError as exc:
        if "no such module: fts5" in str(exc).lower():
            raise RuntimeError("目前的 Python SQLite 未啟用 FTS5，請改用 --mode semantic") from exc
        raise
    finally:
        connection.close()

    os.replace(partial_path, db_path)
    info_temp = index_dir / "index.partial.json"
    info_temp.write_text(
        json.dumps(
            {
                "mode": "fast",
                "chunk_count": len(chunks),
                "fingerprint": fingerprint,
                "database_file": db_name,
                "retrieval": "SQLite FTS5 Chinese bigram/BM25",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(info_temp, info_path)
    print(f"快速索引完成：{len(chunks)} 個法條片段")


def _compact(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u3400-\u9fff]+", "", value).lower()


def _bigrams(value: str, max_terms: int | None = None) -> list[str]:
    compact = _compact(value)
    if not compact:
        return []
    if len(compact) == 1:
        return [compact]
    result: list[str] = []
    seen: set[str] = set()
    for start in range(len(compact) - 1):
        token = compact[start : start + 2]
        if token not in seen:
            seen.add(token)
            result.append(token)
        if max_terms is not None and len(result) >= max_terms:
            break
    return result


def _bigram_tokens(value: str) -> str:
    return " ".join(_bigrams(value))


def _match_query(question: str, max_terms: int = 80) -> str:
    expanded = question
    for keyword, additions in QUERY_EXPANSIONS.items():
        if keyword in question:
            expanded += " " + " ".join(additions)
    return " OR ".join(f'"{token}"' for token in _bigrams(expanded, max_terms=max_terms))


def _legal_name_boost(
    question: str, law_name: str, article_number: str, nature: str, dataset: str
) -> float:
    normalized_question = _compact(question)
    normalized_name = _compact(law_name)
    core_name = normalized_name.removeprefix("中華民國")
    boost = 0.0
    if len(core_name) >= 2 and core_name in normalized_question:
        boost += 100.0
    for alias, official_name in LAW_ALIASES.items():
        if alias in normalized_question and official_name in normalized_name:
            boost += 100.0
    normalized_article = _compact(article_number)
    if normalized_article and normalized_article in normalized_question:
        boost += 100.0
    # General questions should prefer statutes over subordinate regulations.
    # An explicitly named regulation still wins through the stronger name boost above.
    if dataset == "laws" or nature in {"憲法", "法律"}:
        boost += 15.0
    return boost


class FTSStore:
    def __init__(self, index_dir: Path, info: dict[str, object] | None = None):
        self.info = info or json.loads((index_dir / "index.json").read_text(encoding="utf-8"))
        self.db_path = index_dir / str(self.info["database_file"])
        if not self.db_path.exists():
            raise RuntimeError("快速索引資料庫不存在，請重新執行 ingest")

    def search(self, question: str, top_k: int = 6) -> list[SearchResult]:
        match = _match_query(question)
        if not match:
            return []
        connection = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            candidate_limit = max(50, top_k * 10)
            rows = connection.execute(
                """
                SELECT text, law_name, article_number, section, id, url,
                       modified_date, nature, category, dataset,
                       bm25(chunks, 1.0) AS rank
                FROM chunks
                WHERE chunks MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (match, candidate_limit),
            ).fetchall()
        finally:
            connection.close()
        results = [
            SearchResult(
                score=float(-row["rank"])
                + _legal_name_boost(
                    question,
                    str(row["law_name"]),
                    str(row["article_number"]),
                    str(row["nature"]),
                    str(row["dataset"]),
                ),
                chunk={key: str(row[key]) for key in row.keys() if key != "rank"},
            )
            for row in rows
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        return results[: max(1, top_k)]
