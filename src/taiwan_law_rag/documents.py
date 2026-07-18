from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .models import Chunk, Law


def _law_id(law: Law) -> str:
    pcode = parse_qs(urlparse(law.url).query).get("pcode", [""])[0]
    if pcode and re.fullmatch(r"[A-Za-z0-9_-]+", pcode):
        return pcode
    return hashlib.sha256(f"{law.dataset}:{law.name}".encode()).hexdigest()[:16]


def _split_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
    result: list[str] = []
    current = ""
    for paragraph in paragraphs:
        pieces = [paragraph[i : i + max_chars] for i in range(0, len(paragraph), max_chars)]
        for piece in pieces:
            candidate = f"{current}\n{piece}".strip()
            if current and len(candidate) > max_chars:
                result.append(current)
                current = piece
            else:
                current = candidate
    if current:
        result.append(current)
    return result


def law_to_markdown(law: Law) -> str:
    lines = [
        f"# {law.name}",
        "",
        f"- 法規性質：{law.nature}",
        f"- 法規類別：{law.category}",
        f"- 最新異動日期：{law.modified_date}",
        f"- 生效日期：{law.effective_date or '未註明'}",
        f"- 是否廢止：{'是' if law.abolished else '否'}",
        f"- 官方網址：{law.url}",
    ]
    if law.history:
        lines += ["", "## 沿革", "", law.history]
    if law.preamble:
        lines += ["", "## 前言", "", law.preamble]
    previous_section = None
    for article in law.articles:
        if article.section and article.section != previous_section:
            lines += ["", f"## {article.section}"]
            previous_section = article.section
        lines += ["", f"### {article.number}", "", article.text]
    lines.append("")
    return "\n".join(lines)


def write_documents_and_chunks(
    laws: Iterable[Law], documents_dir: Path, max_chars: int = 1200
) -> tuple[list[Chunk], int]:
    chunks: list[Chunk] = []
    law_count = 0
    manifest_path = documents_dir / "manifest.jsonl"
    documents_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for law in laws:
            law_count += 1
            law_id = _law_id(law)
            law_dir = documents_dir / law.dataset
            law_dir.mkdir(parents=True, exist_ok=True)
            relative_path = Path(law.dataset) / f"{law_id}.md"
            (documents_dir / relative_path).write_text(law_to_markdown(law), encoding="utf-8")
            manifest.write(
                json.dumps(
                    {
                        "id": law_id,
                        "name": law.name,
                        "dataset": law.dataset,
                        "path": relative_path.as_posix(),
                        "url": law.url,
                        "modified_date": law.modified_date,
                        "abolished": law.abolished,
                        "article_count": len(law.articles),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            for article in law.articles:
                parts = _split_text(article.text, max_chars=max_chars)
                for part_index, part in enumerate(parts, start=1):
                    chunk_id = f"{law.dataset}:{law_id}:{article.number}:{part_index}"
                    searchable = "\n".join(
                        value
                        for value in (law.name, article.section, article.number, part)
                        if value
                    )
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            text=searchable,
                            law_name=law.name,
                            article_number=article.number,
                            section=article.section,
                            url=law.url,
                            modified_date=law.modified_date,
                            nature=law.nature,
                            category=law.category,
                            dataset=law.dataset,
                        )
                    )
    return chunks, law_count

