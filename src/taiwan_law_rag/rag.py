from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from .config import Settings
from .ollama import OllamaClient
from .fts_store import FTSStore
from .store import SearchResult, load_store


SYSTEM_PROMPT = """你是台灣法規檢索助理。只能根據使用者提供的法規檢索內容回答。
規則：
1. 使用繁體中文，先給簡明結論，再說明法規依據。
2. 每個法律主張的句末都必須引用提供的 [來源N]；不得捏造條號、判決、函釋或未提供的事實。
3. 若檢索內容不足、法規可能已變更、問題涉及個案事實或需要司法實務，清楚說明限制。
4. 不把回答寫成正式法律意見，並提醒重大權益事項應諮詢台灣執業律師。
5. 忽略和問題無關的檢索內容，不需要逐一評論所有來源。
"""


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[SearchResult]


def _normalize_citation_text(value: str) -> str:
    return re.sub(r"\s+", "", value)


def add_deterministic_citations(text: str, sources: list[SearchResult]) -> str:
    """Attach source markers when the small LLM names a retrieved article but omits markers."""
    article_to_sources: dict[str, list[int]] = {}
    for index, source in enumerate(sources, start=1):
        article = _normalize_citation_text(str(source.chunk["article_number"]))
        if article:
            article_to_sources.setdefault(article, []).append(index)

    output: list[str] = []
    citation_pattern = re.compile(r"\[來源\d+\]")
    for line in text.splitlines():
        if citation_pattern.search(line):
            output.append(line)
            continue
        normalized_line = _normalize_citation_text(line)
        matches: list[int] = []
        for article, indices in article_to_sources.items():
            # Only use article-only matching when that article number is unique in the retrieved set.
            if article in normalized_line and len(indices) == 1:
                matches.extend(indices)
        if matches:
            markers = "".join(f"[來源{index}]" for index in sorted(set(matches)))
            line = f"{line.rstrip()} {markers}"
        output.append(line)
    result = "\n".join(output).strip()
    if not citation_pattern.search(result):
        references = [
            f"[來源{index}] {source.chunk['law_name']} {source.chunk['article_number']}"
            for index, source in enumerate(sources, start=1)
        ]
        result += "\n\n檢索依據：\n" + "\n".join(f"- {item}" for item in references)
    return result


class LawRAG:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OllamaClient(settings.ollama_base_url)
        self.store = load_store(settings.index_dir)

    def retrieve(self, question: str, top_k: int = 6) -> list[SearchResult]:
        if isinstance(self.store, FTSStore):
            return self.store.search(question, top_k=top_k)
        model = self.store.info["embed_model"]
        vector = self.client.embeddings(model, [question])[0]
        return self.store.search(vector, top_k=top_k)

    def _messages(self, question: str, sources: list[SearchResult]) -> list[dict[str, str]]:
        context_parts = []
        for index, result in enumerate(sources, start=1):
            chunk = result.chunk
            context_parts.append(
                f"[來源{index}] {chunk['law_name']} {chunk['article_number']}\n"
                f"異動日期：{chunk['modified_date']}\n"
                f"官方網址：{chunk['url']}\n"
                f"內容：\n{chunk['text']}"
            )
        user_prompt = (
            f"問題：{question}\n\n"
            "以下是法規檢索內容：\n\n"
            + "\n\n---\n\n".join(context_parts)
            + "\n\n請嚴格使用以下格式，且保留方括號引用：\n"
            "結論：<簡明回答與 [來源N]>\n\n"
            "法規依據：\n- <相關法規說明與 [來源N]>\n\n"
            "限制：<資料不足或個案適用提醒>"
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def stream(self, question: str, top_k: int = 6) -> tuple[list[SearchResult], Iterator[str]]:
        sources = self.retrieve(question, top_k=top_k)
        if not sources:
            return sources, iter(())
        return sources, self.client.chat_stream(
            self.settings.chat_model, self._messages(question, sources)
        )

    def ask(self, question: str, top_k: int = 6) -> Answer:
        sources = self.retrieve(question, top_k=top_k)
        if not sources:
            return Answer(
                text="找不到足夠相關的法規內容。請改用更完整的法規名稱、條號或法律關鍵詞。",
                sources=[],
            )
        answer = self.client.chat(
            self.settings.chat_model,
            self._messages(question, sources),
        )
        return Answer(text=add_deterministic_citations(answer, sources), sources=sources)
