from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import Settings
from .rag import LawRAG, add_deterministic_citations


app = FastAPI(title="台灣法規 RAG", version="0.1.0")


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=6, ge=1, le=15)


@lru_cache(maxsize=1)
def get_rag() -> LawRAG:
    return LawRAG(Settings.from_env())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(payload: Question) -> dict[str, object]:
    try:
        answer = get_rag().ask(payload.question, top_k=payload.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "answer": answer.text,
        "sources": [
            {
                "score": source.score,
                "law_name": source.chunk["law_name"],
                "article_number": source.chunk["article_number"],
                "modified_date": source.chunk["modified_date"],
                "url": source.chunk["url"],
                "text": source.chunk["text"],
            }
            for source in answer.sources
        ],
    }


def _source_payload(source: object) -> dict[str, object]:
    return {
        "score": source.score,
        "law_name": source.chunk["law_name"],
        "article_number": source.chunk["article_number"],
        "modified_date": source.chunk["modified_date"],
        "url": source.chunk["url"],
        "text": source.chunk["text"],
    }


def _event(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


@app.post("/ask/stream")
def ask_stream(payload: Question) -> StreamingResponse:
    def generate() -> Iterator[str]:
        yield _event({"type": "status", "message": "正在檢索相關法規"})
        try:
            sources, tokens = get_rag().stream(payload.question, top_k=payload.top_k)
            if not sources:
                yield _event(
                    {
                        "type": "final",
                        "text": "找不到足夠相關的法規內容。請改用更完整的法規名稱、條號或法律關鍵詞。",
                    }
                )
                yield _event({"type": "done"})
                return
            yield _event({"type": "sources", "sources": [_source_payload(item) for item in sources]})
            yield _event({"type": "status", "message": "正在生成法規說明"})
            parts: list[str] = []
            for token in tokens:
                parts.append(token)
                yield _event({"type": "delta", "text": token})
            final_text = add_deterministic_citations("".join(parts), sources)
            yield _event({"type": "final", "text": final_text})
            yield _event({"type": "done"})
        except RuntimeError as exc:
            yield _event({"type": "error", "message": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"},
    )


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return files("taiwan_law_rag").joinpath("web/index.html").read_text(encoding="utf-8")
