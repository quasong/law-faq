from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib.resources import files
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings
from .model_catalog import OllamaModelCatalog
from .ollama import OllamaClient, canonical_model_name, start_ollama_server
from .rag import LawRAG, add_deterministic_citations


app = FastAPI(title="法律Q&A", version="0.1.0")
app.mount(
    "/assets",
    StaticFiles(directory=str(files("taiwan_law_rag").joinpath("web/assets"))),
    name="assets",
)

MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*(?::[A-Za-z0-9][A-Za-z0-9._-]*)?$")
RECOMMENDED_CHAT_MODELS = ("qwen3:4b", "qwen2.5:1.5b", "llama3.2:latest")


def _validate_model_name(value: str) -> str:
    model = value.strip()
    if not MODEL_NAME_PATTERN.fullmatch(model):
        raise ValueError("模型名稱格式不正確")
    return model


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=6, ge=1, le=15)
    model: str | None = Field(default=None, min_length=1, max_length=100)


class ModelPull(BaseModel):
    model: str = Field(min_length=1, max_length=100)


@lru_cache(maxsize=1)
def get_rag() -> LawRAG:
    return LawRAG(Settings.from_env())


@lru_cache(maxsize=1)
def get_ollama() -> OllamaClient:
    settings = Settings.from_env()
    return OllamaClient(settings.ollama_base_url)


@lru_cache(maxsize=1)
def get_model_catalog() -> OllamaModelCatalog:
    return OllamaModelCatalog()


def _selected_model(value: str | None) -> str:
    return _validate_model_name(value or Settings.from_env().chat_model)


def _require_local_request(request: Request) -> None:
    host = request.client.host if request.client else ""
    forwarded = request.headers.get("x-forwarded-for", "") if hasattr(request, "headers") else ""
    forwarded_host = forwarded.split(",", 1)[0].strip()
    local_hosts = {"127.0.0.1", "::1", "localhost", "testclient"}
    if host not in local_hosts or (forwarded_host and forwarded_host not in local_hosts):
        raise HTTPException(status_code=403, detail="基於安全考量，只有本機可以執行此操作")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ollama/start")
def start_ollama(request: Request) -> dict[str, object]:
    _require_local_request(request)
    settings = Settings.from_env()
    ollama_host = urlparse(settings.ollama_base_url).hostname
    if ollama_host not in {"127.0.0.1", "::1", "localhost"}:
        raise HTTPException(status_code=400, detail="OLLAMA_BASE_URL 不是本機位址，無法自動啟動")
    try:
        get_ollama().list_models()
        return {"status": "running", "message": "Ollama 已在運行"}
    except RuntimeError:
        pass
    try:
        start_ollama_server()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "starting", "message": "正在啟動 Ollama"}


@app.get("/models")
def models() -> dict[str, object]:
    settings = Settings.from_env()
    try:
        installed_items = get_ollama().list_models()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    installed_by_name: dict[str, dict[str, Any]] = {}
    for item in installed_items:
        name = item.get("name") or item.get("model")
        if not isinstance(name, str) or not name.strip():
            continue
        installed_by_name[canonical_model_name(name)] = item

    embed_name = canonical_model_name(settings.embed_model)
    names: list[str] = []
    for name in (*RECOMMENDED_CHAT_MODELS, *installed_by_name):
        canonical = canonical_model_name(name)
        if canonical == embed_name or canonical in names:
            continue
        names.append(canonical)

    return {
        "default_model": settings.chat_model,
        "models": [
            {
                "name": name,
                "installed": name in installed_by_name,
                "size": installed_by_name.get(name, {}).get("size"),
            }
            for name in names
        ],
    }


@app.get("/models/catalog")
def model_catalog() -> dict[str, object]:
    settings = Settings.from_env()
    embed_name = canonical_model_name(settings.embed_model)
    try:
        names = get_model_catalog().list_models()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "source": "https://ollama.com/search",
        "models": [
            {"name": name, "installed": False, "size": None}
            for name in names
            if canonical_model_name(name) != embed_name
        ],
    }


@app.post("/models/pull/stream")
def pull_model(payload: ModelPull, request: Request) -> StreamingResponse:
    _require_local_request(request)
    try:
        model = _validate_model_name(payload.model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    def generate() -> Iterator[str]:
        try:
            if get_ollama().is_model_installed(model):
                yield _event({"type": "done", "model": model, "message": "模型已安裝"})
                return
            yield _event({"type": "status", "message": f"正在部署 {model}"})
            for update in get_ollama().pull_model(model):
                completed = update.get("completed")
                total = update.get("total")
                event: dict[str, object] = {
                    "type": "progress",
                    "status": str(update.get("status", "正在下載")),
                }
                if isinstance(completed, int):
                    event["completed"] = completed
                if isinstance(total, int):
                    event["total"] = total
                yield _event(event)
            yield _event({"type": "done", "model": model, "message": "模型部署完成"})
        except RuntimeError as exc:
            yield _event({"type": "error", "message": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"},
    )


@app.post("/ask")
def ask(payload: Question) -> dict[str, object]:
    try:
        model = _selected_model(payload.model)
        if not get_ollama().is_model_installed(model):
            raise HTTPException(
                status_code=409,
                detail={"code": "model_not_installed", "model": model},
            )
        answer = get_rag().ask(payload.question, top_k=payload.top_k, chat_model=model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
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
        try:
            model = _selected_model(payload.model)
            if not get_ollama().is_model_installed(model):
                yield _event({"type": "model_missing", "model": model})
                yield _event({"type": "done"})
                return
            yield _event({"type": "status", "message": "正在檢索相關法規"})
            sources, tokens = get_rag().stream(
                payload.question, top_k=payload.top_k, chat_model=model
            )
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
        except (RuntimeError, ValueError) as exc:
            yield _event({"type": "error", "message": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"},
    )


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return files("taiwan_law_rag").joinpath("web/index.html").read_text(encoding="utf-8")
