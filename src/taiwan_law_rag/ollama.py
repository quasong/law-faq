from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any


_START_LOCK = threading.Lock()
_ollama_process: subprocess.Popen[bytes] | None = None


def start_ollama_server() -> int:
    """Start a detached local Ollama server and return its process id."""
    global _ollama_process
    with _START_LOCK:
        if _ollama_process is not None and _ollama_process.poll() is None:
            return _ollama_process.pid
        executable = shutil.which("ollama")
        if executable is None:
            raise RuntimeError("找不到 Ollama CLI，請先安裝 Ollama 並確認 `ollama` 位於 PATH")

        options: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "close_fds": True,
        }
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        else:
            options["start_new_session"] = True
        try:
            _ollama_process = subprocess.Popen([executable, "serve"], **options)
        except OSError as exc:
            raise RuntimeError(f"無法啟動 Ollama：{exc}") from exc
        return _ollama_process.pid


def canonical_model_name(name: str) -> str:
    """Match Ollama's implicit `latest` tag for local model comparisons."""
    value = name.strip()
    final_part = value.rsplit("/", 1)[-1]
    return value if ":" in final_part else f"{value}:latest"


class OllamaClient:
    def __init__(self, base_url: str, timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"無法連線 Ollama ({self.base_url})：{exc}。請確認 `ollama serve` 已啟動且模型已下載。"
            ) from exc

    def _get(self, path: str) -> dict[str, Any]:
        request = urllib.request.Request(f"{self.base_url}{path}", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"無法連線 Ollama ({self.base_url})：{exc}。請確認 `ollama serve` 已啟動。"
            ) from exc

    def list_models(self) -> list[dict[str, Any]]:
        result = self._get("/api/tags")
        models = result.get("models")
        if not isinstance(models, list):
            raise RuntimeError("Ollama /api/tags 回傳格式不正確")
        return [item for item in models if isinstance(item, dict)]

    def model_names(self) -> list[str]:
        names: list[str] = []
        for item in self.list_models():
            name = item.get("name") or item.get("model")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
        return names

    def is_model_installed(self, model: str) -> bool:
        wanted = canonical_model_name(model)
        return any(canonical_model_name(name) == wanted for name in self.model_names())

    def pull_model(self, model: str) -> Iterator[dict[str, Any]]:
        request = urllib.request.Request(
            f"{self.base_url}/api/pull",
            data=json.dumps({"model": model, "stream": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    if not raw_line.strip():
                        continue
                    payload = json.loads(raw_line)
                    if payload.get("error"):
                        raise RuntimeError(f"Ollama 下載模型失敗：{payload['error']}")
                    if isinstance(payload, dict):
                        yield payload
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"無法連線 Ollama ({self.base_url})：{exc}。請確認 `ollama serve` 已啟動。"
            ) from exc

    def embeddings(self, model: str, texts: list[str]) -> list[list[float]]:
        result = self._post("/api/embed", {"model": model, "input": texts, "truncate": True})
        embeddings = result.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Ollama /api/embed 回傳格式或數量不正確")
        return embeddings

    def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        result = self._post(
            "/api/chat",
            {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.1},
            },
        )
        try:
            return str(result["message"]["content"]).strip()
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Ollama /api/chat 回傳格式不正確") from exc

    def chat_stream(self, model: str, messages: list[dict[str, str]]) -> Iterator[str]:
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(
                {
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "options": {"temperature": 0.1},
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    if not raw_line.strip():
                        continue
                    payload = json.loads(raw_line)
                    if payload.get("error"):
                        raise RuntimeError(f"Ollama 生成失敗：{payload['error']}")
                    content = payload.get("message", {}).get("content", "")
                    if content:
                        yield str(content)
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"無法連線 Ollama ({self.base_url})：{exc}。請確認 `ollama serve` 已啟動且模型已下載。"
            ) from exc
