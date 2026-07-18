from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


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

