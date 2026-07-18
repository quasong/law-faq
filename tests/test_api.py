import json

import pytest
from fastapi import HTTPException

import taiwan_law_rag.api as api
from taiwan_law_rag.api import _event, _require_local_request, _validate_model_name, home


def test_home_contains_streaming_llm_interface() -> None:
    html = home()
    assert "法律Q&amp;A" in html
    assert "lady-justice.png" in html
    assert "⚖️" in html
    assert 'id="ask-form"' in html
    assert "'/ask/stream'" in html
    assert "停止生成" in html
    assert "法規來源" in html
    assert 'id="model-name"' in html
    assert "'/models/pull/stream'" in html
    assert "確認部署" in html


def test_ndjson_event() -> None:
    line = _event({"type": "delta", "text": "法規"})
    assert line.endswith("\n")
    assert json.loads(line) == {"type": "delta", "text": "法規"}


def test_validates_ollama_model_name() -> None:
    assert _validate_model_name("library/qwen2.5:1.5b") == "library/qwen2.5:1.5b"
    with pytest.raises(ValueError):
        _validate_model_name("http://example.com/model")


def test_model_pull_is_limited_to_local_requests() -> None:
    local_request = type("Request", (), {"client": type("Client", (), {"host": "127.0.0.1"})()})()
    _require_local_request(local_request)

    remote_request = type("Request", (), {"client": type("Client", (), {"host": "203.0.113.8"})()})()
    with pytest.raises(HTTPException) as exc_info:
        _require_local_request(remote_request)
    assert exc_info.value.status_code == 403


def test_model_catalog_marks_installed_and_excludes_embedding_model(monkeypatch) -> None:
    class FakeOllama:
        def list_models(self):
            return [
                {"name": "qwen2.5:1.5b", "size": 1_000},
                {"name": "bge-m3:latest", "size": 2_000},
            ]

    monkeypatch.setattr(api, "get_ollama", lambda: FakeOllama())
    payload = api.models()
    models_by_name = {item["name"]: item for item in payload["models"]}

    assert models_by_name["qwen2.5:1.5b"]["installed"] is True
    assert models_by_name["llama3.2:latest"]["installed"] is False
    assert "bge-m3:latest" not in models_by_name
