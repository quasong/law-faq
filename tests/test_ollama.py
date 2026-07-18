import io
import json
import urllib.request

import taiwan_law_rag.ollama as ollama
from taiwan_law_rag.ollama import OllamaClient, canonical_model_name, start_ollama_server


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def test_canonical_model_name_adds_latest_tag() -> None:
    assert canonical_model_name("llama3.2") == "llama3.2:latest"
    assert canonical_model_name("qwen2.5:1.5b") == "qwen2.5:1.5b"


def test_lists_and_checks_installed_models(monkeypatch) -> None:
    payload = json.dumps(
        {"models": [{"name": "qwen2.5:1.5b", "size": 1_000}, {"model": "llama3.2:latest"}]}
    ).encode()
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: FakeResponse(payload))
    client = OllamaClient("http://localhost:11434")

    assert client.model_names() == ["qwen2.5:1.5b", "llama3.2:latest"]
    assert client.is_model_installed("llama3.2") is True
    assert client.is_model_installed("gemma3:1b") is False


def test_streams_model_pull_progress(monkeypatch) -> None:
    payload = b"".join(
        [
            b'{"status":"pulling manifest"}\n',
            b'{"status":"downloading","completed":50,"total":100}\n',
            b'{"status":"success"}\n',
        ]
    )
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: FakeResponse(payload))

    updates = list(OllamaClient("http://localhost:11434").pull_model("llama3.2:latest"))

    assert updates[1]["completed"] == 50
    assert updates[-1]["status"] == "success"


def test_starts_detached_ollama_server(monkeypatch) -> None:
    captured = {}

    class FakeProcess:
        pid = 4321

        def poll(self):
            return None

    def fake_popen(command, **options):
        captured["command"] = command
        captured["options"] = options
        return FakeProcess()

    monkeypatch.setattr(ollama, "_ollama_process", None)
    monkeypatch.setattr(ollama.shutil, "which", lambda command: "/usr/local/bin/ollama")
    monkeypatch.setattr(ollama.subprocess, "Popen", fake_popen)

    assert start_ollama_server() == 4321
    assert captured["command"] == ["/usr/local/bin/ollama", "serve"]
    assert captured["options"]["start_new_session"] is True


def test_start_fails_when_ollama_cli_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(ollama, "_ollama_process", None)
    monkeypatch.setattr(ollama.shutil, "which", lambda command: None)

    try:
        start_ollama_server()
    except RuntimeError as exc:
        assert "找不到 Ollama CLI" in str(exc)
    else:
        raise AssertionError("missing Ollama CLI should fail")
