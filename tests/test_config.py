from taiwan_law_rag.config import Settings


def test_default_chat_model_is_qwen(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_CHAT_MODEL", raising=False)
    assert Settings.from_env().chat_model == "qwen2.5:1.5b"
