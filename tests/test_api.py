import json

from taiwan_law_rag.api import _event, home


def test_home_contains_streaming_llm_interface() -> None:
    html = home()
    assert 'id="ask-form"' in html
    assert "'/ask/stream'" in html
    assert "停止生成" in html
    assert "法規來源" in html


def test_ndjson_event() -> None:
    line = _event({"type": "delta", "text": "法規"})
    assert line.endswith("\n")
    assert json.loads(line) == {"type": "delta", "text": "法規"}
