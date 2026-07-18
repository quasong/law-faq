import io

import taiwan_law_rag.model_catalog as catalog
from taiwan_law_rag.model_catalog import OllamaModelCatalog, parse_catalog_page


def _page(items: str, next_page: str = "") -> str:
    more = f'<li hx-get="{next_page}"></li>' if next_page else ""
    return f"<ul>{items}{more}</ul>"


def _item(name: str, badges: list[tuple[str, str]]) -> str:
    badge_html = "".join(f'<span class="{style}">{text}</span>' for style, text in badges)
    return f'<li><a href="/library/{name}"><span>{name}</span>{badge_html}</a></li>'


def test_parses_downloadable_variants_and_next_page() -> None:
    html = _page(
        _item("qwen3", [("bg-[#ddf4ff]", "4b"), ("bg-[#ddf4ff]", "8b")])
        + _item("embeddinggemma", [("bg-indigo-50", "embedding")])
        + _item("cloud-only", [("bg-cyan-50", "cloud")])
        + _item("llama3.2", []),
        "/search?page=2",
    )

    models, next_page = parse_catalog_page(html)

    assert models == ["qwen3:4b", "qwen3:8b", "llama3.2:latest"]
    assert next_page == "/search?page=2"


def test_fetches_all_catalog_pages_and_deduplicates(monkeypatch) -> None:
    pages = {
        "https://ollama.com/search": _page(
            _item("qwen3", [("bg-[#ddf4ff]", "4b")]), "/search?page=2"
        ),
        "https://ollama.com/search?page=2": _page(
            _item("qwen3", [("bg-[#ddf4ff]", "4b")])
            + _item("gemma3", [("bg-[#ddf4ff]", "4b")])
        ),
    }

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.close()

    def fake_urlopen(request, timeout):
        return FakeResponse(pages[request.full_url].encode())

    monkeypatch.setattr(catalog.urllib.request, "urlopen", fake_urlopen)

    assert OllamaModelCatalog().list_models() == ["qwen3:4b", "gemma3:4b"]
