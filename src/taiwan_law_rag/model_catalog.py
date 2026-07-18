from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse


CATALOG_URL = "https://ollama.com/search"
CATALOG_ORIGIN = "https://ollama.com"


class _CatalogPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.models: list[str] = []
        self.next_page: str | None = None
        self._item_depth = 0
        self._item_name = ""
        self._item_variants: list[str] = []
        self._item_embedding = False
        self._item_cloud = False
        self._span_class = ""
        self._span_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "li":
            next_page = attributes.get("hx-get")
            if next_page and next_page.startswith("/search?page="):
                self.next_page = next_page
            if self._item_depth == 0:
                self._item_name = ""
                self._item_variants = []
                self._item_embedding = False
                self._item_cloud = False
            self._item_depth += 1
        if not self._item_depth:
            return
        if tag == "a":
            href = attributes.get("href") or ""
            if href.startswith("/library/"):
                self._item_name = href.removeprefix("/library/").strip("/")
        if tag == "span":
            self._span_class = attributes.get("class") or ""
            self._span_text = []

    def handle_data(self, data: str) -> None:
        if self._item_depth and self._span_class:
            self._span_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "span" and self._item_depth and self._span_class:
            value = "".join(self._span_text).strip().lower()
            if "bg-[#ddf4ff]" in self._span_class and value:
                self._item_variants.append(value)
            if "bg-indigo-50" in self._span_class and value == "embedding":
                self._item_embedding = True
            if "bg-cyan-50" in self._span_class and value == "cloud":
                self._item_cloud = True
            self._span_class = ""
            self._span_text = []
        if tag != "li" or not self._item_depth:
            return
        self._item_depth -= 1
        if self._item_depth:
            return
        if not self._item_name or self._item_embedding:
            return
        variants = list(dict.fromkeys(self._item_variants))
        if variants:
            self.models.extend(f"{self._item_name}:{variant}" for variant in variants)
        elif not self._item_cloud:
            self.models.append(f"{self._item_name}:latest")


def parse_catalog_page(html: str) -> tuple[list[str], str | None]:
    parser = _CatalogPageParser()
    parser.feed(html)
    return parser.models, parser.next_page


class OllamaModelCatalog:
    def __init__(self, timeout: int = 12, cache_seconds: int = 6 * 60 * 60):
        self.timeout = timeout
        self.cache_seconds = cache_seconds
        self._models: list[str] = []
        self._fetched_at = 0.0
        self._lock = threading.Lock()

    def list_models(self) -> list[str]:
        now = time.monotonic()
        if self._models and now - self._fetched_at < self.cache_seconds:
            return list(self._models)
        with self._lock:
            now = time.monotonic()
            if self._models and now - self._fetched_at < self.cache_seconds:
                return list(self._models)
            models = self._fetch_all_pages()
            self._models = models
            self._fetched_at = time.monotonic()
            return list(models)

    def _fetch_all_pages(self) -> list[str]:
        page_url: str | None = CATALOG_URL
        visited: set[str] = set()
        models: list[str] = []
        while page_url and page_url not in visited and len(visited) < 50:
            visited.add(page_url)
            request = urllib.request.Request(
                page_url,
                headers={
                    "User-Agent": "law-faq/0.1 (+https://ollama.com/search)",
                    "HX-Request": "true",
                },
                method="GET",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    html = response.read().decode("utf-8", errors="replace")
            except (OSError, urllib.error.URLError) as exc:
                if models:
                    break
                raise RuntimeError(f"無法取得 Ollama 官方模型目錄：{exc}") from exc
            page_models, next_page = parse_catalog_page(html)
            models.extend(page_models)
            page_url = urljoin(CATALOG_ORIGIN, next_page) if next_page else None
            if page_url and urlparse(page_url).netloc != "ollama.com":
                page_url = None
        unique_models = list(dict.fromkeys(models))
        if not unique_models:
            raise RuntimeError("Ollama 官方模型目錄未回傳可下載模型")
        return unique_models
