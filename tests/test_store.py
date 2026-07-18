from pathlib import Path

from taiwan_law_rag.models import Chunk
from taiwan_law_rag.store import VectorStore, build_index


class FakeClient:
    def embeddings(self, model: str, texts: list[str]) -> list[list[float]]:
        del model
        return [[float("言論" in text), float("財產" in text), 0.1] for text in texts]


def _chunk(identifier: str, text: str) -> Chunk:
    return Chunk(
        id=identifier,
        text=text,
        law_name="測試法",
        article_number=identifier,
        section="",
        url="https://example.invalid",
        modified_date="20260718",
        nature="法律",
        category="測試",
        dataset="laws",
    )


def test_build_and_search_index(tmp_path: Path) -> None:
    chunks = [_chunk("第 1 條", "保障言論自由"), _chunk("第 2 條", "保障財產權")]
    build_index(chunks, tmp_path, FakeClient(), "fake", batch_size=1)
    store = VectorStore(tmp_path)
    results = store.search([1.0, 0.0, 0.1], top_k=1)
    assert results[0].chunk["article_number"] == "第 1 條"
    assert store.info["fingerprint"]

