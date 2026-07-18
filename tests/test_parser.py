from pathlib import Path

from taiwan_law_rag.documents import write_documents_and_chunks
from taiwan_law_rag.parser import iter_laws


FIXTURE = Path(__file__).parent / "fixtures" / "sample.xml"


def test_parser_excludes_abolished_by_default() -> None:
    laws = list(iter_laws(FIXTURE, dataset="laws"))
    assert len(laws) == 1
    assert laws[0].name == "測試法"
    assert laws[0].articles[0].section == "第 一 章 總則"
    assert laws[0].articles[1].text == "第一項。\\n第二項。"


def test_parser_can_include_abolished() -> None:
    laws = list(iter_laws(FIXTURE, dataset="laws", include_abolished=True))
    assert len(laws) == 2
    assert laws[1].abolished is True


def test_writes_markdown_and_chunks(tmp_path: Path) -> None:
    laws = list(iter_laws(FIXTURE, dataset="laws"))
    chunks, count = write_documents_and_chunks(laws, tmp_path, max_chars=1200)
    assert count == 1
    assert len(chunks) == 2
    document = tmp_path / "laws" / "Z0000001.md"
    assert document.exists()
    assert "# 測試法" in document.read_text(encoding="utf-8")
    assert (tmp_path / "manifest.jsonl").exists()

