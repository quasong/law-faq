from pathlib import Path

from taiwan_law_rag.fts_store import FTSStore, build_fts_index
from taiwan_law_rag.models import Chunk


def _chunk(identifier: str, article: str, text: str) -> Chunk:
    return Chunk(
        id=identifier,
        text=text,
        law_name="中華民國憲法",
        article_number=article,
        section="人民之權利義務",
        url="https://example.invalid",
        modified_date="19470101",
        nature="憲法",
        category="憲法",
        dataset="laws",
    )


def test_fast_chinese_search(tmp_path: Path) -> None:
    chunks = [
        _chunk("speech", "第 11 條", "人民有言論、講學、著作及出版之自由。"),
        _chunk("property", "第 15 條", "人民之生存權、工作權及財產權，應予保障。"),
    ]
    build_fts_index(chunks, tmp_path)
    results = FTSStore(tmp_path).search("憲法如何保障人民言論自由？", top_k=2)
    assert results
    assert results[0].chunk["article_number"] == "第 11 條"


def test_fast_search_single_character_query_returns_empty(tmp_path: Path) -> None:
    build_fts_index([_chunk("speech", "第 11 條", "人民有言論自由。")], tmp_path)
    assert FTSStore(tmp_path).search("言", top_k=2) == []


def test_expands_common_termination_term(tmp_path: Path) -> None:
    chunks = [
        _chunk("labor", "第 11 條", "雇主非有法定情事，不得預告勞工終止勞動契約。"),
        _chunk("other", "第 2 條", "大量解僱勞工應依規定辦理。"),
    ]
    build_fts_index(chunks, tmp_path)
    results = FTSStore(tmp_path).search("雇主可以任意解僱勞工嗎？", top_k=2)
    assert results[0].chunk["id"] == "labor"
