from taiwan_law_rag.rag import add_deterministic_citations
from taiwan_law_rag.store import SearchResult


def _source(article: str) -> SearchResult:
    return SearchResult(
        score=0.9,
        chunk={
            "law_name": "中華民國憲法",
            "article_number": article,
            "url": "https://example.invalid",
            "modified_date": "19470101",
            "text": "測試",
        },
    )


def test_adds_missing_citation_to_named_article() -> None:
    text = "結論：憲法第 11 條保障言論自由。\n\n限制：仍須視個案而定。"
    result = add_deterministic_citations(text, [_source("第 11 條")])
    assert "言論自由。 [來源1]" in result
    assert "限制：仍須視個案而定。 [來源1]" not in result


def test_keeps_model_citation() -> None:
    text = "憲法保障言論自由。[來源1]"
    assert add_deterministic_citations(text, [_source("第 11 條")]) == text

