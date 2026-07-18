from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

from .models import Article, Law


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return "\n".join(line.rstrip() for line in "".join(element.itertext()).strip().splitlines()).strip()


def _is_abolished(value: str) -> bool:
    compact = re.sub(r"\s+", "", value)
    return bool(compact and compact not in {"N", "否", "無"})


def iter_laws(xml_path: Path, dataset: str, include_abolished: bool = False) -> Iterator[Law]:
    """Stream laws from the Ministry of Justice XML without loading it all in memory."""
    context = ET.iterparse(xml_path, events=("end",))
    for _, element in context:
        if element.tag != "法規":
            continue
        articles: list[Article] = []
        current_section = ""
        content = element.find("法規內容")
        if content is not None:
            for child in content:
                if child.tag == "編章節":
                    current_section = _text(child)
                elif child.tag == "條文":
                    number = _text(child.find("條號"))
                    article_text = _text(child.find("條文內容"))
                    if article_text:
                        articles.append(Article(number=number, text=article_text, section=current_section))

        abolished = _is_abolished(_text(element.find("廢止註記")))
        if articles and (include_abolished or not abolished):
            yield Law(
                name=_text(element.find("法規名稱")),
                nature=_text(element.find("法規性質")),
                category=_text(element.find("法規類別")),
                url=_text(element.find("法規網址")),
                modified_date=_text(element.find("最新異動日期")),
                effective_date=_text(element.find("生效日期")),
                effective_note=_text(element.find("生效內容")),
                abolished=abolished,
                history=_text(element.find("沿革內容")),
                preamble=_text(element.find("前言")),
                articles=tuple(articles),
                dataset=dataset,
            )
        element.clear()

