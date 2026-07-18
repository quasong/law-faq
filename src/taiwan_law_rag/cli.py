from __future__ import annotations

import argparse
import json

from .config import Settings
from .ingest import ingest
from .rag import LawRAG
from .sources import SOURCES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="台灣法規 RAG")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="下載法規、輸出文件並建立檢索索引")
    ingest_parser.add_argument(
        "--dataset", choices=[*SOURCES, "all"], default="all", help="預設同時匯入法律與命令"
    )
    ingest_parser.add_argument("--include-abolished", action="store_true", help="包含已廢止法規")
    ingest_parser.add_argument("--skip-download", action="store_true", help="使用 data/raw 既有 XML")
    ingest_parser.add_argument("--max-chars", type=int, default=1200, help="單一法條片段最大字數")
    ingest_parser.add_argument(
        "--batch-size", type=int, default=16, help="semantic 模式的 Ollama embedding 批次大小"
    )
    ingest_parser.add_argument(
        "--mode",
        choices=["fast", "semantic"],
        default="fast",
        help="fast 使用 SQLite 全文檢索（預設）；semantic 預先向量化全部法條",
    )

    ask_parser = subparsers.add_parser("ask", help="詢問法規問題")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=6)
    ask_parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    settings = Settings.from_env()
    if args.command == "ingest":
        datasets = list(SOURCES) if args.dataset == "all" else [args.dataset]
        ingest(
            settings,
            datasets=datasets,
            include_abolished=args.include_abolished,
            max_chars=args.max_chars,
            batch_size=args.batch_size,
            skip_download=args.skip_download,
            mode=args.mode,
        )
        return
    rag = LawRAG(settings)
    answer = rag.ask(args.question, top_k=args.top_k)
    payload = {
        "answer": answer.text,
        "sources": [
            {
                "score": source.score,
                "law_name": source.chunk["law_name"],
                "article_number": source.chunk["article_number"],
                "url": source.chunk["url"],
            }
            for source in answer.sources
        ],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(answer.text)
        print("\n檢索來源：")
        for index, source in enumerate(answer.sources, start=1):
            chunk = source.chunk
            print(f"[{index}] {chunk['law_name']} {chunk['article_number']} {chunk['url']}")


if __name__ == "__main__":
    main()
