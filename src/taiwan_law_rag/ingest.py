from __future__ import annotations

import itertools

from .config import Settings
from .documents import write_documents_and_chunks
from .fts_store import build_fts_index
from .ollama import OllamaClient
from .parser import iter_laws
from .sources import SOURCES, download_dataset
from .store import build_index


def ingest(
    settings: Settings,
    datasets: list[str],
    include_abolished: bool = False,
    max_chars: int = 1200,
    batch_size: int = 16,
    skip_download: bool = False,
    mode: str = "fast",
) -> tuple[int, int]:
    xml_paths = []
    for dataset in datasets:
        source = SOURCES[dataset]
        if skip_download:
            candidates = sorted((settings.raw_dir / dataset).glob("*.xml"))
            if not candidates:
                raise RuntimeError(f"找不到 {dataset} 的本地 XML，請移除 --skip-download")
            xml_path = candidates[0]
        else:
            print(f"下載：{source.label}")
            xml_path = download_dataset(source, settings.raw_dir)
        xml_paths.append((dataset, xml_path))

    laws = itertools.chain.from_iterable(
        iter_laws(path, dataset=dataset, include_abolished=include_abolished)
        for dataset, path in xml_paths
    )
    chunks, law_count = write_documents_and_chunks(laws, settings.documents_dir, max_chars=max_chars)
    print(f"已輸出 {law_count} 部法規、{len(chunks)} 個法條片段至 {settings.documents_dir}")
    if mode == "fast":
        build_fts_index(chunks, settings.index_dir)
    elif mode == "semantic":
        client = OllamaClient(settings.ollama_base_url)
        build_index(chunks, settings.index_dir, client, settings.embed_model, batch_size=batch_size)
    else:
        raise ValueError(f"不支援的索引模式：{mode}")
    return law_count, len(chunks)
