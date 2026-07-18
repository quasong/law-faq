from __future__ import annotations

import json
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import Chunk
from .ollama import OllamaClient


@dataclass(frozen=True)
class SearchResult:
    score: float
    chunk: dict[str, str]


def build_index(
    chunks: list[Chunk],
    index_dir: Path,
    client: OllamaClient,
    embed_model: str,
    batch_size: int = 16,
) -> None:
    if not chunks:
        raise ValueError("沒有可建立索引的法條")
    index_dir.mkdir(parents=True, exist_ok=True)
    fingerprint_hash = hashlib.sha256(embed_model.encode("utf-8"))
    for chunk in chunks:
        fingerprint_hash.update(chunk.id.encode("utf-8"))
        fingerprint_hash.update(chunk.text.encode("utf-8"))
    fingerprint = fingerprint_hash.hexdigest()[:16]
    partial_vectors_path = index_dir / f"embeddings-{fingerprint}.partial.npy"
    final_vectors_path = index_dir / f"embeddings-{fingerprint}.npy"
    chunks_path = index_dir / f"chunks-{fingerprint}.jsonl"
    progress_path = index_dir / f"progress-{fingerprint}.json"

    current_info_path = index_dir / "index.json"
    if current_info_path.exists() and final_vectors_path.exists() and chunks_path.exists():
        current_info = json.loads(current_info_path.read_text(encoding="utf-8"))
        if current_info.get("fingerprint") == fingerprint:
            print(f"索引內容未變更，沿用既有 {len(chunks)} 個向量")
            return

    completed = 0
    matrix: np.memmap
    if partial_vectors_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("embed_model") == embed_model and progress.get("chunk_count") == len(chunks):
            matrix = np.lib.format.open_memmap(partial_vectors_path, mode="r+")
            completed = int(progress.get("completed", 0))
            print(f"從上次進度繼續：{completed}/{len(chunks)}")
        else:
            raise RuntimeError("發現不相容的部分索引；請移走 data/index/progress-* 後重試")
    else:
        first_batch = chunks[:batch_size]
        first_values = client.embeddings(embed_model, [chunk.text for chunk in first_batch])
        first_matrix = np.asarray(first_values, dtype=np.float32)
        if first_matrix.ndim != 2:
            raise RuntimeError("Ollama embedding 回傳的向量維度不正確")
        first_norms = np.linalg.norm(first_matrix, axis=1, keepdims=True)
        first_matrix /= np.maximum(first_norms, 1e-12)
        matrix = np.lib.format.open_memmap(
            partial_vectors_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(chunks), first_matrix.shape[1]),
        )
        matrix[: len(first_batch)] = first_matrix
        matrix.flush()
        completed = len(first_batch)
        progress_path.write_text(
            json.dumps(
                {"embed_model": embed_model, "chunk_count": len(chunks), "completed": completed},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\r向量化：{completed}/{len(chunks)}", end="", flush=True)

    for start in range(completed, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        values = client.embeddings(embed_model, [chunk.text for chunk in batch])
        batch_matrix = np.asarray(values, dtype=np.float32)
        if batch_matrix.shape[1] != matrix.shape[1]:
            raise RuntimeError("Ollama embedding 模型的向量維度在建置中發生變化")
        norms = np.linalg.norm(batch_matrix, axis=1, keepdims=True)
        batch_matrix /= np.maximum(norms, 1e-12)
        end = start + len(batch)
        matrix[start:end] = batch_matrix
        matrix.flush()
        progress_path.write_text(
            json.dumps(
                {"embed_model": embed_model, "chunk_count": len(chunks), "completed": end},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\r向量化：{end}/{len(chunks)}", end="", flush=True)
    print()
    del matrix
    os.replace(partial_vectors_path, final_vectors_path)
    with chunks_path.open("w", encoding="utf-8") as output:
        for chunk in chunks:
            output.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    info_temp = index_dir / "index.partial.json"
    info_temp.write_text(
        json.dumps(
            {
                "mode": "semantic",
                "embed_model": embed_model,
                "chunk_count": len(chunks),
                "fingerprint": fingerprint,
                "embeddings_file": final_vectors_path.name,
                "chunks_file": chunks_path.name,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(info_temp, current_info_path)
    progress_path.unlink(missing_ok=True)


class VectorStore:
    def __init__(self, index_dir: Path):
        info_path = index_dir / "index.json"
        if not info_path.exists():
            raise RuntimeError("找不到向量索引，請先執行 `law-rag ingest`")
        self.info = json.loads(info_path.read_text(encoding="utf-8"))
        vectors_file = self.info.get("embeddings_file", "embeddings.npy")
        chunks_file = self.info.get("chunks_file", "chunks.jsonl")
        self.vectors = np.load(index_dir / vectors_file, mmap_mode="r")
        with (index_dir / chunks_file).open(encoding="utf-8") as source:
            self.chunks = [json.loads(line) for line in source if line.strip()]
        if len(self.chunks) != len(self.vectors):
            raise RuntimeError("索引檔案不一致，請重新執行 ingest")

    def search(self, query_vector: list[float], top_k: int = 6) -> list[SearchResult]:
        query = np.asarray(query_vector, dtype=np.float32)
        query /= max(float(np.linalg.norm(query)), 1e-12)
        if query.shape[0] != self.vectors.shape[1]:
            raise RuntimeError("查詢向量維度不同；embedding 模型變更後必須重建索引")
        scores = self.vectors @ query
        k = min(max(top_k, 1), len(scores))
        indices = np.argpartition(scores, -k)[-k:]
        indices = indices[np.argsort(scores[indices])[::-1]]
        return [SearchResult(score=float(scores[i]), chunk=self.chunks[int(i)]) for i in indices]


def load_store(index_dir: Path) -> "VectorStore | object":
    info_path = index_dir / "index.json"
    if not info_path.exists():
        raise RuntimeError("找不到索引，請先執行 `law-rag ingest`")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("mode") == "fast":
        from .fts_store import FTSStore

        return FTSStore(index_dir, info)
    return VectorStore(index_dir)
