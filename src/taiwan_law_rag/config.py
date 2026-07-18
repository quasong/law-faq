from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    ollama_base_url: str
    chat_model: str
    embed_model: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_dir=Path(os.getenv("LAW_RAG_DATA_DIR", "./data")).resolve(),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            chat_model=os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:1.5b"),
            embed_model=os.getenv("OLLAMA_EMBED_MODEL", "bge-m3:latest"),
        )

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"
