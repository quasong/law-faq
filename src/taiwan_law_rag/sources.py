from __future__ import annotations

import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSource:
    key: str
    label: str
    url: str


SOURCES: dict[str, DatasetSource] = {
    "laws": DatasetSource(
        key="laws",
        label="中文法規－法律",
        url="https://sendlaw.moj.gov.tw/PublicData/GetFile.ashx?AuData=CF&DType=XML",
    ),
    "orders": DatasetSource(
        key="orders",
        label="中文法規－命令",
        url="https://sendlaw.moj.gov.tw/PublicData/GetFile.ashx?AuData=CM&DType=XML",
    ),
}


def download_dataset(source: DatasetSource, raw_dir: Path, timeout: int = 180) -> Path:
    """Download and extract one official ZIP, returning its XML path."""
    target_dir = raw_dir / source.key
    target_dir.mkdir(parents=True, exist_ok=True)
    archive = target_dir / f"{source.key}.zip"
    request = urllib.request.Request(
        source.url,
        headers={"User-Agent": "taiwan-law-rag/0.1 (+official-open-data-client)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"下載 {source.label} 失敗：{exc}") from exc

    if not zipfile.is_zipfile(archive):
        raise RuntimeError(f"{source.label} 回傳的檔案不是有效 ZIP：{archive}")
    with zipfile.ZipFile(archive) as zf:
        xml_names = [name for name in zf.namelist() if name.lower().endswith(".xml")]
        if not xml_names:
            raise RuntimeError(f"{source.label} ZIP 中找不到 XML")
        # The official archive has one XML payload. Avoid extracting arbitrary paths.
        xml_name = xml_names[0]
        destination = target_dir / Path(xml_name).name
        with zf.open(xml_name) as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return destination

