# 法律Q&A：台灣法規 RAG

以法務部「全國法規資料庫」官方開放 XML 建立本地法規文件與檢索索引，並使用 Ollama 的 `qwen2.5:1.5b` 生成繁體中文回答。每個回答會附回官方法規頁的來源。

> 本專案是法規檢索工具，不構成法律意見。法規有時間效力、適用範圍與個案事實問題；涉及重大權益時，請向台灣執業律師確認。

## 資料範圍

預設下載並匯入：

- 中文法規－法律（憲法、法律等）
- 中文法規－命令（法規命令）
- 僅建立未標示廢止的法規索引；可用 `--include-abolished` 改變
- 原始 ZIP/XML 保存在 `data/raw/`
- 每部法規轉成一份 Markdown，保存在 `data/documents/`
- 每條法條是一個檢索單位；過長法條會分段，但保留法規名、條號與章節資訊

官方資料集每月更新。重新執行 ingest 即可下載新版並重建文件與索引。

## 需求

- Python 3.10+
- [Ollama](https://ollama.com/)
- 約 2–4 GB 額外磁碟空間（模型、原始資料與索引；依 Ollama 模型版本而異）

`qwen2.5:1.5b` 負責生成回答。預設快速模式使用 SQLite FTS5 中文 bigram/BM25，不需要 embedding 模型；只有選用較慢的完整語意模式時，才會使用 `bge-m3:latest`。

## 安裝

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ".[dev]"

ollama pull qwen2.5:1.5b

# 只有要使用 --mode semantic 時才需要
ollama pull bge-m3:latest
```

如果 Ollama 未在背景執行：

```bash
ollama serve
```

## 下載法規並建立索引

```bash
law-rag ingest
```

預設 `fast` 模式不會逐條呼叫 embedding 模型，通常幾分鐘內可完成，而且建庫時不會長時間吃滿 CPU/GPU。若只想先匯入法律：

```bash
law-rag ingest --dataset laws
```

需要較強的同義詞與語意檢索時，可選擇完整向量模式。這會慢很多，但支援中斷續跑：

```bash
law-rag ingest --dataset laws --mode semantic
```

其他選項：

```bash
# 使用已下載的 XML 重建，不重新連線下載
law-rag ingest --dataset laws --skip-download --mode fast

# 納入已廢止法規（回答歷史法問題時才建議）
law-rag ingest --include-abolished
```

## 使用方式

命令列：

```bash
law-rag ask "民法對成年年齡如何規定？"
```

Web UI 與 JSON API：

```bash
uvicorn taiwan_law_rag.api:app --host 127.0.0.1 --port 8000
```

打開 `http://127.0.0.1:8000`，或呼叫：

Web UI 會逐字呈現本地模型輸出、顯示可展開的法規來源，且每次新問題會取代上一題，不保存對話歷史。使用者可以從「回答模型」欄位選擇已安裝模型，或輸入其他 Ollama 模型名稱；若模型尚未安裝，介面會先請求確認，再串流顯示下載進度，完成後自動開始回答。

基於安全考量，模型部署端點只接受從 `localhost` 發出的請求。若把問答網站公開到網路，訪客不能透過此端點任意下載模型或消耗主機磁碟空間。

`POST /ask` 提供一般 JSON 回應；`POST /ask/stream` 提供 NDJSON 串流回應。兩者都可以傳入 `model` 選擇回答模型：

```bash
curl http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"雇主可以任意解僱勞工嗎？","top_k":6,"model":"qwen2.5:1.5b"}'
```

模型管理 API：

- `GET /models`：列出建議模型與本機安裝狀態
- `POST /models/pull/stream`：確認後從 Ollama 串流下載模型，僅限本機呼叫

## 設定

可用環境變數覆寫預設值：

| 變數 | 預設值 | 說明 |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API |
| `OLLAMA_CHAT_MODEL` | `qwen2.5:1.5b` | 回答生成模型 |
| `OLLAMA_EMBED_MODEL` | `bge-m3:latest` | 僅 semantic 模式使用；改變後須重建索引 |
| `LAW_RAG_DATA_DIR` | `./data` | 原始資料、文件與索引位置 |

## 專案結構

```text
src/taiwan_law_rag/
  sources.py      # 官方資料下載
  parser.py       # 串流解析 XML
  documents.py    # Markdown 與法條 chunks
  fts_store.py    # 快速中文 bigram/BM25 索引（預設）
  store.py        # 可選的 NumPy cosine 向量索引
  ollama.py       # 本地 Ollama API client
  rag.py          # 檢索、提示詞、回答與引用
  api.py          # FastAPI 與簡易 Web UI
  cli.py          # ingest / ask 指令
```

## 已知範圍與後續擴充

目前索引的是中央法規條文，不包含地方自治法規、裁判書、憲法法庭判決、行政函釋或法規附件內的 PDF/表格。若要做正式產品，建議接著加入：法規版本/生效日查詢、地方自治法規、司法實務資料、reranker、檢索評測集、存取日標記與定期更新工作。

## 資料來源與授權

- [政府資料開放平臺：中文法規－法律](https://data.gov.tw/dataset/18289)
- [政府資料開放平臺：中文法規－命令](https://data.gov.tw/dataset/18290)
- 資料授權：政府資料開放授權條款－第 1 版
