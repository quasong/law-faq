from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import Settings
from .rag import LawRAG


app = FastAPI(title="台灣法規 RAG", version="0.1.0")


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=6, ge=1, le=15)


@lru_cache(maxsize=1)
def get_rag() -> LawRAG:
    return LawRAG(Settings.from_env())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(payload: Question) -> dict[str, object]:
    try:
        answer = get_rag().ask(payload.question, top_k=payload.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "answer": answer.text,
        "sources": [
            {
                "score": source.score,
                "law_name": source.chunk["law_name"],
                "article_number": source.chunk["article_number"],
                "modified_date": source.chunk["modified_date"],
                "url": source.chunk["url"],
                "text": source.chunk["text"],
            }
            for source in answer.sources
        ],
    }


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>台灣法規 RAG</title><style>
body{max-width:900px;margin:40px auto;padding:0 20px;font-family:system-ui;color:#17212b;background:#f7f8fa}
textarea,button{font:inherit}textarea{width:100%;min-height:110px;padding:12px;box-sizing:border-box}
button{margin-top:10px;padding:10px 20px;background:#075985;color:white;border:0;border-radius:6px;cursor:pointer}
.card{background:white;padding:20px;border-radius:10px;box-shadow:0 2px 10px #0001;margin:16px 0;white-space:pre-wrap}
.muted{color:#64748b;font-size:.9rem}a{color:#0369a1}</style></head>
<body><h1>台灣法規 RAG</h1><p class="muted">資料來自法務部公開資料。回答僅供法規檢索參考，不構成法律意見。</p>
<textarea id="q" placeholder="例如：雇主可以任意解僱勞工嗎？"></textarea><br><button id="send">查詢</button>
<div id="result"></div><script>
const send=document.querySelector('#send'),result=document.querySelector('#result');
send.onclick=async()=>{const question=document.querySelector('#q').value.trim();if(!question)return;
send.disabled=true;result.innerHTML='<div class="card">查詢中…</div>';
try{const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question})});
const d=await r.json();if(!r.ok)throw new Error(d.detail||'查詢失敗');
const sources=d.sources.map((s,i)=>`<div class="card"><b>來源 ${i+1}：${esc(s.law_name)} ${esc(s.article_number)}</b><br><a target="_blank" href="${esc(s.url)}">官方法規頁</a><p>${esc(s.text)}</p></div>`).join('');
result.innerHTML=`<div class="card">${esc(d.answer)}</div><h2>檢索來源</h2>${sources}`;
}catch(e){result.innerHTML=`<div class="card">${esc(e.message)}</div>`}finally{send.disabled=false}};
function esc(v){const d=document.createElement('div');d.textContent=v;return d.innerHTML}
</script></body></html>"""

