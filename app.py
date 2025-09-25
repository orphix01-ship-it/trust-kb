import os, sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv(dotenv_path=str(Path.cwd()/".env"), override=True)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX = os.environ["PINECONE_INDEX"]
EMBED_MODEL = os.environ.get("EMBED_MODEL","text-embedding-3-small")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

app = FastAPI(title="Trust KB Search", version="1.0.0")

class SearchBody(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None

class HydrateBody(BaseModel):
    ids: List[str]

@app.get("/healthz")
def healthz(): return {"ok": True}

LEVELS = ["L1","L2","L3","L4","L5"]
PER_LEVEL = 2  # 2 hits per level = 10

@app.post("/search")
def search(b: SearchBody):
    q_vec = client.embeddings.create(model=EMBED_MODEL, input=b.query).data[0].embedding
    def fetch_level(lvl, need):
        if need<=0: return []
        res = index.query(vector=q_vec, top_k=min(need*6,60), include_metadata=True,
                          filter={**(b.filters or {}), "doc_level": lvl})
        seen, picks = set(), []
        for m in res.matches:
            md = m.metadata or {}; key = (md.get("doc_id"), md.get("page"))
            if None in key or key in seen: continue
            seen.add(key); picks.append(m)
            if len(picks)>=need: break
        return picks
    results, missing = [], 0
    for lvl in LEVELS:
        got = fetch_level(lvl, PER_LEVEL); results += got
        if len(got)<PER_LEVEL: missing += PER_LEVEL-len(got)
    if missing>0:
        res = index.query(vector=q_vec, top_k=120, include_metadata=True, filter=b.filters or {})
        seen = {((m.metadata or {}).get("doc_id"), (m.metadata or {}).get("page")) for m in results}
        for m in res.matches:
            md=m.metadata or {}; key=(md.get("doc_id"), md.get("page"))
            if None in key or key in seen: continue
            seen.add(key); results.append(m)
            if len(results)>=len(LEVELS)*PER_LEVEL: break
    out=[]
    for m in results[:len(LEVELS)*PER_LEVEL]:
        md=m.metadata or {}
        out.append({"id":m.id,"doc_id":md.get("doc_id"),"title":md.get("title"),
                    "page":md.get("page"),"version":md.get("version"),
                    "jurisdiction":md.get("jurisdiction"),"doc_level":md.get("doc_level"),
                    "score":m.score})
    return {"results": out}

TEXT_DB_PATH="text.db"
def get_db():
    c=sqlite3.connect(TEXT_DB_PATH)
    c.execute("CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, text TEXT)")
    return c

@app.post("/hydrate")
def hydrate(b: HydrateBody):
    if not b.ids: raise HTTPException(status_code=400, detail="No IDs provided")
    c=get_db(); qmarks=",".join(["?"]*len(b.ids))
    rows=c.execute(f"SELECT id,text FROM chunks WHERE id IN ({qmarks})", b.ids).fetchall()
    if not rows: raise HTTPException(status_code=404, detail="No texts found for given ids")
    return {"texts":[{"id":r[0],"text":r[1]} for r in rows]}
