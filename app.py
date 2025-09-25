grep -n "@app.post(\"/search" ~/Desktop/trust-kb/app.py
import os, sqlite3
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.getenv("PINECONE_INDEX","trust-knowledge"))
EMBED_MODEL = os.getenv("EMBED_MODEL","text-embedding-3-small")

app = FastAPI(title="Trust KB Search", version="1.0.0")

class SearchBody(BaseModel):
    query: str
    top_k: int = 8
    filters: dict | None = None

class HydrateBody(BaseModel):
    ids: list[str]

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/search")
def search(b: SearchBody):
    q_vec = client.embeddings.create(model=EMBED_MODEL, input=b.query).data[0].embedding
    res = index.query(
        vector=q_vec, top_k=b.top_k, include_metadata=True,
        filter=b.filters or {}
    )
    hits = []
    for m in res.matches:
        md = m.metadata or {}
        hits.append({
            "id": m.id,
            "doc_id": md.get("doc_id"),
            "title": md.get("title"),
            "page": md.get("page"),
            "version": md.get("version"),
            "jurisdiction": md.get("jurisdiction"),
            "score": m.score
        })
    return {"results": hits}

TEXT_DB_PATH = "text.db"
def get_text_db():
    conn = sqlite3.connect(TEXT_DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY, text TEXT
    );""")
    return conn

@app.post("/hydrate")
def hydrate(b: HydrateBody):
    conn = get_text_db()
    if not b.ids:
        raise HTTPException(status_code=400, detail="No IDs provided")
    qmarks = ",".join(["?"]*len(b.ids))
    cur = conn.execute(f"SELECT id, text FROM chunks WHERE id IN ({qmarks})", b.ids)
    rows = cur.fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="No texts found for given ids")
    return {"texts": [{"id": r[0], "text": r[1]} for r in rows]}
