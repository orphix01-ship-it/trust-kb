import os, glob, uuid, sqlite3, time
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from utils import chunk_text

# Load API keys and settings
load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.getenv("PINECONE_INDEX", "trust-knowledge")
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# Clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create Pinecone index if it doesn’t exist
dims = 1536 if "small" in EMBED_MODEL else 3072
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        INDEX_NAME,
        dimension=dims,
        metric="cosine",
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )
index = pc.Index(INDEX_NAME)

# SQLite to hydrate text
TEXT_DB_PATH = "text.db"
conn = sqlite3.connect(TEXT_DB_PATH)
conn.execute("CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, text TEXT)")

def clean_metadata(md: dict) -> dict:
    out = {}
    for k, v in md.items():
        if v is None:
            continue
        if isinstance(v, list):
            out[k] = [str(x) for x in v if x is not None]
        else:
            out[k] = v
    return out

def embed_texts(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def detect_level(title: str) -> str:
    for lx in ["L1_", "L2_", "L3_", "L4_", "L5_"]:
        if title.startswith(lx):
            return lx.rstrip("_")
    return "L5"

def pdf_to_records(path, base_meta):
    reader = PdfReader(path)
    for pageno, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        for order, chunk in enumerate(chunk_text(raw, max_tokens=900, overlap=120)):
            yield {
                "id": str(uuid.uuid4()),
                "text": chunk,
                "metadata": {
                    **base_meta,
                    "page": pageno,
                    "order": order,
                },
            }

def save_texts(batch):
    conn.executemany(
        "INSERT OR REPLACE INTO chunks (id, text) VALUES (?, ?)",
        [(b["id"], b["text"]) for b in batch],
    )
    conn.commit()

def upsert_batch(batch):
    if not batch:
        return
    vecs = embed_texts([b["text"] for b in batch])
    to_upsert = []
    for rec, vec in zip(batch, vecs):
        md = clean_metadata(rec["metadata"].copy())
        to_upsert.append({"id": rec["id"], "values": vec, "metadata": md})
    index.upsert(vectors=to_upsert)
    time.sleep(0.5)  # avoid 429 errors
    save_texts(batch)

def ingest_folder(folder="docs", version="v1", jurisdiction=None, tags=None):
    pdfs = glob.glob(os.path.join(folder, "*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {folder}/")
        return
    batch = []
    for path in pdfs:
        title = os.path.splitext(os.path.basename(path))[0]
        level = detect_level(title)
        base_meta = {
            "doc_id": f"{title}-{version}",
            "title": title,
            "version": version,
            "jurisdiction": (jurisdiction or []),
            "tags": (tags or []),
            "version_active": True,
            "doc_level": level,
        }
        print(f"Processing {os.path.basename(path)} (level={level})")
        for rec in pdf_to_records(path, base_meta):
            batch.append(rec)
            if len(batch) == 16:
                upsert_batch(batch)
                print("  ..batch upserted")
                batch = []
    upsert_batch(batch)
    print("  ..final batch upserted")
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_folder(folder="docs", version="v1")

