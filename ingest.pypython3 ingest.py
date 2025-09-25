import os, glob, uuid, sqlite3, time
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from utils import chunk_text

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.getenv("PINECONE_INDEX", "trust-knowledge")
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# OpenAI / Pinecone clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist (dim must match embedding model)
dims = 1536 if "small" in EMBED_MODEL else 3072
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        INDEX_NAME,
        dimension=dims,
        metric="cosine",
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )
index = pc.Index(INDEX_NAME)

# SQLite for hydration text
TEXT_DB_PATH = "text.db"
conn = sqlite3.connect(TEXT_DB_PATH)
conn.execute("CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, text TEXT)")

def clean_metadata(md: dict) -> dict:
    """Strip None and normalize list types for Pinecone metadata."""
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

def detect_level_from_title(title: str) -> str:
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
    time.sleep(0.5)  # be gentle with rate limits
    save_texts(batch)

def ingest_folder(folder="docs", version="v1", jurisdiction=None, tags=None):
    pdfs = glob.glob(os.path.join(folder, "**/*.pdf"), recursive=True) or glob.glob(os.path.join(folder, 
"*.pdf"))
    if not pdfs:
        print(f"No PDFs found in ./{folder}. Put files there and re-run.")
        return
    batch = []
    for path in pdfs:
        title = os.path.splitext(os.path.basename(path))[0]
        level = detect_level_from_title(title)
        base_meta = {
            "doc_id": f"{title}-{version}",
            "title": title,
            "version": version,
            "jurisdiction": (jurisdiction or []),  # must not be None
            "tags": (tags or []),                  # must not be None
            "version_active": True,
            "doc_level": level,
        }
        print(f"Processing: {os.path.basename(path)}  (level={level})")
        for rec in pdf_to_records(path, base_meta):
            batch.append(rec)
            if len(batch) == 16:   # small bursts to reduce 429s
                upsert_batch(batch)
                print("  ..batch upserted")
                batch = []
    upsert_batch(batch)
    print("  ..final batch upserted")
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_folder(folder="docs", version="v1")


python3 -c "import os; from dotenv import load_dotenv; load_dotenv(); from pinecone import Pinecone; 
pc=Pinecone(api_key=os.environ['PINECONE_API_KEY']); idx=pc.Index(os.environ['PINECONE_INDEX']); 
print(idx.describe_index_stats())"
import os, glob, uuid, sqlite3 from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from utils import chunk_text

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.getenv("PINECONE_INDEX", "trust-knowledge")
CLOUD = os.getenv("PINECONE_CLOUD","aws")
REGION = os.getenv("PINECONE_REGION","us-east-1")
EMBED_MODEL = os.getenv("EMBED_MODEL","text-embedding-3-small")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
dims = 1536 if 'small' in EMBED_MODEL else 3072
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(INDEX_NAME, dimension=dims, metric="cosine",
                    spec=ServerlessSpec(cloud=CLOUD, region=REGION))
index = pc.Index(INDEX_NAME)

# SQLite for hydration text
TEXT_DB_PATH = "text.db"
conn = sqlite3.connect(TEXT_DB_PATH)
conn.execute("CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, text TEXT)")

def embed_texts(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def pdf_to_records(path, base_meta):
    reader = PdfReader(path)
    for p, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        for i, chunk in enumerate(chunk_text(raw)):
            yield {
                "id": str(uuid.uuid4()),
                "text": chunk,
                "metadata": {
                    **base_meta,
                    "page": p,
                    "order": i
                }
            }

def save_texts(batch):
    conn.executemany("INSERT OR REPLACE INTO chunks (id, text) VALUES (?,?)",
                     [(b["id"], b["text"]) for b in batch])
    conn.commit()

def clean_metadata(md):
    out={}
    for k,v in md.items():
        if v is None: continue
        if isinstance(v,list): out[k]=[str(x) for x in v if x is not None]
        else: out[k]=v
    return out

def upsert_batch(batch):
    if not batch:
        return
    vecs = embed_texts([b["text"] for b in batch])
    to_upsert = []
    for rec, vec in zip(batch, vecs):
        md = clean_metadata(rec["metadata"].copy())
        to_upsert.append({"id": rec["id"], "values": vec, "metadata": md})
    index.upsert(vectors=to_upsert)
    save_texts(batch)

def ingest_folder(folder="docs", version="v1", jurisdiction=None, tags=None):
    pdfs = glob.glob(os.path.join(folder, "**/*.pdf"), recursive=True)
    if not pdfs:
        print(f"No PDFs found in ./{folder}. Put files there and re-run.")
        return
    batch = []
    for path in pdfs:
        title = os.path.splitext(os.path.basename(path))[0]
        base_meta = {
            "doc_id": f"{title}-{version}",
            "title": title,
            "version": version,
            "jurisdiction": (jurisdiction or []),
            "tags": (tags or []),
            "version_active": True
        }
        for rec in pdf_to_records(path, base_meta):
            batch.append(rec)
            if len(batch) == 64:
                upsert_batch(batch); batch = []
    upsert_batch(batch)
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_folder(folder="docs", version="v1")
