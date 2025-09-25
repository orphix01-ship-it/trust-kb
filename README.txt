=== Quick Start (Mac/Windows) ===

1) Unzip this folder somewhere (e.g., Desktop/trust-kb).
2) Open a terminal:
   - Mac: Terminal app
   - Windows: PowerShell
3) Change into the folder:
   cd path/to/trust-kb

4) Create and activate a virtual environment:
   python -m venv .venv
   # Mac:
   source .venv/bin/activate
   # Windows:
   .venv\Scripts\Activate

5) Install deps:
   pip install -r requirements.txt

6) Create .env from example and fill keys:
   copy .env.example to .env
   open .env and fill OPENAI_API_KEY and PINECONE_API_KEY, etc.

7) Put your PDFs into the docs/ folder.

8) Run ingestion:
   python ingest.py
   # Expect: 'Ingestion complete.'

9) Run API locally:
   uvicorn app:app --reload --port 8080
   Visit http://localhost:8080/healthz

10) Wire into your Custom GPT via Actions using openapi.json.
