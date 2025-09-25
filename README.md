# Trust KB API

FastAPI service that searches Pinecone (L1–L5 precedence-aware) and hydrates chunks for GPT Actions.

## Endpoints
- POST `/search` – quota + dedupe (2 per level L1..L5)
- POST `/hydrate` – return full text for chunk ids
- GET  `/healthz` – health check

## Env vars (set in deployment, not committed)
OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX, PINECONE_CLOUD, PINECONE_REGION, EMBED_MODEL
