"""
FastAPI service.

Two endpoints:
  POST /ingest  — upload a document's raw text to be chunked/embedded/stored
  POST /query   — ask a question, get back a grounded, cited, verified answer

Run:
    uvicorn app:app --reload

The retriever (Pinecone connection, Cohere client, BM25 index) is built
ONCE at startup and reused across requests — never re-instantiate it per
request, since loading it is comparatively slow.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from ingest import ingest_text
from retrieval import HybridRetriever
from generate import generate_answer

app = FastAPI(title="Hybrid RAG over Internal Docs")

# Built once at startup, reused for every request.
retriever = HybridRetriever()


class IngestRequest(BaseModel):
    text: str
    source: str


class QueryRequest(BaseModel):
    query: str


@app.post("/ingest")
def ingest(req: IngestRequest):
    n_chunks = ingest_text(req.text, source=req.source)
    # NOTE: the retriever's BM25 index was loaded at startup from disk.
    # Since a new document changes that corpus, reload it so BM25 search
    # picks up what was just ingested — otherwise it would only be
    # visible to Pinecone (vector) search until the process restarts.
    retriever._load_bm25()
    return {"source": req.source, "chunks_created": n_chunks}


@app.post("/query")
def query(req: QueryRequest):
    chunks = retriever.retrieve(req.query)
    result = generate_answer(req.query, chunks)

    return {
        "query": req.query,
        "answer": result["answer"],
        "verified": result["verified"],
        "verification_note": result["verification_note"],
        "sources": [
            {
                "index": i + 1,
                "source": c["metadata"].get("source"),
                "text": c["text"],
                "rerank_score": c.get("rerank_score"),
            }
            for i, c in enumerate(chunks)
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok"}