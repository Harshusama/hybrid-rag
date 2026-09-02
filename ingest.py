"""
Ingestion pipeline (OFFLINE — run when documents are added/changed)
---------------------------------------------------------------------
Documents -> chunk (LangChain splitter) -> embed (Cohere) -> store in
Pinecone, and separately persist the raw chunk text so we can build a
BM25 keyword index at query time (Pinecone stores vectors only, not a
full-text/keyword index).

Run this as a script:
    python ingest.py doc1.txt doc2.txt
"""

import json
import os
import sys

from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from chunking import load_and_chunk
import config


def get_pinecone_index():
    """
    Connects to Pinecone and creates the index if it doesn't exist yet.
    The index dimension MUST match the embedding model's output size
    (1024 for Cohere's embed-english-v3.0) — this can't be changed later
    without recreating the index.
    """
    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    existing = [idx["name"] for idx in pc.list_indexes()]
    if config.PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=config.EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )

    return pc.Index(config.PINECONE_INDEX_NAME)


def get_embeddings():
    # input_type="search_document" tells Cohere's model this text is being
    # indexed for later retrieval (as opposed to being a search query) —
    # Cohere's v3 embedding models are trained to encode queries and
    # documents slightly differently for better retrieval accuracy.
    return CohereEmbeddings(
        cohere_api_key=config.COHERE_API_KEY,
        model=config.COHERE_EMBED_MODEL,
    )


def _load_bm25_corpus() -> dict:
    if os.path.exists(config.BM25_CORPUS_PATH):
        with open(config.BM25_CORPUS_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_bm25_corpus(corpus: dict):
    with open(config.BM25_CORPUS_PATH, "w") as f:
        json.dump(corpus, f)


def ingest_text(text: str, source: str, extra_metadata: dict | None = None) -> int:
    """
    Chunk a document, embed the chunks into Pinecone, and append them to
    the local BM25 corpus. Returns the number of chunks created.
    """
    documents = load_and_chunk(text, source=source, extra_metadata=extra_metadata)
    if not documents:
        return 0

    embeddings = get_embeddings()
    index = get_pinecone_index()

    # PineconeVectorStore.from_documents embeds each Document's text and
    # upserts (vector, metadata) pairs into the index in one call.
    PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=config.PINECONE_INDEX_NAME,
    )

    # Mirror the same chunks into the local BM25 corpus, keyed by a stable id.
    corpus = _load_bm25_corpus()
    for doc in documents:
        chunk_id = f"{source}::{doc.metadata['chunk_index']}"
        corpus[chunk_id] = {
            "text": doc.page_content,
            "metadata": doc.metadata,
        }
    _save_bm25_corpus(corpus)

    return len(documents)


def ingest_file(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return ingest_text(text, source=os.path.basename(path))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <file1.txt> [file2.txt ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        n = ingest_file(path)
        print(f"Ingested {path}: {n} chunks")