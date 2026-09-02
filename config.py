"""
Central configuration.

All tunable values live here so you can iterate on chunk size, top-k, and
the RRF constant without hunting through every file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API keys (set these in a .env file — see .env.example) ----------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

# --- Pinecone ----------------------------------------------------------------
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "internal-docs")
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# --- Embeddings ----------------------------------------------------------
# Cohere's embed-english-v3.0 outputs 1024-dim vectors. This MUST match
# the Pinecone index dimension you create — see ingest.py.
COHERE_EMBED_MODEL = "embed-english-v3.0"
EMBEDDING_DIMENSION = 1024

# --- Chunking ------------------------------------------------------------
CHUNK_SIZE = 500          # characters per chunk (LangChain splitter default unit)
CHUNK_OVERLAP = 100        # character overlap between adjacent chunks

# --- Retrieval -----------------------------------------------------------
VECTOR_TOP_K = 20            # candidates from Pinecone
BM25_TOP_K = 20               # candidates from BM25
RRF_K = 60                     # RRF damping constant (60 is the standard default)
RERANK_TOP_K = 5                # final chunks kept after Cohere reranks the merged set
COHERE_RERANK_MODEL = "rerank-english-v3.0"

# --- Generation ------------------------------------------------------------
GENERATION_MODEL = "claude-sonnet-4-5"
MAX_ANSWER_TOKENS = 1024

# --- Local BM25 persistence ------------------------------------------------
# BM25 needs the full corpus in memory to score against. We persist the raw
# chunk text/metadata locally (Pinecone stores vectors, not full-text search).
BM25_CORPUS_PATH = "./bm25_corpus.json"
