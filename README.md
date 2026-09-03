Hybrid RAG Pipeline

A Retrieval-Augmented Generation system that combines dense semantic search with sparse keyword search, merges and deduplicates the retrieved candidates, and uses reranking to select the strongest evidence before generating an answer.

Project Overview

Traditional vector search is effective at finding passages with similar meaning, but it can miss exact names, identifiers, codes, or technical terms. Keyword search handles exact matches well, but it may miss paraphrases and related concepts.

This project combines both approaches:

Dense retrieval finds semantically similar passages using embeddings.

Sparse retrieval finds passages containing exact keywords or terms.

Candidate fusion combines results from both retrieval methods.

Deduplication removes repeated passages.

Reranking reorders candidates according to their relevance to the complete question.

Grounded generation asks the LLM to answer using the selected evidence.

Architecture

flowchart TD
    A[Source Documents] --> B[Clean and Chunk]
    B --> C[Dense Vector Index]
    B --> D[Sparse Keyword Index]
    E[User Question] --> F[Hybrid Retrieval]
    C --> F
    D --> F
    F --> G[Merge and Deduplicate]
    G --> H[Rerank Candidates]
    H --> I[Build Grounded Prompt]
    I --> J[LLM Answer]

Workflow

Load and clean the source documents.

Divide the documents into meaningful, overlapping chunks.

Generate embeddings and store them in a vector index.

Create a sparse or keyword-search representation of the same chunks.

Send the user question to both retrieval systems.

Merge and deduplicate the candidate chunks.

Rerank the candidates against the original question.

Pass only the strongest evidence to the LLM.

Generate a grounded answer and, when available, return source citations.

Why Hybrid Retrieval?

Retrieval method

Strength

Limitation

Dense/vector search

Understands meaning, context, and paraphrases

May miss exact identifiers or rare terms

Sparse/keyword search

Preserves exact words, codes, and names

May miss semantic relationships

Hybrid search

Combines semantic and exact-match signals

Requires fusion, deduplication, and tuning

Reranking

Improves final candidate ordering

Adds latency and computational cost

Key Features

Document ingestion and preprocessing

Configurable chunking and overlap

Embedding-based semantic retrieval

Keyword or sparse retrieval

Candidate merging and deduplication

Relevance-based reranking

Context-controlled prompt construction

Grounded LLM response generation

Environment-based secret management

Modular components for testing and improvement

Technology Stack

Python

LLM and embedding APIs

Vector similarity search

Keyword or BM25-style retrieval

Reranking model or API

Git and GitHub

Update this list with the exact database, model provider, and framework used in your implementation.

Installation

1. Clone the repository

git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY

2. Create a virtual environment

Windows PowerShell:

python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

macOS/Linux:

python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Configure environment variables

Create a .env file based on .env.example:

OPENAI_API_KEY=your_api_key

Add any vector database or reranking credentials required by your implementation.

Never commit .env or real API keys to GitHub.

5. Run the project

Run the entrypoint used by your implementation, for example:

python app.py

Replace app.py if your actual entrypoint has a different filename.

Main Engineering Challenges

Chunk-size tradeoff

Chunks that are too small may lose important context. Chunks that are too large can add noise, reduce retrieval precision, and consume more input tokens.

Retrieval gaps

Dense retrieval may miss exact identifiers, while keyword retrieval may miss paraphrased meaning. Hybrid retrieval reduces these individual weaknesses.

Duplicate candidates

The same passage can be returned by both retrievers. Candidates should be deduplicated before reranking and prompt construction.

Candidate ordering

Initial retrieval scores from different systems are not always directly comparable. Fusion and reranking are used to produce a stronger final ordering.

Grounding and hallucinations

The generator should receive only relevant evidence and should be instructed to avoid unsupported claims. When evidence is insufficient, the system should say that it does not have enough information.

Latency and cost

Hybrid retrieval and reranking improve quality but add processing time and expense. The number of retrieved and reranked candidates should be controlled.

Evaluation

The retrieval and generation stages should be evaluated separately.

Retrieval metrics

Recall@K

Precision@K

Mean Reciprocal Rank (MRR)

Normalized Discounted Cumulative Gain (NDCG)

Generation metrics

Answer relevance

Faithfulness to retrieved evidence

Citation accuracy

Unsupported-claim rate

Latency

Token usage and cost

Security and Reliability

Store credentials in environment variables or a managed secret store.

Exclude .env from Git.

Validate uploaded files and user inputs.

Treat retrieved content as untrusted data.

Add prompt-injection defenses and access controls.

Apply timeouts and bounded retries to external API calls.

Log retrieval decisions without exposing confidential content.

Set API spending and rate limits.

What I Learned

RAG adds external knowledge at request time; it does not retrain the LLM.

Retrieved chunks are evidence passages, not final answers.

A fluent model cannot compensate for missing or irrelevant evidence.

Hybrid search improves recall by combining complementary retrieval signals.

Reranking can improve precision but introduces latency and cost.

Retrieval quality, grounding, evaluation, security, and observability are essential parts of a reliable RAG system.

Interview Explanation

I built a Hybrid RAG pipeline that combines dense vector retrieval with sparse keyword retrieval. The vector retriever captures semantic meaning, while keyword retrieval preserves exact terms and identifiers. I merge and deduplicate both result sets, then rerank the candidates against the original question before passing the strongest chunks to the LLM. I evaluate retrieval separately from generation because a well-written answer can still be unsupported if the correct evidence was not retrieved.

Future Improvements

Create a labeled evaluation dataset.

Compare dense-only, sparse-only, and hybrid retrieval.

Tune chunk size, overlap, top-K retrieval, and reranking depth.

Add citations and evidence-based refusal behavior.

Add automated regression tests.

Add structured logging, tracing, caching, and monitoring.

Deploy using managed vector storage and secret management.

Project Status

This is a portfolio and learning implementation demonstrating Hybrid RAG architecture, retrieval design, and reranking concepts. Production readiness requires independent evaluation, security review, load testing, monitoring, and deployment verification.
