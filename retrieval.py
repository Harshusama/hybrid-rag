"""
Retrieval (ONLINE — runs on every user query)
------------------------------------------------
Query
  -> Pinecone vector search (top VECTOR_TOP_K)
  -> BM25 keyword search   (top BM25_TOP_K)
  -> merge both ranked lists with Reciprocal Rank Fusion (RRF)
  -> rerank the merged set with Cohere rerank
  -> return top RERANK_TOP_K chunks

Each stage is its own method so you can test/inspect it independently —
useful when you're debugging why a particular query retrieved poorly.
"""

import json
import os

import cohere
from rank_bm25 import BM25Okapi
from langchain_pinecone import PineconeVectorStore

from ingest import get_embeddings, get_pinecone_index
import config


class HybridRetriever:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.vectorstore = PineconeVectorStore(
            index=get_pinecone_index(),
            embedding=self.embeddings,
        )
        self.cohere_client = cohere.Client(config.COHERE_API_KEY)
        self._load_bm25()

    def _load_bm25(self):
        """
        BM25 needs the whole corpus tokenized in memory. We rebuild it
        from the local JSON file written during ingestion. For a large or
        frequently-changing corpus, you'd persist the tokenized index
        itself rather than rebuilding on every startup — kept simple here.
        """
        if not os.path.exists(config.BM25_CORPUS_PATH):
            self._bm25 = None
            self._bm25_docs = {}
            return

        with open(config.BM25_CORPUS_PATH, "r") as f:
            self._bm25_docs = json.load(f)

        self._bm25_ids = list(self._bm25_docs.keys())
        tokenized = [self._bm25_docs[cid]["text"].lower().split() for cid in self._bm25_ids]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    # ---- Stage 1: vector search ------------------------------------------
    def vector_search(self, query: str, top_k: int) -> list[dict]:
        results = self.vectorstore.similarity_search(query, k=top_k)
        out = []
        for doc in results:
            key = f"{doc.metadata.get('source')}::{doc.metadata.get('chunk_index')}"
            out.append({"key": key, "text": doc.page_content, "metadata": doc.metadata})
        return out

    # ---- Stage 2: keyword search --------------------------------------------
    def bm25_search(self, query: str, top_k: int) -> list[dict]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(zip(self._bm25_ids, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {
                "key": cid,
                "text": self._bm25_docs[cid]["text"],
                "metadata": self._bm25_docs[cid]["metadata"],
            }
            for cid, score in ranked
            if score > 0
        ]

    # ---- Stage 3: Reciprocal Rank Fusion -------------------------------------
    def reciprocal_rank_fusion(self, ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
        """
        Merges multiple ranked lists into one, using only rank position
        (not raw scores) from each list:

            RRF_score(doc) = sum over lists of  1 / (k + rank_in_that_list)

        A doc that appears near the top of EITHER list scores highly. A
        doc appearing in BOTH lists (even at moderate rank in each) often
        outranks a doc that's #1 in only one list — which is usually what
        you want: agreement across two different retrieval methods is a
        stronger relevance signal than one method being very confident.

        k=60 is the constant from the original RRF paper — it dampens how
        much rank #1 vs #2 vs #3 differ, so the fusion isn't dominated by
        whichever single list has the most extreme top score.
        """
        scores: dict[str, float] = {}
        doc_lookup: dict[str, dict] = {}

        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list, start=1):
                key = doc["key"]
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
                doc_lookup[key] = doc

        merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{**doc_lookup[key], "rrf_score": score} for key, score in merged]

    # ---- Stage 4: rerank with Cohere -------------------------------------
    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """
        RRF gets us a good SHORTLIST cheaply, but it still doesn't
        understand the query's meaning — it's just combining two separate
        rankings. Cohere's rerank model reads the query and each candidate
        chunk TOGETHER and scores relevance directly, which is far more
        accurate. We only run it on the merged shortlist (not the full
        corpus) because it's too slow/expensive to run over everything.
        """
        if not candidates:
            return []

        response = self.cohere_client.rerank(
            model=config.COHERE_RERANK_MODEL,
            query=query,
            documents=[c["text"] for c in candidates],
            top_n=top_k,
        )

        reranked = []
        for result in response.results:
            original = candidates[result.index]
            reranked.append({**original, "rerank_score": result.relevance_score})
        return reranked

    # ---- Full pipeline ----------------------------------------------------
    def retrieve(self, query: str) -> list[dict]:
        vector_results = self.vector_search(query, top_k=config.VECTOR_TOP_K)
        bm25_results = self.bm25_search(query, top_k=config.BM25_TOP_K)

        merged = self.reciprocal_rank_fusion([vector_results, bm25_results], k=config.RRF_K)

        return self.rerank(query, merged, top_k=config.RERANK_TOP_K)