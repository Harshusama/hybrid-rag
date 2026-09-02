"""
Chunking
--------
Loads a raw document and splits it into LangChain Document objects
(chunks of text + metadata), using RecursiveCharacterTextSplitter.

WHY RecursiveCharacterTextSplitter: it tries to split on the "most natural"
boundary first — paragraphs ("\\n\\n"), then sentences/lines ("\\n"), then
words, then characters — only falling back to a harder split when a chunk
is still too big. This is the LangChain-ecosystem equivalent of the
paragraph-aware chunker we'd otherwise write by hand.

chunk_size / chunk_overlap are measured in CHARACTERS here (LangChain's
default unit for this splitter). ~4 characters ≈ 1 token for English, so
CHUNK_SIZE=500 chars ≈ 125 tokens — tune this in config.py per your model
and prompt budget.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config


def load_and_chunk(text: str, source: str, extra_metadata: dict | None = None) -> list[Document]:
    """
    Splits raw text into chunked LangChain Documents, each carrying
    metadata (source filename, chunk index) needed later for citations
    and for the local BM25 corpus.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # try paragraph, then line, then sentence, then word, then char
    )

    raw_chunks = splitter.split_text(text)

    documents = []
    for i, chunk_text in enumerate(raw_chunks):
        metadata = {"source": source, "chunk_index": i, **(extra_metadata or {})}
        documents.append(Document(page_content=chunk_text, metadata=metadata))

    return documents