"""
Agent: BM25 Index
==================
Lexical search using BM25Okapi.
Index is built lazily from the ChromaDB collection.
"""

import re
from typing import Optional, List
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .shared import add_trace

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db_legal"
COLLECTION  = "bayan_sulu_rag"

embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory=str(CHROMA_DIR),
    collection_name=COLLECTION,
    embedding_function=embeddings,
)

_bm25_cache: Optional[BM25Okapi]    = None
_docs_cache: Optional[List[Document]] = None


def _tokenize(text: str) -> list:
    return re.findall(r"\w+", text.lower())


def get_bm25():
    global _bm25_cache, _docs_cache
    if _bm25_cache is not None:
        return _bm25_cache, _docs_cache
    col    = vectorstore._collection
    result = col.get(include=["documents", "metadatas"])
    _docs_cache = [
        Document(page_content=doc, metadata=meta or {})
        for doc, meta in zip(result["documents"], result["metadatas"])
    ]
    if not _docs_cache:
        print("  [BM25] index empty — no documents in ChromaDB yet")
        return None, []
    _bm25_cache = BM25Okapi([_tokenize(d.page_content) for d in _docs_cache])
    print(f"  [BM25] index built: {len(_docs_cache)} docs")
    return _bm25_cache, _docs_cache


def node_retrieve_bm25(state: dict) -> dict:
    query       = state["current_query"]
    bm25, all_docs = get_bm25()

    if not all_docs:
        return {"bm25_docs": [],
                "trace": add_trace(state, "BM25_INDEX", "Empty index")}

    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:15]

    docs = []
    for rank, (idx, score) in enumerate(ranked, 1):
        d = Document(
            page_content=all_docs[idx].page_content,
            metadata={**all_docs[idx].metadata,
                      "bm25_score": round(float(score), 4),
                      "bm25_rank":  rank},
        )
        docs.append(d)

    top = ranked[0][1] if ranked else 0
    return {
        "bm25_docs": docs,
        "trace": add_trace(state, "BM25_INDEX",
                           f"Found {len(docs)} docs | top_score={top:.4f}"),
    }
