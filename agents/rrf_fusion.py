"""
Agent: RRF Fusion
==================
Reciprocal Rank Fusion — combines Vector DB and BM25 rankings.
Formula: score(doc) = Σ 1 / (K + rank_i)   where K = 60
"""

from langchain_core.documents import Document
from .shared import add_trace

K = 60  # standard RRF constant


def node_rrf_fusion(state: dict) -> dict:
    vector_docs: list[Document] = state.get("vector_docs", [])
    bm25_docs:   list[Document] = state.get("bm25_docs",   [])

    rrf_scores: dict[str, float]    = {}
    doc_map:    dict[str, Document] = {}

    for rank, doc in enumerate(vector_docs, 1):
        key = doc.page_content[:120]
        rrf_scores[key]  = rrf_scores.get(key, 0.0) + 1.0 / (K + rank)
        doc_map[key]     = doc

    for rank, doc in enumerate(bm25_docs, 1):
        key = doc.page_content[:120]
        rrf_scores[key]  = rrf_scores.get(key, 0.0) + 1.0 / (K + rank)
        if key not in doc_map:
            doc_map[key] = doc

    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

    fused = []
    for key in sorted_keys[:8]:
        doc = doc_map[key]
        doc.metadata["rrf_score"] = round(rrf_scores[key], 6)
        fused.append(doc)

    top = fused[0].metadata.get("rrf_score") if fused else 0
    return {
        "fused_docs": fused,
        "trace": add_trace(
            state, "RRF_FUSION",
            f"Merged {len(vector_docs)}(vec) + {len(bm25_docs)}(bm25) → {len(fused)} docs",
            {"top_rrf_score": top},
        ),
    }
