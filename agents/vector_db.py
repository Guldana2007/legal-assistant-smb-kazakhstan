"""
Agent: Vector DB
=================
Semantic search using ChromaDB + OpenAI embeddings.
"""

from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from .shared import add_trace

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db_legal"
COLLECTION  = "bayan_sulu_rag"

embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory=str(CHROMA_DIR),
    collection_name=COLLECTION,
    embedding_function=embeddings,
)


def node_retrieve_vector(state: dict) -> dict:
    query   = state["current_query"]
    results = vectorstore.similarity_search_with_relevance_scores(query, k=15)

    docs = []
    for rank, (doc, score) in enumerate(results, 1):
        doc.metadata["vector_score"] = round(float(score), 4)
        doc.metadata["vector_rank"]  = rank
        docs.append(doc)

    top = docs[0].metadata["vector_score"] if docs else 0
    return {
        "vector_docs": docs,
        "trace": add_trace(state, "VECTOR_DB",
                           f"Found {len(docs)} docs | top_score={top}"),
    }
