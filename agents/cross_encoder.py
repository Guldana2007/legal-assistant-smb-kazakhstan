"""
Agent: Cross-Encoder
=====================
LLM-based re-ranking: scores each document against the question (0–10)
and sorts documents by relevance score.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .shared import get_lf_config, parse_json, add_trace

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Rate the relevance of the fragment for answering the question. Score 0–10.\n"
     "Reply STRICTLY as JSON: {{\"score\": N, \"reason\": \"briefly\"}}"),
    ("human", "Question: {question}\n\nFragment:\n{doc}"),
])
chain = PROMPT | llm | StrOutputParser()


def node_cross_encoder(state: dict) -> dict:
    question = state["question"]
    docs     = state.get("graded_docs", [])
    scored   = []

    for doc in docs:
        raw    = chain.invoke(
            {"question": question, "doc": doc.page_content[:500]},
            config=get_lf_config(),
        )
        result = parse_json(raw, {"score": 5})
        score  = float(result.get("score", 5))
        doc.metadata["cross_encoder_score"] = score
        scored.append((doc, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    reranked = [d for d, _ in scored]

    top = scored[0][1] if scored else 0
    return {
        "reranked_docs": reranked,
        "trace": add_trace(state, "CROSS_ENCODER",
                           f"Re-ranked {len(reranked)} docs | top_score={top:.1f}"),
    }
