"""
Agentic RAG — LangGraph Controller
=====================================
Assembles all agent nodes into a LangGraph StateGraph.

Graph flow:
  Query Rewrite → Vector DB ──┐
                              ├─► RRF Fusion → Doc Grader → Web Search
                 BM25 Index ──┘
                              → Cross-Encoder → LLM Generate
                              → Hallucination Check → Reflect
                              → done? ──yes──► END
                                       ──no───► Reformulate → Query Rewrite
"""

import sys, warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from typing import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END, START

# ── Import all agents ──────────────────────────────────────────────────────────
from agents import (
    node_query_rewrite,
    node_retrieve_vector,
    node_retrieve_bm25,
    node_rrf_fusion,
    node_doc_grader,
    node_mcp_legal_search,
    node_cross_encoder,
    node_generate,
    node_hallucination_check,
)
from agents.shared import get_lf_config, get_lf_handler, flush_langfuse, parse_json, add_trace, log_trace
from agents.vector_db import vectorstore       # exposed for Gradio chunk count

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas.metrics.collections import Faithfulness, AnswerRelevancy
from ragas.llms import llm_factory as ragas_llm_factory
from ragas.embeddings import OpenAIEmbeddings as RagasOAIEmbeddings
import openai, asyncio

# ── LLM + RAGAS judge setup ───────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

_openai_async_client = openai.AsyncOpenAI()
_ragas_llm = ragas_llm_factory("gpt-4o-mini", client=_openai_async_client)
_ragas_emb = RagasOAIEmbeddings(model="text-embedding-3-small", client=_openai_async_client)

_faithfulness = Faithfulness(llm=_ragas_llm)
_relevancy    = AnswerRelevancy(llm=_ragas_llm, embeddings=_ragas_emb)


def _run_ragas(question: str, answer: str, contexts: list) -> tuple:
    """Run RAGAS metrics (faithfulness + answer_relevancy); return (scores, overall 0-10)."""
    ctxs = contexts if contexts else [""]

    async def _score():
        f = await _faithfulness.ascore(
            user_input=question, response=answer, retrieved_contexts=ctxs)
        r = await _relevancy.ascore(
            user_input=question, response=answer)
        return float(f.value), float(r.value)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    f, r = loop.run_until_complete(_score())

    scores = {
        "faithfulness":     round(f * 10, 1),
        "answer_relevancy": round(r * 10, 1),
    }
    overall = round((f + r) / 2 * 10, 1)
    return scores, overall


# ── Reformulate prompt (routing logic stays LLM-based) ───────────────────────

REFORMULATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Reformulate the question to improve document search. "
     "Return ONLY the new question, nothing else."),
    ("human",
     "Original question: {question}\n"
     "Problem with previous search: {reasoning}\n\nNew question:"),
])
reformulate_chain = REFORMULATE_PROMPT | llm | StrOutputParser()


# ══════════════════════════════════════
#  Graph State
# ══════════════════════════════════════

class RAGState(TypedDict):
    # ── Input ──
    question:         str
    max_attempts:     int
    language:         str   # "ru" | "kz" | "en"
    # ── Runtime ──
    current_query:    str
    expansion:        str
    attempt:          int
    # ── Retrieval ──
    vector_docs:      List[Document]
    bm25_docs:        List[Document]
    fused_docs:       List[Document]
    graded_docs:      List[Document]
    reranked_docs:    List[Document]
    # ── Generation ──
    web_context:      str
    mcp_context:      str
    answer:           str
    # ── Evaluation ──
    hallucination_ok: bool
    decision:         str   # "accept" | "retry_retrieval" | "retry_generation"
    reasoning:        str
    judge_scores:     dict  # {relevance, faithfulness, completeness, context_quality}
    overall_score:    float
    strategies_used:  List[str]
    # ── Trace ──
    trace:            List[dict]


# ══════════════════════════════════════
#  Controller nodes (reflect / reformulate)
# ══════════════════════════════════════

def node_reflect(state: RAGState) -> dict:
    """Score answer; decide accept / retry.
    On attempt 1 with hallucination_ok=True, accept immediately and run RAGAS in background.
    """
    question        = state["question"]
    answer          = state["answer"]
    docs            = state.get("reranked_docs") or state.get("graded_docs", [])
    attempt         = state.get("attempt", 0)
    max_attempts    = state.get("max_attempts", 3)
    hallucination_ok = state.get("hallucination_ok", True)
    strategies      = list(state.get("strategies_used", []))
    expansion       = state.get("expansion", "none")
    strategies.append(f"a{attempt}:{expansion}")

    contexts = [d.page_content for d in docs[:3]] if docs else []
    mcp = state.get("mcp_context", "")
    if mcp and "не найдено" not in mcp and mcp.strip():
        contexts.append(mcp[:1500])

    # Fast path: first attempt + grounded → accept immediately, RAGAS runs in ask()
    if attempt == 0 and hallucination_ok:
        return {
            "decision": "accept", "reasoning": "Grounded on first attempt — fast accept",
            "judge_scores": {"faithfulness": None, "answer_relevancy": None},
            "overall_score": 0.0,
            "attempt": attempt + 1, "strategies_used": strategies,
            "trace": add_trace(state, "REFLECT",
                               "Decision: accept (fast-path) — hallucination_ok"),
        }

    # Slow path: run RAGAS synchronously for retry decisions
    scores, overall = _run_ragas(question, answer, contexts)

    if attempt >= max_attempts - 1:
        return {
            "decision": "accept", "reasoning": "Max attempts reached",
            "judge_scores": scores, "overall_score": overall,
            "attempt": attempt + 1, "strategies_used": strategies,
            "trace": add_trace(state, "REFLECT",
                               f"FORCE ACCEPT (attempt {attempt+1}/{max_attempts}) score={overall}/10"),
        }

    if overall >= 7.0:
        decision  = "accept"
        reasoning = f"RAGAS overall {overall}/10 ≥ 7 — accepted"
    elif scores.get("faithfulness", 10) < 5:
        decision  = "retry_retrieval"
        reasoning = f"faithfulness={scores.get('faithfulness')}/10 — need better context"
    else:
        decision  = "retry_generation"
        reasoning = f"RAGAS overall {overall}/10 < 7 — regenerating"

    return {
        "decision": decision, "reasoning": reasoning,
        "judge_scores": scores, "overall_score": overall,
        "attempt": attempt + 1, "strategies_used": strategies,
        "trace": add_trace(state, "REFLECT",
                           f"Decision: {decision} (score={overall}/10) — {reasoning[:80]}"),
    }


def node_reformulate(state: RAGState) -> dict:
    """Rewrite query based on reflection feedback; rotate expansion strategy."""
    question  = state["question"]
    reasoning = state.get("reasoning", "")

    new_query = reformulate_chain.invoke(
        {"question": question, "reasoning": reasoning},
        config=get_lf_config(),
    ).strip()

    expansions     = ["none", "hyde", "step_back", "keyword"]
    current        = state.get("expansion", "none")
    idx            = expansions.index(current) if current in expansions else 0
    next_expansion = expansions[(idx + 1) % len(expansions)]

    return {
        "current_query": new_query, "expansion": next_expansion,
        "trace": add_trace(state, "REFORMULATE",
                           f"New query: '{new_query[:80]}' | next: {next_expansion}"),
    }


# ══════════════════════════════════════
#  Routing
# ══════════════════════════════════════

def route_after_reflect(state: RAGState) -> str:
    decision = state.get("decision", "accept")
    if decision == "accept":
        return "done"
    elif decision == "retry_retrieval":
        return "reformulate"
    else:                           # retry_generation
        return "generate"


LAW_CODES = {
    "Трудовой кодекс РК.pdf",
    "Налоговый кодекс РК.pdf",
    "Гражданский кодекс РК (общая часть).pdf",
    "Предпринимательский кодекс РК.pdf",
}

def route_after_doc_grader(state: RAGState) -> str:
    """Cross-Encoder if: ≥2 docs found, OR ≥1 doc from a user-uploaded (non-law) file."""
    graded_docs = state.get("graded_docs", [])
    if len(graded_docs) >= 2:
        return "cross_encoder"
    if len(graded_docs) >= 1:
        sources = {doc.metadata.get("source", "") for doc in graded_docs}
        if not sources.issubset(LAW_CODES):
            return "cross_encoder"
    return "mcp_legal_search"


# ══════════════════════════════════════
#  Build graph
# ══════════════════════════════════════

def build_graph():
    g = StateGraph(RAGState)

    # ── Register all nodes ──
    g.add_node("query_rewrite",       node_query_rewrite)
    g.add_node("retrieve_vector",     node_retrieve_vector)
    g.add_node("retrieve_bm25",       node_retrieve_bm25)
    g.add_node("rrf_fusion",          node_rrf_fusion)
    g.add_node("doc_grader",          node_doc_grader)
    g.add_node("mcp_legal_search",    node_mcp_legal_search)
    g.add_node("cross_encoder",       node_cross_encoder)
    g.add_node("generate",            node_generate)
    g.add_node("hallucination_check", node_hallucination_check)
    g.add_node("reflect",             node_reflect)
    g.add_node("reformulate",         node_reformulate)

    # ── Main pipeline ──
    g.add_edge(START,               "query_rewrite")
    g.add_edge("query_rewrite",     "retrieve_vector")
    g.add_edge("retrieve_vector",   "retrieve_bm25")
    g.add_edge("retrieve_bm25",     "rrf_fusion")
    g.add_edge("rrf_fusion",        "doc_grader")
    g.add_edge("cross_encoder",     "generate")
    g.add_edge("generate",          "hallucination_check")
    g.add_edge("hallucination_check", "reflect")
    g.add_edge("mcp_legal_search",  "cross_encoder")

    # ── Conditional routing ──
    # After Doc Grader: enough docs → Cross-Encoder, else → MCP (adilet.zan.kz)
    g.add_conditional_edges(
        "doc_grader",
        route_after_doc_grader,
        {"cross_encoder": "cross_encoder", "mcp_legal_search": "mcp_legal_search"},
    )
    g.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {"done": END, "reformulate": "reformulate", "generate": "generate"},
    )
    g.add_edge("reformulate", "query_rewrite")

    compiled = g.compile()
    return compiled


rag_graph = build_graph()


# ══════════════════════════════════════
#  Public API
# ══════════════════════════════════════

def run_graph(
    question:     str,
    max_attempts: int  = 3,
    expansion:    str  = "none",
    language:     str  = "ru",
) -> RAGState:
    initial: RAGState = {
        "question":        question,
        "max_attempts":    max_attempts,
        "language":        language,
        "current_query":   question,
        "expansion":       expansion,
        "attempt":         0,
        "vector_docs":     [],
        "bm25_docs":       [],
        "fused_docs":      [],
        "graded_docs":     [],
        "reranked_docs":   [],
        "web_context":     "",
        "mcp_context":     "",
        "answer":          "",
        "hallucination_ok": True,
        "decision":        "",
        "reasoning":       "",
        "judge_scores":    {},
        "overall_score":   0.0,
        "strategies_used": [],
        "trace":           [],
    }
    result = rag_graph.invoke(initial)

    # ── Send trace to LangFuse ──
    log_trace(
        name="agentic-rag",
        input_data={"question": question, "expansion": expansion},
        output_data={"answer": result.get("answer", ""), "decision": result.get("decision", "")},
        metadata={
            "attempts":        result.get("attempt", 0),
            "overall_score":   result.get("overall_score", 0.0),
            "judge_scores":    result.get("judge_scores", {}),
            "hallucination_ok": result.get("hallucination_ok", True),
            "strategies_used": result.get("strategies_used", []),
        },
    )

    return result
