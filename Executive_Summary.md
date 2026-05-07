# Executive Summary
## Agentic RAG — Legal Assistant for SMB Kazakhstan

**Author:** Dana Kassym  
**Program:** EPAM Generative AI for Software Development, 2025  
**Date:** May 19, 2026

---

## Problem Statement

Small and medium businesses (SMB) in Kazakhstan face a significant challenge accessing legal information. Kazakhstan legislation spans multiple complex codes — Labor, Tax, Civil, and Entrepreneurship — totalling thousands of pages. Legal consultations are expensive, and online search often returns outdated or irrelevant results.

The goal of this project was to build an **intelligent, reliable legal assistant** that can answer specific legal questions about Kazakhstan legislation in real time, in three languages (Russian, Kazakh, English), and clearly cite its sources.

---

## Solution

**Agentic RAG** (Retrieval-Augmented Generation) — a multi-agent pipeline built on LangGraph that combines local document search with live web retrieval, evaluates answer quality, and automatically retries when the quality is insufficient.

Unlike standard RAG (retrieve → generate), this system uses a **feedback loop**: if the generated answer scores below 7/10 on RAGAS metrics, the system reformulates the query and tries again — up to 3 attempts.

---

## Architecture

The pipeline consists of **7 sequential agents** orchestrated by a LangGraph StateGraph:

| # | Agent | Role |
|---|-------|------|
| 1 | Query Rewrite | Expands query using HyDE, Step-Back, or Keyword strategies |
| 2 | Hybrid Search | Combines ChromaDB semantic search + BM25 lexical search |
| 3 | RRF Fusion | Merges results using Reciprocal Rank Fusion (K=60) |
| 4 | Doc Grader | Filters irrelevant chunks in parallel (6 threads, ~6× speedup) |
| 5 | Cross-Encoder | LLM re-ranks documents by relevance score (0–10) |
| 6 | Generate + Hallucination Check | GPT-4.1-mini generates answer; verifies grounding |
| 7 | Reflect / Reformulate | RAGAS judge; triggers retry loop if quality < 7/10 |

**MCP Fallback:** when local documents don't contain sufficient information, the system queries `adilet.zan.kz` (official Kazakhstan legal database) live via a custom MCP server.

---

## Key Features

- **Hybrid search** — semantic (ChromaDB, ~17K chunks) + lexical (BM25) with RRF merging
- **Quality evaluation** — RAGAS Faithfulness + Answer Relevancy after every answer
- **Automatic retry** — up to 3 attempts with query reformulation
- **MCP integration** — live search on adilet.zan.kz, kgd.gov.kz, egov.kz
- **Observability** — full trace in LangFuse (every node, score, decision)
- **Multilingual** — Russian, Kazakh, English
- **Document upload** — users can add their own PDFs/DOCXs at runtime
- **Streaming UI** — live agent trace in Gradio while the answer generates

---

## Technology Stack

- **LangGraph** — agentic orchestration (StateGraph with conditional routing)
- **ChromaDB** — vector store (OpenAI text-embedding-3-small)
- **GPT-4.1-mini** — generation, grading, re-ranking
- **GPT-4o-mini** — RAGAS evaluation judge
- **RAGAS** — automated RAG quality metrics
- **LangFuse** — observability and tracing
- **Gradio** — web UI with streaming
- **BM25Okapi** — lexical search
- **MCP** — Model Context Protocol for live web search

---

## Results

The system was tested on representative SMB legal questions:

| Query type | Source used | Result |
|------------|-------------|--------|
| МЗП 2026 (minimum wage) | MCP → adilet.zan.kz | 85,000 KZT — correct |
| TechStart daily allowance | RAG (local DB) | 5,000 KZT — correct |
| Labor Code vacation days | RAG (local DB) | 24 calendar days — correct |
| Tax Code VAT rate | RAG (local DB) | 12% — correct |

RAGAS scores on answered queries typically range **7–9/10** for Faithfulness and Answer Relevancy, indicating well-grounded responses.

---

## Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Doc Grader was slow (sequential LLM calls) | Parallelized with ThreadPoolExecutor (6 threads, ~6× faster) |
| RAGAS blocked fast path on first attempt | Moved RAGAS to background after streaming; fast-path accepts on attempt 1 |
| MCP search returned noisy URLs | Regex-based URL extraction filtering /compare and /info paths |
| Single language UI | Full TRANSLATIONS dict for RU/KZ/EN with language state management |

---

## Conclusion

This project demonstrates a production-grade approach to domain-specific RAG, going beyond simple retrieve-and-generate to a multi-agent system with evaluation, retry logic, and real-time observability. The system is practical for real SMB use cases in Kazakhstan and can be extended with additional legal codes or regulatory documents at runtime.

---

*Project repository: GitHub (see README for setup instructions)*
