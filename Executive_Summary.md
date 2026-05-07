# Executive Summary
## Agentic RAG — Legal Assistant for SMB Kazakhstan

**Author:** Guldana Kassym-Ashim  
**Program:** EPAM Generative AI for Software Development, 2026  
**GitHub:** https://github.com/Guldana2007/legal-assistant-smb-kazakhstan

---

## Problem Statement

Small and medium businesses (SMB) in Kazakhstan face a significant challenge accessing legal information. Kazakhstan legislation spans multiple complex codes — Labor, Tax, Civil, and Entrepreneurship — totalling thousands of pages. A single legal consultation costs $20–$100 per hour, and laws change constantly. SMBs simply cannot afford a full-time legal team.

The goal of this project was to build an **intelligent, reliable legal assistant** that answers specific legal questions about Kazakhstan legislation in real time, in three languages (Kazakh, English, Russian), and clearly cites its sources — at a fraction of the cost.

---

## Solution

**Agentic RAG** (Retrieval-Augmented Generation) — a multi-agent pipeline built on LangGraph that combines local document search with live government web retrieval, evaluates answer quality, and automatically retries when quality is insufficient.

Unlike standard RAG (retrieve → generate), this system uses a **self-correction loop**: if the generated answer scores below 7/10 on RAGAS metrics, the system reformulates the query with a different strategy and tries again — up to 3 attempts automatically.

**Cost:** ~$0.01 per query — up to **10,000× cheaper** than a traditional lawyer consultation.

---

## Architecture

The pipeline consists of **8 steps** orchestrated by a LangGraph StateGraph:

| # | Step | Role |
|---|------|------|
| 1 | Query Rewrite | Translates EN/KZ→RU; expands query (HyDE / Step-Back / Keyword / none) |
| 2a/2b | Hybrid Search | ChromaDB semantic (15 docs) + BM25 lexical (15 docs) run in parallel |
| 2c | RRF Fusion | Reciprocal Rank Fusion merges both lists → top 8 documents |
| 3 | Doc Grader | LLM filters irrelevant chunks in parallel |
| 4 | Cross-Encoder | LLM re-ranks all documents by relevance (score 0–10, gpt-4.1-mini) |
| 5 | LLM Generate | GPT-4.1-mini generates answer with source citations |
| 6 | Hallucination Check | Verifies every fact is grounded in retrieved context |
| 7 | Reflect | RAGAS judge (gpt-4o-mini); accepts answer or triggers retry |
| 8 | Reformulate | Rewrites query with new expansion strategy for retry |

**MCP Fallback:** when Doc Grader finds fewer than 2 relevant chunks, the system queries official Kazakhstan government portals live via a custom MCP server (`adilet.zan.kz`, `kgd.gov.kz`, `egov.kz`). After MCP, all documents always pass through Cross-Encoder before generation.

---

## Key Features

- **Hybrid search** — semantic (ChromaDB, ~17K chunks) + lexical (BM25) with RRF merging
- **Multilingual** — Kazakh, English, Russian (auto-translation for retrieval)
- **MCP integration** — live search on 3 official Kazakhstan government portals
- **Quality evaluation** — RAGAS Faithfulness + Answer Relevancy after every answer (~$0.001/eval)
- **Automatic retry** — up to 3 attempts with query reformulation
- **Observability** — full pipeline trace in LangFuse (every LLM call, token cost, RAGAS score)
- **Document upload** — users can add their own PDFs/DOCXs at runtime
- **Streaming UI** — live agent trace in Gradio while the answer generates
- **Test suite** — 24 tests (10 unit + 14 integration) covering all agents

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph (StateGraph, conditional routing) |
| Vector DB | ChromaDB + OpenAI text-embedding-3-small |
| LLM (agents) | GPT-4.1-mini (generation, grading, re-ranking) |
| LLM (evaluation) | GPT-4o-mini (RAGAS judge) |
| Lexical search | BM25Okapi |
| Evaluation | RAGAS (Faithfulness + AnswerRelevancy) |
| Observability | LangFuse |
| UI | Gradio (streaming, multilingual) |
| Live data | Custom FastMCP server |

---

## Cost Analysis & ROI

| Option | Cost |
|--------|------|
| Lawyer (1 hour, Kazakhstan) | $20 – $100 |
| PRG / Paragraf legal DB subscription | $17 – $45/month |
| **Our AI Assistant (1 query)** | **~$0.01** |
| 1,000 queries | ~$8 |

**ROI: up to 10,000× cheaper than a lawyer. Unlimited legal access for the price of a coffee.**

---

## Results

The system was evaluated with two sets of 8 questions each (Russian and English):

**Russian queries** (2026-05-02):

| Query | Source | Score | Result |
|-------|--------|-------|--------|
| МЗП 2026 (minimum wage) | MCP → adilet.zan.kz | 10.0/10 | 85,000 KZT — correct |
| Labor Code vacation days | RAG (local DB) | 7.5/10 | 24 calendar days — correct |
| Civil Code statute of limitations | RAG (local DB) | 7.8/10 | 3 years — correct |
| ИП registration | RAG + MCP | 7.4/10 | Correct procedure — correct |

**Average: 6.2/10 · Pass rate (≥7.0): 4/8**

**English queries** (2026-05-07):

| Query | Source | Score | Result |
|-------|--------|-------|--------|
| LLP registration documents | Local + MCP | 8.3/10 | Correct documents — correct |
| Vacation days (Labor Code) | RAG (local DB) | 7.7/10 | 24 calendar days — correct |
| Statute of limitations | MCP | 7.4/10 | 3 years — correct |

**Average: 5.7/10 · Pass rate (≥7.0): 3/8**

Strong results on well-defined legal questions in both languages; lower scores on out-of-scope queries (expected behaviour for a domain-specific assistant). English queries show slightly lower scores due to EN→RU translation for retrieval occasionally retrieving a less precise chunk.

---

## Conclusion

This project demonstrates a production-grade approach to domain-specific RAG, going beyond simple retrieve-and-generate to a multi-agent system with self-correction, automated evaluation, live government data integration, and real-time observability. The system is practical for real SMB use cases in Kazakhstan and can be extended with additional legal codes or regulatory documents at runtime.
