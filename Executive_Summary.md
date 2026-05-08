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

### How the pipeline flows

**Step 1 — Query Rewrite:** The user's question is received. If the language is English or Kazakh, it is translated to Russian (the language of the knowledge base). The query is then expanded using one of 4 strategies: HyDE, Step-Back, Keyword, or none.

**Steps 2a/2b — Hybrid Search:** Vector search (ChromaDB) retrieves 15 documents by semantic meaning; BM25 retrieves 15 documents by keyword matching. Both run in parallel.

**Step 2c — RRF Fusion:** Both result lists are merged using Reciprocal Rank Fusion → top 8 documents.

**Step 3 — Doc Grader:** Each of the 8 documents is evaluated by an LLM for relevance. Runs in parallel (ThreadPoolExecutor).
> **Condition:** if fewer than 2 relevant documents are found → **MCP fallback**: the system queries live government portals (`adilet.zan.kz`, `kgd.gov.kz`, `egov.kz`), then continues to step 4.

**Step 4 — Cross-Encoder:** LLM re-ranks all documents with a score 0–10. Always runs — both after RAG and after MCP.

**Step 5 — LLM Generate:** GPT-4.1-mini generates the answer with source citations.

**Step 6 — Hallucination Check:** LLM verifies that every fact in the answer is supported by the retrieved documents. If not grounded, sets a flag that Reflect (step 7) uses to trigger retry.

**Step 7 — Reflect (RAGAS):** gpt-4o-mini evaluates answer quality: Faithfulness + Answer Relevancy.
> **Condition:** if overall score ≥ 7/10 → answer accepted, done. If score < 7/10 → go to step 8.

**Step 8 — Reformulate:** The query is rewritten with a different expansion strategy and the pipeline loops back to step 1.
> **Retry loop:** maximum 3 attempts. After the 3rd attempt, the best answer obtained is returned.

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

### Why RAGAS?
RAGAS provides automated, reference-free evaluation specifically designed for RAG systems. **Faithfulness** checks whether every fact in the answer is grounded in the retrieved documents. **Answer Relevancy** checks whether the answer actually addresses the question. Together they cover the two main failure modes of RAG — hallucination and irrelevance. Running RAGAS after every response closes the quality loop automatically and drives the retry decision: if the score is below 7/10, the system reformulates the query and tries again. Cost: ~$0.001 per evaluation (gpt-4o-mini as judge) — up to 100× cheaper than using GPT-4o as evaluator, and far more scalable than human annotation.

### Why LangFuse?
LangFuse provides full observability over the pipeline at zero infrastructure cost — it is **free and open-source**, deployable locally via Docker. Every LLM call, token count, latency, RAGAS score, and user feedback (👍/👎) is logged as a structured trace. This makes it possible to debug retrieval failures, monitor costs in real time, and track quality trends across sessions — without sending any data to a third-party cloud.

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

The system was tested on English SMB legal questions (live session, 2026-05-07):

| Query | Score | Result |
|-------|-------|--------|
| Statute of limitations (Civil Code) | 9.9/10 | 3 years — correct |
| Employer liability for delayed salary | 9.4/10 | Penalty formula — correct |
| Vacation days (Labor Code) | 9.2/10 | 24 calendar days — correct |
| Labor Code 2025 amendments | 9.1/10 | Key changes — correct |
| Tax Code 2025 amendments | 7.6/10 | Correct — correct |
| VAT definition | 5.3/10 | Retrieved outdated chunk |
| Minimum wage 2026 | 5.0/10 | No specific figure retrieved |

**Average RAGAS score: 7.9/10 · Pass rate (≥7.0): 5/7.**  
Strong results on well-defined legal questions; lower scores on edge cases where retrieval returned an outdated or indirect source.

---

## Conclusion

This project demonstrates a production-grade approach to domain-specific RAG, going beyond simple retrieve-and-generate to a multi-agent system with self-correction, automated evaluation, live government data integration, and real-time observability. The system is practical for real SMB use cases in Kazakhstan and can be extended with additional legal codes or regulatory documents at runtime.
