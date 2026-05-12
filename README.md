# Agentic RAG — Legal Assistant for SMB Kazakhstan

A conversational legal assistant for small and medium businesses in Kazakhstan, built on an agentic RAG pipeline with LangGraph, hybrid search, MCP integration, and automated quality evaluation.

**Author:** Guldana Kassym-Ashim  
**Program:** EPAM Generative AI for Software Development, 2026  
**GitHub:** https://github.com/Guldana2007/legal-assistant-smb-kazakhstan

---

## Overview

The system answers legal questions about Kazakhstan legislation by combining local document search (RAG) with live government web search via MCP (adilet.zan.kz, kgd.gov.kz, egov.kz). Supports Kazakh, English, and Russian. Every answer is grounded, hallucination-checked, and evaluated with RAGAS.

**Pipeline:**
```
1. Query Rewrite → 2a. Vector DB → 2b. BM25 → 2c. RRF Fusion → 3. Doc Grader
  → 4. Cross-Encoder → 5. LLM Generate → 6. Hallucination Check → 7. Reflect
  └─ fallback:          Doc Grader (<2 docs) → MCP → 4. Cross-Encoder
  └─ retry_retrieval:   7. Reflect → 8. Reformulate → 1. Query Rewrite (up to 3 attempts)
  └─ retry_generation:  7. Reflect → 5. LLM Generate (skips retrieval, faithfulness ≥ 5)
```

## Architecture

![Architecture Diagram](architecture_diagram_v4.png)

**LangGraph Graph Walk — 3 specialized agents, 8 pipeline steps:**

| # | Agent | Steps | Role |
|---|-------|-------|------|
| 1 | **Retrieval Agent** | 1–4 + MCP fallback | Query rewrite, hybrid search (Vector DB + BM25), RRF fusion, doc grading, cross-encoder re-ranking. Falls back to MCP live search when fewer than 2 relevant docs are found. |
| 2 | **Generation Agent** | 5–6 | Generates the answer (GPT-4.1-mini) with source citations; verifies every fact is grounded in retrieved context. |
| 3 | **Orchestrating Agent** | 7–8 | RAGAS judge — autonomously decides: accept answer / retry retrieval / retry generation / stop. The only node that makes independent decisions. |

**Pipeline steps:**

| Step | Name | Description |
|------|------|-------------|
| 1 | Query Rewrite | Translates EN/KZ→RU; expands query (HyDE / Step-Back / Keyword / none) |
| 2a/2b | Hybrid Search | ChromaDB semantic (15 docs) + BM25 lexical (15 docs) |
| 2c | RRF Fusion | Reciprocal Rank Fusion (K=60) → top 8 docs |
| 3 | Doc Grader | Filters irrelevant chunks in parallel (LLM per doc) |
| 4 | Cross-Encoder | LLM re-ranking, scores 0–10 |
| 5 | LLM Generate | GPT-4.1-mini generates answer with source citations |
| 6 | Hallucination Check | Verifies every fact is grounded in retrieved context |
| 7 | Reflect ★ | Fast-accept on attempt 0 if grounded; else runs RAGAS and decides: accept (≥7/10) / retry_retrieval (faithfulness <5) / retry_generation (score <7, faithfulness ≥5) |
| 8 | Reformulate | Rewrites query with next expansion strategy; used only for retry_retrieval |

**MCP fallback:** when Doc Grader finds < 2 relevant chunks, queries government portals live via custom MCP server. After MCP, always passes through Cross-Encoder.

**Two retry paths:**
- `retry_retrieval` — full restart: Reformulate → Query Rewrite → new retrieval (triggered when faithfulness < 5)
- `retry_generation` — regeneration only: goes directly back to LLM Generate without new retrieval (triggered when score < 7 but faithfulness ≥ 5)

**RAGAS timing:** on the first attempt Reflect uses a fast path — accepts immediately if the answer is grounded, without running RAGAS. RAGAS scores appear ~5 seconds after the answer, computed asynchronously.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph (StateGraph, TypedDict state) |
| Vector DB | ChromaDB (~17K chunks, 4 legal codes) |
| Embeddings | OpenAI text-embedding-3-small |
| LLM (agents) | GPT-4.1-mini |
| LLM (evaluation) | GPT-4o-mini (RAGAS judge, temperature=0) |
| Lexical search | BM25Okapi (rank-bm25) |
| Evaluation | RAGAS (Faithfulness + AnswerRelevancy, ~$0.001/eval) |
| Observability | LangFuse (full pipeline trace, token costs) |
| UI | Gradio 6.x (streaming, multilingual) |
| MCP server | Custom FastMCP (adilet.zan.kz, kgd.gov.kz, egov.kz) |

**Cost:** ~$0.01 per query — up to 10,000× cheaper than a lawyer ($20–$100/hr).

---

## Knowledge Base

Documents indexed in ChromaDB:
- Трудовой кодекс РК (Labor Code)
- Налоговый кодекс РК (Tax Code)
- Гражданский кодекс РК (Civil Code)
- Предпринимательский кодекс РК (Entrepreneurship Code)

Users can upload additional PDF/DOCX files at runtime via the **Upload** tab — they are chunked and indexed into ChromaDB automatically.

---

## Requirements

- Python 3.11+
- OpenAI API key
- Docker (optional — for LangFuse observability)

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Guldana2007/legal-assistant-smb-kazakhstan.git
cd legal-assistant-smb-kazakhstan

# 2. Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...

# LangFuse (optional — for observability)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

---

## Running

### 1. Start LangFuse (optional)
```bash
docker-compose -f docker-compose.langfuse.yml up -d
# Dashboard: http://localhost:3000
```

### 2. Index documents
```bash
python ingest.py
```

### 3. Launch the app
```bash
python app_agentic_rag.py
# Open: http://localhost:7861
```

---

## Usage

1. Select language: **Қазақша / English / Русский**
2. Type a legal question or click a sample question
3. Adjust **Max Attempts** (1–5) and **Query Expansion** strategy if needed
4. Click **Submit** — the answer streams with live Agent Trace
5. RAGAS scores (Faithfulness + Answer Relevancy) appear ~5 sec after the answer

### Upload custom documents

Go to the **Upload** tab to add your own PDF or DOCX files. They will be chunked and indexed automatically into ChromaDB.

---

## Tests

```bash
# Fast unit tests — no LLM calls (~3 seconds)
python -m pytest tests/ -v -m unit

# Full integration tests — real LLM calls (~3-5 minutes)
python -m pytest tests/ -v -m integration

# All 24 tests
python -m pytest tests/ -v
```

**24/24 tests pass:** 10 unit + 14 integration ✅ (verified 2026-05-11)

```
platform win32 -- Python 3.11.0, pytest-9.0.3

UNIT TESTS (10/10 passed, 31s)
tests/test_pipeline.py::TestQueryRewrite::test_no_expansion_returns_original    PASSED
tests/test_pipeline.py::TestQueryRewrite::test_returns_non_empty_query          PASSED
tests/test_pipeline.py::TestRRFFusion::test_fuses_both_sources                  PASSED
tests/test_pipeline.py::TestRRFFusion::test_rrf_scores_assigned                 PASSED
tests/test_pipeline.py::TestRRFFusion::test_empty_inputs_returns_empty          PASSED
tests/test_pipeline.py::TestDocGrader::test_relevant_document_kept              PASSED
tests/test_pipeline.py::TestDocGrader::test_irrelevant_document_filtered        PASSED
tests/test_pipeline.py::TestDocGrader::test_empty_docs_returns_empty            PASSED
tests/test_pipeline.py::TestHallucinationCheck::test_grounded_answer_passes     PASSED
tests/test_pipeline.py::TestHallucinationCheck::test_hallucinated_answer_flagged PASSED

INTEGRATION TESTS (14/14 passed, ~4 min)
tests/test_pipeline.py::TestPositiveScenarios::test_labor_vacation_days         PASSED
tests/test_pipeline.py::TestPositiveScenarios::test_vat_rate                    PASSED
tests/test_pipeline.py::TestPositiveScenarios::test_civil_code_statute_of_limitations PASSED
tests/test_pipeline.py::TestPositiveScenarios::test_decision_is_accept          PASSED
tests/test_pipeline.py::TestPositiveScenarios::test_sources_cited_in_answer     PASSED
tests/test_pipeline.py::TestPositiveScenarios::test_pipeline_english_mode       PASSED
tests/test_pipeline.py::TestPositiveScenarios::test_pipeline_kazakh_mode        PASSED
tests/test_pipeline.py::TestPositiveScenarios::test_all_required_fields_present PASSED
tests/test_pipeline.py::TestNegativeScenarios::test_off_topic_question          PASSED
tests/test_pipeline.py::TestNegativeScenarios::test_prompt_injection_attack     PASSED
tests/test_pipeline.py::TestNegativeScenarios::test_empty_question_handled      PASSED
tests/test_pipeline.py::TestNegativeScenarios::test_very_long_question          PASSED
tests/test_pipeline.py::TestNegativeScenarios::test_special_characters          PASSED
tests/test_pipeline.py::TestNegativeScenarios::test_question_wrong_country      PASSED

24 passed in 255s
```

Covers: Query Rewrite, RRF Fusion, Doc Grader, Hallucination Check, full pipeline (positive + negative scenarios), multilingual mode (RU/KZ/EN), adversarial inputs (prompt injection, XSS, empty input, off-topic).

---

## Project Structure

```
├── app_agentic_rag.py              # Gradio UI + streaming handler
├── langgraph_rag.py                # LangGraph StateGraph (pipeline controller)
├── ingest.py                       # Document indexing script
├── eval_run.py                     # Batch RAGAS evaluation runner
├── pytest.ini                      # Test markers (unit / integration)
├── agents/
│   ├── query_rewrite.py            # Step 1: Query expansion + translation
│   ├── vector_db.py                # Step 2a: Semantic search (ChromaDB, 15 docs)
│   ├── bm25_index.py               # Step 2b: Lexical search (BM25, 15 docs)
│   ├── rrf_fusion.py               # Step 2c: RRF Fusion → 8 docs
│   ├── doc_grader.py               # Step 3: Document grading (parallel LLM)
│   ├── cross_encoder.py            # Step 4: LLM re-ranking (score 0–10)
│   ├── llm_generate.py             # Step 5: Answer generation + citations
│   ├── hallucination_check.py      # Step 6: Grounding verification
│   ├── mcp_legal_search.py         # MCP fallback: gov portals
│   └── shared.py                   # LangFuse + utilities
├── mcp_server/
│   └── legal_kz_server.py          # FastMCP server for Kazakhstan legal DB
├── tests/
│   └── test_pipeline.py            # 24 tests (10 unit + 14 integration)
├── eval_results/                   # RAGAS evaluation output (JSON + TXT)
├── docker-compose.langfuse.yml     # LangFuse + PostgreSQL
├── architecture_diagram_v4.png     # Final architecture diagram (3 agents)
├── architecture_diagram_v4.svg     # Architecture diagram — vector
├── Executive_Summary.md            # 1–2 page project overview
├── Self_Review.md                  # Architecture decisions and trade-offs
├── Capstone_Presentation.pptx      # Slide deck
└── .env.example                    # Environment variable template
```

---

## Evaluation Results

RAGAS scores from automated English evaluation (2026-05-12, 12 questions, scale 0–10):

**Scenario 1 — Local RAG** (ChromaDB + BM25, 4 legal codes):

| Query | Faith | Relev | Score |
|-------|-------|-------|-------|
| Statute of limitations (Civil Code) | 10.0 | 9.8 | **9.9** |
| Employer liability for delayed salary | 10.0 | 9.1 | **9.5** |
| Vacation days (Labor Code) | 10.0 | 9.1 | **9.5** |
| Grounds for terminating employment contract | 10.0 | 9.7 | **9.8** |
| Grounds for declaring a transaction void | 10.0 | 9.5 | **9.8** |

**Scenario 2 — MCP Fallback** (adilet.zan.kz, kgd.gov.kz, egov.kz):

| Query | Faith | Relev | Score |
|-------|-------|-------|-------|
| Penalties for late tax filing (SME) | 6.2 | 10.0 | **8.1** |
| Sanitary requirements for food service | 7.5 | 9.1 | **8.3** |
| Business registration status on egov.kz | 7.5 | 10.0 | **8.7** |
| Electronic digital signature (EDS) for business | 10.0 | 9.2 | **9.6** |
| Requirements for opening a pharmacy | 6.7 | 10.0 | **8.3** |

**Scenario 3 — Edge Cases** (out-of-scope questions):

| Query | Faith | Relev | Score |
|-------|-------|-------|-------|
| Customer slips and falls in shop (tort liability) | 0.0 | 5.4 | 2.7 |
| Temporarily closing business for vacation | 0.0 | 7.5 | 3.7 |

**Overall: 8.2/10 · Pass rate (≥7.0): 10/12 questions**

Scenario 1 avg: **9.7/10** (5/5 pass) — Local RAG covers all 4 legal codes with high faithfulness.  
Scenario 2 avg: **8.6/10** (5/5 pass) — MCP fallback finds relevant context on government portals.  
Scenario 3 avg: **3.2/10** (0/2 pass, expected) — Out-of-scope questions; LLM answers from its own knowledge, Faithfulness = 0.0, FORCE ACCEPT after 3 retries. See Known Limitations.

---

## Known Limitations

**LangFuse shows $0.00 cost.** The integration uses the Python SDK's `trace()` method to log pipeline-level data (input, output, RAGAS scores, retry metadata). Individual OpenAI API calls are not wrapped as LangFuse generation spans, so token counts and per-call costs are not captured inside LangFuse. Actual cost (~$0.01/query) is tracked in the OpenAI Dashboard. The SDK approach was chosen over LangChain callbacks because callbacks conflict with Gradio's streaming generator pattern.

**Reflect uses a fast-path on the first attempt.** When `hallucination_ok=True` on attempt 0, Reflect accepts the answer immediately without running RAGAS — to keep first-response latency low. RAGAS is computed asynchronously afterwards and displayed in the UI ~5 seconds later. This means a low-scoring answer (e.g. 3.5/10) can be accepted on the first attempt if the hallucination check passed. On retries (attempts 1+), RAGAS always runs synchronously and drives the accept/retry decision. Fix: run RAGAS synchronously on all attempts at the cost of ~5–10 seconds added to first-attempt latency.

**Out-of-scope questions: Faithfulness = 0.0, FORCE ACCEPT after 3 attempts.** When a question is outside the 4 legal codes and MCP cannot find grounded context on government portals, the LLM generates an answer from its own knowledge. Hallucination Check flags it as `Grounded: False`. RAGAS Faithfulness = 0.0 (stable with temperature=0). The system retries all 3 attempts and FORCE ACCEPTs the best answer. This is expected — the system signals the knowledge gap rather than silently returning an ungrounded answer.

Examples:
- "If a customer slips and falls in my shop, am I liable?" → 0/6 local docs kept, MCP found no links on all 3 attempts → LLM answered from own knowledge → Faithfulness **0.0/10**, Score **2.7/10**, FORCE ACCEPT

---

## Future Improvements

- **MCP query optimization** — send a short keyword query to DuckDuckGo instead of the full question, to improve URL resolution rate and reduce snippet-only results
- **Full-page MCP scraping** — when a URL is found, scrape the full page instead of the snippet to provide richer context for the LLM and improve Faithfulness scores
- **Expanded knowledge base** — add environmental, licensing, and IP law codes to cover more SMB scenarios
- **Kazakh-language embeddings** — add a dedicated KZ-language embedding model alongside the current RU/EN pipeline
- **Latency optimization** — reduce first-response time across the pipeline
