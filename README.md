# Agentic RAG — Legal Assistant for SMB Kazakhstan

A conversational legal assistant for small and medium businesses in Kazakhstan, built on an agentic RAG pipeline with LangGraph, ChromaDB, and Gradio.

## Overview

The system answers legal questions about Kazakhstan legislation by combining local document search (RAG) with live web search via MCP (adilet.zan.kz). It supports Russian, Kazakh, and English.

**Pipeline:**
```
Query Rewrite → Hybrid Search (Vector DB + BM25) → RRF Fusion → Doc Grader
    → Cross-Encoder → Generate → Hallucination Check → Reflect
    └─ fallback: Doc Grader → MCP (adilet.zan.kz) → Cross-Encoder
    └─ retry:    Reflect → Reformulate → Query Rewrite (loop)
```

## Architecture

![Architecture Diagram](architecture_diagram.png)

**7 agents in the pipeline:**
1. **Query Rewrite** — expands query (HyDE / Step-Back / Keyword / none)
2. **Hybrid Search** — ChromaDB (semantic) + BM25 (lexical) run in parallel
3. **RRF Fusion** — Reciprocal Rank Fusion (K=60) merges both result sets
4. **Doc Grader** — filters irrelevant chunks (parallel, 6 threads)
5. **Cross-Encoder** — LLM re-ranking, scores 0–10 (gpt-4.1-mini)
6. **Hallucination Check** — verifies answer is grounded in context
7. **Reflect / Reformulate** — RAGAS-based quality judge; retries if score < 7

**MCP fallback:** when Doc Grader finds < 2 relevant chunks, the system queries adilet.zan.kz live via MCP server.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph (StateGraph) |
| Vector DB | ChromaDB (~17K chunks) |
| Embeddings | OpenAI text-embedding-3-small |
| LLM | GPT-4.1-mini |
| Lexical search | BM25Okapi (rank-bm25) |
| Evaluation | RAGAS (Faithfulness + AnswerRelevancy) |
| Observability | LangFuse |
| UI | Gradio |
| MCP server | Custom (adilet.zan.kz scraper) |

## Knowledge Base

Local documents indexed in ChromaDB:
- Трудовой кодекс РК (Labor Code)
- Налоговый кодекс РК (Tax Code)
- Гражданский кодекс РК (Civil Code)
- Предпринимательский кодекс РК (Entrepreneurship Code)
- TechStart_Reglament_2025.docx
- DigiCor_Reglament_2025.docx

## Requirements

- Python 3.11+
- Docker (for LangFuse observability)
- OpenAI API key

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/capstone-epam-rag.git
cd capstone-epam-rag

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

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...

# LangFuse (optional — for observability)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

## Running

### 1. Start LangFuse (optional, for observability)
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

## Usage

1. Select language: Қазақша / English / Русский
2. Type a legal question or click a sample question
3. Adjust **Max Attempts** (1–5) and **Query Expansion** strategy if needed
4. Click **Submit** — the answer streams with live Agent Trace
5. RAGAS scores (Faithfulness + Answer Relevancy) appear ~5 sec after the answer

### Upload custom documents

Go to the **Upload** tab to add your own PDF or DOCX files. They will be chunked and indexed automatically.

## Project Structure

```
├── app_agentic_rag.py          # Gradio UI + streaming handler
├── langgraph_rag.py            # LangGraph StateGraph (pipeline controller)
├── ingest.py                   # Document indexing script
├── agents/
│   ├── query_rewrite.py        # Agent 1: Query expansion
│   ├── vector_db.py            # Agent 2a: Semantic search (ChromaDB)
│   ├── bm25_index.py           # Agent 2b: Lexical search (BM25)
│   ├── rrf_fusion.py           # Agent 3: RRF Fusion
│   ├── doc_grader.py           # Agent 4: Document grading (parallel)
│   ├── cross_encoder.py        # Agent 5: LLM re-ranking
│   ├── llm_generate.py         # Agent 6: Answer generation
│   ├── hallucination_check.py  # Agent 6b: Grounding verification
│   ├── mcp_legal_search.py     # MCP fallback: adilet.zan.kz
│   └── shared.py               # LangFuse + utilities
├── mcp_server/
│   └── legal_kz_server.py      # MCP server for Kazakhstan legal DB
├── docs/                       # PDF/DOCX knowledge base files
├── docker-compose.langfuse.yml # LangFuse + PostgreSQL
└── architecture_diagram.png    # System architecture diagram
```

## Author

Dana Kassym — EPAM Generative AI for Software Development, 2025
