# Self-Review
## Agentic RAG — Legal Assistant for SMB Kazakhstan

**Author:** Dana Kassym  
**Date:** May 19, 2026

---

## 1. Project Goals — Achieved?

| Goal | Status | Notes |
|------|--------|-------|
| Build a working RAG pipeline for Kazakhstan law | ✅ Done | LangGraph + ChromaDB + 4 legal codes |
| Implement agentic behavior (retry loop) | ✅ Done | Reflect/Reformulate nodes, up to 3 attempts |
| Integrate external data source (MCP) | ✅ Done | adilet.zan.kz, kgd.gov.kz, egov.kz |
| Multilingual support (RU/KZ/EN) | ✅ Done | Full UI + LLM prompts per language |
| Quality evaluation (RAGAS) | ✅ Done | Faithfulness + AnswerRelevancy after every response |
| Observability | ✅ Done | LangFuse tracing with full pipeline trace |
| Document upload at runtime | ✅ Done | PDF + DOCX indexing via Gradio UI |
| User feedback (👍/👎) | ⚠️ Partial | Not implemented — identified as missing NFR |

---

## 2. Technical Decisions — Rationale

### Why LangGraph?
LangGraph provides explicit state management and conditional routing between agents. This was essential for the retry loop (Reflect → Reformulate → Query Rewrite) and the fallback path (Doc Grader → MCP → Cross-Encoder). A simple chain would not support this level of control flow.

### Why Hybrid Search (Vector + BM25)?
Semantic search alone misses exact legal terms (article numbers, specific amounts). BM25 handles exact keyword matching. RRF Fusion combines both rankings without requiring score normalization — this is a well-established retrieval technique.

### Why LLM-based Cross-Encoder?
A dedicated neural cross-encoder (e.g., ms-marco) would require a separate model deployment. Using GPT-4.1-mini for re-ranking keeps the stack simple, leverages the same model already used for generation, and produces interpretable scores with reasoning.

### Why RAGAS for evaluation?
RAGAS provides automated, reference-free metrics specifically designed for RAG systems. Faithfulness checks if the answer is grounded in retrieved context; Answer Relevancy checks if the answer addresses the question. Together they cover the two main failure modes of RAG.

### Why MCP?
MCP (Model Context Protocol) is the standard protocol for connecting LLMs to external tools. Building a custom MCP server for adilet.zan.kz demonstrates understanding of tool integration and provides access to up-to-date legal information not in the local knowledge base.

---

## 3. What Went Well

- **Pipeline reliability** — the retry loop successfully catches hallucinations and poor retrievals, regenerating with reformulated queries
- **Performance optimization** — parallelizing Doc Grader (ThreadPoolExecutor) reduced grading latency ~6× without changing logic
- **MCP integration** — live search results complement the static knowledge base effectively for current information (e.g., 2026 minimum wage)
- **Streaming UI** — users see the agent trace in real time, making the system transparent and debuggable
- **Multilingual** — all three languages (RU/KZ/EN) work correctly end-to-end including UI, prompts, and citations

---

## 4. What Could Be Improved

### User Feedback (👍/👎)
Not implemented. Adding thumbs up/down buttons would enable collecting user satisfaction signals and logging them to LangFuse for further analysis. This would close the evaluation loop from automated (RAGAS) to human feedback.

### Parallel Hybrid Search
In the current implementation, VectorDB and BM25 run sequentially in LangGraph. Architecturally, both operations are independent and could run in parallel using LangGraph's parallel node execution, which would reduce latency.

### Test Coverage
No automated tests were written. Unit tests for individual agents (mock LLM responses) and integration tests for the full pipeline would improve reliability and make refactoring safer.

### Cross-Encoder Efficiency
The current LLM-based cross-encoder makes one API call per document (up to 6 calls). A dedicated neural cross-encoder model would be faster and cheaper for high-traffic scenarios.

### Knowledge Base Coverage
The current knowledge base covers 4 major codes + 2 company regulations. Adding more regulatory documents (e.g., environmental, licensing, IP law) would expand coverage for SMB needs.

---

## 5. Lessons Learned

- **Agentic systems need explicit state** — LangGraph's TypedDict state made debugging much easier than implicit chain state
- **Evaluation must be designed upfront** — integrating RAGAS early helped catch quality issues during development
- **LLM prompts are code** — small wording changes in grader/checker prompts significantly affect behavior; they need the same versioning rigor as code
- **Streaming changes UX fundamentally** — seeing the agent trace in real time makes the system feel transparent; users understand why an answer takes longer on retries
- **MCP is powerful but fragile** — web scraping depends on site structure; `adilet.zan.kz` layout changes could break retrieval

---

## 6. Overall Self-Assessment

The project successfully implements a production-grade agentic RAG system that goes significantly beyond a basic retrieve-and-generate pipeline. The key differentiators — retry loop, RAGAS evaluation, MCP fallback, and observability — demonstrate understanding of real-world RAG challenges.

The main gap is the missing user feedback mechanism (👍/👎), which would have completed the evaluation story from automated metrics to human signals.

**Self-score: 8.5 / 10**
