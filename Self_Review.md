# Self-Review
## Agentic RAG — Legal Assistant for SMB Kazakhstan

**Author:** Guldana Kassym-Ashim  
**Program:** EPAM Generative AI for Software Development, 2026

---

## 1. Project Goals — Achieved?

| Goal | Status | Notes |
|------|--------|-------|
| Build a working RAG pipeline for Kazakhstan law | ✅ Done | LangGraph + ChromaDB + 4 legal codes |
| Implement agentic behavior (retry loop) | ✅ Done | Reflect/Reformulate nodes, up to 3 attempts |
| Integrate external data source (MCP) | ✅ Done | adilet.zan.kz, kgd.gov.kz, egov.kz |
| Multilingual support (KZ/EN/RU) | ✅ Done | Full UI + LLM prompts + auto-translation for retrieval |
| Quality evaluation (RAGAS) | ✅ Done | Faithfulness + AnswerRelevancy, gpt-4o-mini judge |
| Observability | ✅ Done | LangFuse tracing with full pipeline trace + token costs |
| Document upload at runtime | ✅ Done | PDF + DOCX indexing via Gradio UI |
| Test suite | ✅ Done | 24/24 pass — 10 unit + 14 integration (verified 2026-05-07) |
| User feedback (👍/👎) | ⚠️ Partial | Not implemented — identified as missing NFR |

---

## 2. Technical Decisions — Rationale

### Why LangGraph?
LangGraph provides explicit state management (TypedDict) and conditional routing between agents. This was essential for the retry loop (Reflect → Reformulate → Query Rewrite) and the fallback path (Doc Grader → MCP → Cross-Encoder). A simple chain would not support this level of control flow.

### Why Hybrid Search (Vector + BM25)?
Semantic search alone misses exact legal terms (article numbers, specific monetary amounts). BM25 handles exact keyword matching. RRF Fusion combines both rankings without requiring score normalization — a well-established retrieval technique that consistently outperforms either method alone.

### Why LLM-based Cross-Encoder?
Using GPT-4.1-mini for re-ranking keeps the stack simple, leverages the same model already used for generation, and produces interpretable scores with reasoning. It always runs — both after local retrieval and after MCP fallback — ensuring consistent quality before generation.

### Why Two Different Models (GPT-4.1-mini and GPT-4o-mini)?
GPT-4.1-mini handles all 6 pipeline agents (query rewrite, grading, re-ranking, generation, hallucination check, reformulation). GPT-4o-mini is used exclusively for RAGAS evaluation (~$0.001/eval) — a deliberate cost optimization since evaluation requires different capabilities than generation.

### Why RAGAS for evaluation?
RAGAS provides automated, reference-free metrics specifically designed for RAG systems. Faithfulness checks if the answer is grounded in retrieved context; Answer Relevancy checks if the answer addresses the question. Together they cover the two main failure modes of RAG. Running it after every response closes the quality loop automatically.

### Why MCP?
MCP (Model Context Protocol) is the standard protocol for connecting LLMs to external tools. Building a custom MCP server for Kazakhstan government portals demonstrates understanding of tool integration and provides access to up-to-date legal information not in the static knowledge base.

---

## 3. What Went Well

- **Pipeline reliability** — the retry loop successfully catches hallucinations and poor retrievals, regenerating with reformulated queries using different expansion strategies
- **Performance optimization** — parallelizing Doc Grader (ThreadPoolExecutor) reduced grading latency ~6× without changing logic
- **MCP integration** — live search results complement the static knowledge base effectively for current information (e.g., 2026 minimum wage from adilet.zan.kz)
- **Streaming UI** — users see the agent trace in real time, making the system transparent and debuggable
- **Multilingual** — all three languages (KZ/EN/RU) work correctly end-to-end including UI, prompts, citations, and auto-translation for retrieval
- **Test coverage** — 24 tests covering unit behavior and full integration scenarios including adversarial inputs and multilingual mode

---

## 4. What Could Be Improved

### User Feedback (👍/👎)
Not implemented. Adding thumbs up/down buttons would enable collecting user satisfaction signals and logging them to LangFuse for further analysis. This would close the evaluation loop from automated (RAGAS) to human feedback.

### Parallel Hybrid Search
In the current implementation, VectorDB and BM25 run sequentially in LangGraph. Architecturally, both operations are independent and could run in parallel using LangGraph's parallel node execution, which would reduce latency.

### Cross-Encoder Efficiency
The current LLM-based cross-encoder makes one API call per document. A dedicated neural cross-encoder model (e.g., ms-marco) would be faster and cheaper for high-traffic scenarios, though it would add a separate model deployment.

### Knowledge Base Coverage
The current knowledge base covers 4 major codes + 2 company regulations. Adding more regulatory documents (environmental, licensing, IP law) would expand coverage for SMB needs.

---

## 5. Lessons Learned

- **Agentic systems need explicit state** — LangGraph's TypedDict state made debugging much easier than implicit chain state
- **Evaluation must be designed upfront** — integrating RAGAS early helped catch quality issues during development
- **LLM prompts are code** — small wording changes in grader/checker prompts significantly affect behavior (e.g., Doc Grader prompt had to be relaxed for cross-language relevance judgment)
- **Streaming changes UX fundamentally** — seeing the agent trace in real time makes the system feel transparent; users understand why an answer takes longer on retries
- **MCP is powerful but fragile** — web scraping depends on site structure; adilet.zan.kz layout changes could break retrieval

---

## 6. Overall Self-Assessment

The project successfully implements a production-grade agentic RAG system that goes significantly beyond a basic retrieve-and-generate pipeline. The key differentiators — retry loop, RAGAS evaluation with two dedicated models, MCP fallback, multilingual support, observability, and a full test suite — demonstrate understanding of real-world RAG challenges.

The main gap is the missing user feedback mechanism (👍/👎), which would have completed the evaluation story from automated metrics to human signals.

**Self-score: 9 / 10**
