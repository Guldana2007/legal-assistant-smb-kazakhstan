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
| User feedback (👍/👎) | ✅ Done | 👍/👎 buttons log to LangFuse via log_feedback() |

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
Implemented — 👍/👎 buttons in the Gradio UI log user ratings to LangFuse via `log_feedback()`. Each rating is stored as a trace with the question, answer, and positive/negative signal. This closes the evaluation loop from automated RAGAS metrics to human feedback.

### Cross-Encoder Efficiency
The current LLM-based cross-encoder makes one API call per document. A dedicated neural cross-encoder model (e.g., ms-marco) would be faster and cheaper for high-traffic scenarios, though it would add a separate model deployment.

### Knowledge Base Coverage
The current knowledge base covers 4 major legal codes. Adding more regulatory documents (environmental, licensing, IP law) would expand coverage for SMB needs.

### LangFuse Cost Tracking
LangFuse traces show $0.00 cost and zero token usage because the integration uses the Python SDK's `_lf_client.trace()` directly — logging the full pipeline input, output, RAGAS scores, and metadata — but does not instrument individual OpenAI API calls as LangFuse generation spans. Without generation-level spans (model name + token counts), LangFuse has no data to price. Actual per-query cost (~$0.01) is tracked in the OpenAI Dashboard. The decision to use SDK-level tracing rather than LangChain callbacks was intentional: LangChain callbacks conflict with Gradio's streaming generator pattern, causing incomplete traces. A future improvement is to wrap each LLM call in a `_lf_client.generation()` span to capture token costs inside LangFuse.

### MCP Faithfulness vs. URL Resolution
RAGAS Faithfulness for MCP answers depends on whether the retrieved snippets actually contain the specific answer — not on URL presence. When MCP finds snippets with direct content (e.g. egov.kz page mentioning "Personal Account → My applications"), Faithfulness is 10.0/10. When MCP returns general or off-topic snippets, the LLM fills gaps from its own knowledge and Faithfulness drops (e.g. 2.0/10 for state duty fees where MCP returned passport/investment pages instead of fee schedules). Fix: shorter keyword queries to improve snippet relevance + full-page scraping when a URL is resolved.

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

User feedback (👍/👎) is implemented and logged to LangFuse, completing the evaluation loop from automated RAGAS metrics to human signals.


