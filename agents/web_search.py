"""
Agent: Web Search
==================
Searches DuckDuckGo for additional context and synthesizes a short summary.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .shared import get_lf_config, add_trace

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Write a brief summary based on the search results. 2–3 sentences."),
    ("human", "Query: {query}\n\nResults:\n{results}"),
])
synthesis_chain = SYNTHESIS_PROMPT | llm | StrOutputParser()


def _ddg_search(query: str, max_results: int = 3) -> list[dict]:
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        print(f"  [WebSearch] error: {e}")
        return []


def node_web_search(state: dict) -> dict:
    if not state.get("use_web", True):
        return {"web_context": "Web search disabled",
                "trace": add_trace(state, "WEB_SEARCH", "Disabled")}

    question = state["question"]
    results  = _ddg_search(question)

    if not results:
        return {"web_context": "No web results found",
                "trace": add_trace(state, "WEB_SEARCH", "No results")}

    results_text = "\n\n".join(
        f"[{i+1}] {r.get('title','')}\n{r.get('body','')}"
        for i, r in enumerate(results)
    )
    summary = synthesis_chain.invoke(
        {"query": question, "results": results_text},
        config=get_lf_config(),
    )

    # Append URLs so LLM can cite them in the answer
    urls_block = "\n".join(
        f"[{i+1}] {r.get('href', '')}"
        for i, r in enumerate(results) if r.get('href')
    )
    web_context = f"{summary}\n\nWeb sources:\n{urls_block}"

    return {
        "web_context": web_context,
        "trace": add_trace(state, "WEB_SEARCH",
                           f"Found {len(results)} results | '{summary[:80]}'"),
    }
