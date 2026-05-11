
"""
Agent: Hallucination Check
===========================
Verifies that the generated answer is grounded in the retrieved context
and does not contain hallucinated facts.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .shared import get_lf_config, parse_json, add_trace

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Check: is the answer grounded in the provided context (documents + web + MCP)?\n"
     "The answer is correct if it relies on ANY of the context sources.\n"
     "Reply STRICTLY as JSON: "
     "{{\"grounded\": true|false, "
     "\"issues\": \"if there are issues — what exactly, otherwise empty\"}}"),
    ("human", "Document context:\n{context}\n\n"
              "Web context:\n{web_context}\n\n"
              "MCP context:\n{mcp_context}\n\n"
              "Answer:\n{answer}"),
])
chain = PROMPT | llm | StrOutputParser()


def node_hallucination_check(state: dict) -> dict:
    docs        = state.get("reranked_docs") or state.get("graded_docs", [])
    answer      = state["answer"]
    web_context = state.get("web_context", "")
    mcp_context = state.get("mcp_context", "")
    context     = "\n\n---\n\n".join(d.page_content for d in docs[:3])

    raw    = chain.invoke(
        {"context": context, "web_context": web_context,
         "mcp_context": mcp_context, "answer": answer},
        config=get_lf_config(),
    )
    result   = parse_json(raw, {"grounded": True, "issues": ""})
    grounded = result.get("grounded", True)
    issues   = result.get("issues", "")

    detail = f"Grounded: {grounded}" + (f" | Issues: {issues}" if issues else "")
    return {
        "hallucination_ok": grounded,
        "trace": add_trace(state, "HALLUCINATION_CHECK", detail),
    }
