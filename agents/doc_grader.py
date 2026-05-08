"""
Agent: Doc Grader
==================
Filters retrieved documents, keeping only those relevant to the question.
Grades all docs in parallel to reduce latency.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .shared import get_lf_config, parse_json, add_trace

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Assess whether the fragment directly helps answer the question (they may be in different languages).\n"
     "Return true ONLY if the fragment contains specific information that directly answers the question.\n"
     "Return false if the fragment only shares keywords or is tangentially related but does not actually answer the question.\n"
     "Reply STRICTLY as JSON: {{\"relevant\": true|false, \"reason\": \"briefly\"}}"),
    ("human", "Question: {question}\n\nFragment:\n{doc}"),
])
chain = PROMPT | llm | StrOutputParser()


def _grade_single(doc, question: str):
    raw    = chain.invoke(
        {"question": question, "doc": doc.page_content[:500]},
        config=get_lf_config(),
    )
    result = parse_json(raw, {"relevant": False})
    return doc if result.get("relevant", False) else None


def node_doc_grader(state: dict) -> dict:
    # Use the translated retrieval query so grader compares same-language question vs docs
    question = state.get("current_query") or state["question"]
    docs     = state.get("fused_docs", [])
    batch    = docs[:6]

    # Grade all docs in parallel — ~6x faster than sequential
    order   = {id(doc): i for i, doc in enumerate(batch)}
    graded  = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_grade_single, doc, question): doc for doc in batch}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                graded.append(result)

    graded.sort(key=lambda d: order.get(id(d), 999))

    return {
        "graded_docs": graded,
        "trace": add_trace(state, "DOC_GRADER",
                           f"Kept {len(graded)}/{len(batch)} relevant docs"
                           + (" → routing to MCP" if len(graded) < 2 else "")),
    }
