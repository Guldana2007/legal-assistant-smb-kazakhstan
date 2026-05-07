"""
Agent: Query Rewrite
=====================
Rewrites / expands the user query before retrieval.
Strategies: none | hyde | step_back | keyword
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .shared import get_lf_config, add_trace

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a search query specialist. Strategy: {strategy}\n"
     "- hyde: write a hypothetical paragraph from Kazakhstan legislation containing the answer\n"
     "- step_back: broaden the question to a more general form but ALWAYS keep Kazakhstan (RK) context\n"
     "- keyword: extract 3–5 key terms separated by spaces\n"
     "- none: return the question unchanged\n"
     "IMPORTANT: Always preserve the Kazakhstan (RK) context. Never change the country.\n"
     "Return ONLY the reformulated query, nothing else."),
    ("human", "{question}"),
])
chain = PROMPT | llm | StrOutputParser()

TRANSLATE_TO_RU_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Translate the following question to Russian for searching in a Russian-language "
     "legal document database. Return ONLY the translated question, nothing else."),
    ("human", "{question}"),
])
translate_chain = TRANSLATE_TO_RU_PROMPT | llm | StrOutputParser()


def node_query_rewrite(state: dict) -> dict:
    question  = state["question"]
    expansion = state.get("expansion", "none")
    attempt   = state.get("attempt", 0)
    language  = state.get("language", "ru")

    if expansion == "none" or attempt == 0:
        new_query = question
    else:
        new_query = chain.invoke(
            {"question": question, "strategy": expansion},
            config=get_lf_config(),
        ).strip()

    # Translate query to Russian for retrieval when UI language is EN or KZ
    # (knowledge base documents are in Russian)
    if language in ("en", "kz") and new_query.strip():
        new_query = translate_chain.invoke(
            {"question": new_query},
            config=get_lf_config(),
        ).strip()

    return {
        "current_query": new_query,
        "trace": add_trace(state, "QUERY_REWRITE",
                           f"strategy={expansion} | lang={language} | query='{new_query[:80]}'"),
    }
