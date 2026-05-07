"""
Agent: LLM Generate
====================
Synthesizes the final answer from reranked document chunks + web context.
Citation block is built programmatically after LLM generation.
"""

import os
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .shared import get_lf_config, add_trace

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

_SYSTEM_BY_LANG = {
    "ru": (
        "Ты — юридический ассистент для МСБ Казахстана. Правила:\n"
        "1. ГЛАВНЫЙ ИСТОЧНИК — «Контекст из документа» (локальная база законов РК).\n"
        "2. «MCP (adilet.zan.kz)» — используй если в документе нет ответа.\n"
        "3. «Веб» — только если нет ответа ни в документе, ни в MCP.\n"
        "4. Если информации нет нигде — скажи «Информация не найдена».\n\n"
        "Дай ответ: 2-4 предложения на русском языке. "
        "НЕ добавляй раздел Источники — он будет добавлен автоматически."
    ),
    "kz": (
        "Сіз — Қазақстан шағын және орта бизнесіне арналған заңдық көмекшісіз. Ережелер:\n"
        "1. БАСТЫ ДЕРЕК КӨЗ — «Құжаттан контекст» (ҚР заңдарының жергілікті базасы).\n"
        "2. «MCP (adilet.zan.kz)» — құжатта жауап болмаса пайдалан.\n"
        "3. «Веб» — тек құжатта да, MCP-де де жауап болмаса.\n"
        "4. Ешжерде ақпарат болмаса — «Ақпарат табылмады» де.\n\n"
        "Жауапты қазақ тілінде 2-4 сөйлеммен бер. "
        "Дереккөздер бөлімін қоспа — ол автоматты қосылады."
    ),
    "en": (
        "You are a legal assistant for SMB in Kazakhstan. Rules:\n"
        "1. PRIMARY SOURCE — 'Document context' (local Kazakhstan law database).\n"
        "2. 'MCP (adilet.zan.kz)' — use if the document has no answer.\n"
        "3. 'Web' — only if neither document nor MCP has the answer.\n"
        "4. If no information found anywhere — say 'Information not found'.\n\n"
        "Give a 2-4 sentence answer in English. "
        "Do NOT add a Sources section — it will be added automatically."
    ),
}

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    ("human",
     "Контекст из документа (законы РК):\n{context}\n\n"
     "MCP — adilet.zan.kz:\n{mcp_context}\n\n"
     "Дополнительно (веб):\n{web_context}\n\n"
     "Вопрос: {question}"),
])
chain = PROMPT | llm | StrOutputParser()


def _build_context(docs: list) -> str:
    parts = []
    for doc in docs[:4]:
        meta        = doc.metadata
        source      = meta.get("source", meta.get("file", ""))
        page        = meta.get("page", "?")
        page_label  = meta.get("page_label", "стр.")
        source_name = os.path.splitext(os.path.basename(source))[0] if source else "Документ"
        parts.append(f"[Источник: {source_name} | {page_label} {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _extract_mcp_sources(mcp_context: str) -> list:
    sources = []
    seen_urls = set()
    blocks = re.split(r'\n---\n', mcp_context)
    for block in blocks:
        url_match   = re.search(r'(https?://[^\s\)]+)', block)
        title_match = re.search(r'\*\*(.+?)\*\*', block)
        if url_match:
            url = url_match.group(1).rstrip(".,)")
            if "/compare" in url or "/info" in url or url in seen_urls:
                continue
            seen_urls.add(url)
            title = title_match.group(1).strip() if title_match else url
            sources.append({"title": title, "url": url})
    return sources


_CITATION_LABELS = {
    "ru": {"sources": "Источники", "local_db": "Локальная база", "article": "Статья", "page": "стр.", "doc": "Документ"},
    "kz": {"sources": "Дереккөздер", "local_db": "Жергілікті база", "article": "Бап", "page": "бет", "doc": "Құжат"},
    "en": {"sources": "Sources", "local_db": "Local DB", "article": "Article", "page": "p.", "doc": "Document"},
}


def _build_citation(docs: list, mcp_context: str, lang: str = "ru") -> str:
    lbl   = _CITATION_LABELS.get(lang, _CITATION_LABELS["ru"])
    lines = []

    if docs:
        seen = set()
        for doc in docs[:2]:
            meta        = doc.metadata
            source      = meta.get("source", "")
            page        = meta.get("page", "?")
            page_label  = lbl["page"]
            source_name = os.path.splitext(os.path.basename(source))[0] if source else lbl["doc"]
            key = f"{source_name}_{page}"
            if key in seen:
                continue
            seen.add(key)
            article_match = re.search(r'[Сс]татья\s+(\d+)|[Аа]рticle\s+(\d+)', doc.page_content)
            if article_match:
                num = article_match.group(1) or article_match.group(2)
                article = f", {lbl['article']} {num}"
            else:
                article = ""
            lines.append(f"📄 [{lbl['local_db']}] {source_name}{article}, {page_label} {page}")

    if mcp_context and "не найдено" not in mcp_context:
        for src in _extract_mcp_sources(mcp_context)[:2]:
            domain = re.search(r'https?://([^/]+)', src['url'])
            domain_label = domain.group(1) if domain else "gov.kz"
            lines.append(f"⚖️ [MCP {domain_label}] [{src['title']}]({src['url']})")

    if not lines:
        return ""
    return f"\n\n---\n**{lbl['sources']}:**\n" + "\n".join(f"- {l}" for l in lines)


def node_generate(state: dict) -> dict:
    question      = state["question"]
    docs          = state.get("reranked_docs") or state.get("graded_docs", [])
    web_context   = state.get("web_context", "")
    mcp_context   = state.get("mcp_context", "")
    lang          = state.get("language", "ru")
    context       = _build_context(docs)
    system_prompt = _SYSTEM_BY_LANG.get(lang, _SYSTEM_BY_LANG["ru"])

    answer_text = chain.invoke(
        {"system_prompt": system_prompt, "context": context,
         "mcp_context": mcp_context, "web_context": web_context,
         "question": question},
        config=get_lf_config(),
    )

    citation = _build_citation(docs, mcp_context, lang)
    answer   = answer_text.strip() + citation

    return {
        "answer": answer,
        "trace": add_trace(state, "LLM_GENERATE", f"Answer: '{answer_text[:100]}'"),
    }
