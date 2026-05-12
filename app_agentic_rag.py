"""
Gradio UI — Legal Assistant for SMB Kazakhstan
===============================================
Pipeline: Query Rewrite → Vector DB + BM25 → RRF Fusion → Doc Grader
          → Cross-Encoder (RAG) | MCP adilet.zan.kz → LLM Generate
          → Hallucination Check → Reflect (done? / retry loop)

Languages: RU (русский) | KZ (қазақша) | EN (English)
"""

import sys, time, warnings
warnings.filterwarnings("ignore")

from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import atexit
from langgraph_rag import run_graph, rag_graph, vectorstore, _run_ragas
from agents.shared import flush_langfuse, log_trace

atexit.register(flush_langfuse)

# ── Translations ──────────────────────────────────────────────────────────────

TRANSLATIONS = {
    "ru": {
        "title":            "# ⚖️ Юридический ассистент для МСБ Казахстана",
        "tab_ask":          "Задать вопрос",
        "tab_upload":       "Загрузить документ",
        "q_label":          "Вопрос",
        "q_placeholder":    "Задайте вопрос по законодательству Казахстана...",
        "submit":           "Отправить",
        "processing":       "⏳ Обработка...",
        "no_question":      "_Введите вопрос_",
        "scenario2":        "**Сценарий 2 — RAG не нашёл → MCP (adilet.zan.kz, kgd.gov.kz, egov.kz)**",
        "scenario1":        "**Сценарий 1 — только RAG (локальная база)**",
        "upload_title":     "### Управление базой знаний\nПоддерживаемые форматы: **PDF**, **DOCX**",
        "upload_section":   "#### Загрузить документ",
        "file_label":       "Выберите PDF или DOCX файл",
        "upload_btn":       "Загрузить и индексировать",
        "delete_section":   "#### Удалить документ",
        "delete_label":     "Имя файла (например: TechStart.docx)",
        "delete_ph":        "имя_файла.docx",
        "delete_btn":       "Удалить из базы",
        "docs_title":       "### Документы в базе",
        "refresh_btn":      "Обновить список",
        "no_docs":          "_Нет документов_",
        "lang_label":       "Язык",
        "sources_line":     "**Источники:** Локальная база (Налоговый, Трудовой, Гражданский, Предпринимательский кодексы РК) + MCP ([adilet.zan.kz](https://adilet.zan.kz), [kgd.gov.kz](https://kgd.gov.kz), [egov.kz](https://egov.kz))",
        "pipeline_line":    "**Pipeline:** Query Rewrite → Hybrid Search (Vector DB + BM25) → RRF Fusion → Doc Grader → Cross-Encoder → Generate → Hallucination Check → Reflect | нет докум. → MCP → Cross-Encoder | retry → Reformulate (loop)",
        "hard_q": [
            "Каковы штрафы за несвоевременную подачу налоговой отчётности для МСБ в Казахстане?",
            "Каковы санитарные требования для открытия предприятия общественного питания в Казахстане?",
            "Как проверить статус заявки на регистрацию бизнеса на портале egov.kz?",
            "Как получить электронную цифровую подпись (ЭЦП) для бизнеса в Казахстане?",
            "Каковы требования для открытия аптеки в Казахстане?",
        ],
        "easy_q": [
            "Какой срок исковой давности по Гражданскому кодексу РК?",
            "Какова ответственность работодателя за задержку выплаты заработной платы по ТК РК?",
            "Сколько дней отпуска положено работнику в Казахстане?",
            "Каковы основания для расторжения трудового договора по инициативе работодателя по ТК РК?",
            "Каковы основания для признания сделки недействительной по Гражданскому кодексу РК?",
        ],
        "scenario3":        "**Сценарий 3 — Сложные / пограничные случаи**",
        "edge_q": [
            "Каковы штрафы за несвоевременную подачу налоговой отчётности для МСБ в Казахстане?",
            "Если покупатель поскользнётся и упадёт в моём магазине, несу ли я ответственность?",
            "Если я хочу временно закрыть бизнес на время отпуска, нужно ли уведомлять государственные органы?",
        ],
    },
    "kz": {
        "title":            "# ⚖️ Қазақстан шағын және орта бизнесіне арналған заңдық көмекші",
        "tab_ask":          "Сұрақ қою",
        "tab_upload":       "Құжатты жүктеу",
        "q_label":          "Сұрақ",
        "q_placeholder":    "Қазақстан заңнамасы бойынша сұрақ қойыңыз...",
        "submit":           "Жіберу",
        "processing":       "⏳ Өңделуде...",
        "no_question":      "_Сұрақ енгізіңіз_",
        "scenario2":        "**Сценарий 2 — RAG таппады → MCP (adilet.zan.kz, kgd.gov.kz, egov.kz)**",
        "scenario1":        "**Сценарий 1 — тек RAG (жергілікті база)**",
        "upload_title":     "### Білім базасын басқару\nҚолдау көрсетілетін форматтар: **PDF**, **DOCX**",
        "upload_section":   "#### Құжатты жүктеу",
        "file_label":       "PDF немесе DOCX файлды таңдаңыз",
        "upload_btn":       "Жүктеу және индекстеу",
        "delete_section":   "#### Құжатты жою",
        "delete_label":     "Файл атауы (мысалы: TechStart.docx)",
        "delete_ph":        "файл_атауы.docx",
        "delete_btn":       "Базадан жою",
        "docs_title":       "### Базадағы құжаттар",
        "refresh_btn":      "Тізімді жаңарту",
        "no_docs":          "_Құжаттар жоқ_",
        "lang_label":       "Тіл",
        "sources_line":     "**Дереккөздер:** Жергілікті база (Салық, Еңбек, Азаматтық, Кәсіпкерлік кодекстер) + MCP ([adilet.zan.kz](https://adilet.zan.kz), [kgd.gov.kz](https://kgd.gov.kz), [egov.kz](https://egov.kz))",
        "pipeline_line":    "**Pipeline:** Query Rewrite → Hybrid Search (Vector DB + BM25) → RRF Fusion → Doc Grader → Cross-Encoder → Generate → Hallucination Check → Reflect | құжат жоқ → MCP → Cross-Encoder | retry → Reformulate (цикл)",
        "hard_q": [
            "Қазақстанда шағын және орта бизнес үшін салықтық есептілікті кешіктіргені үшін айыппұлдар қандай?",
            "Қазақстанда қоғамдық тамақтану орнын ашу үшін санитарлық талаптар қандай?",
            "egov.kz порталында бизнесті тіркеу өтінішінің мәртебесін қалай тексеруге болады?",
            "Қазақстанда бизнес үшін электрондық цифрлық қолтаңба (ЭЦҚ) қалай алуға болады?",
            "Қазақстанда дәріхана ашу талаптары қандай?",
        ],
        "easy_q": [
            "ҚР Азаматтық кодексі бойынша талап қою мерзімі қандай?",
            "ҚР ЕК бойынша жалақы төлемін кешіктіргені үшін жұмыс берушінің жауапкершілігі қандай?",
            "Қазақстанда жұмысшыға қанша күн демалыс берілуі тиіс?",
            "ҚР ЕК бойынша жұмыс берушінің бастамасымен еңбек шартын бұзудың негіздері қандай?",
            "ҚР АК бойынша мәмілені жарамсыз деп тануының негіздері қандай?",
        ],
        "scenario3":        "**Сценарий 3 — Күрделі / шекті жағдайлар**",
        "edge_q": [
            "Қазақстанда шағын және орта бизнес үшін салықтық есептілікті кешіктіргені үшін айыппұлдар қандай?",
            "Егер сатып алушы дүкенімде сүрініп жықылса, мен жауапты боламын ба?",
            "Егер демалысқа бизнесімді уақытша жабқым келсе, мемлекеттік органдарды хабардар ету керек пе?",
        ],
    },
    "en": {
        "title":            "# ⚖️ Legal Assistant for SMB Kazakhstan",
        "tab_ask":          "Ask a Question",
        "tab_upload":       "Upload Document",
        "q_label":          "Question",
        "q_placeholder":    "Ask a question about Kazakhstan legislation...",
        "submit":           "Submit",
        "processing":       "⏳ Processing...",
        "no_question":      "_Please enter a question_",
        "scenario2":        "**Scenario 2 — RAG not found → MCP (adilet.zan.kz, kgd.gov.kz, egov.kz)**",
        "scenario1":        "**Scenario 1 — Local RAG only**",
        "upload_title":     "### Knowledge Base Management\nSupported formats: **PDF**, **DOCX**",
        "upload_section":   "#### Upload Document",
        "file_label":       "Select PDF or DOCX file",
        "upload_btn":       "Upload & Index",
        "delete_section":   "#### Delete Document",
        "delete_label":     "File name (e.g. TechStart.docx)",
        "delete_ph":        "filename.docx",
        "delete_btn":       "Remove from database",
        "docs_title":       "### Documents in database",
        "refresh_btn":      "Refresh list",
        "no_docs":          "_No documents_",
        "lang_label":       "Language",
        "sources_line":     "**Sources:** Local database (Tax, Labor, Civil, Entrepreneurship Codes of Kazakhstan) + MCP ([adilet.zan.kz](https://adilet.zan.kz), [kgd.gov.kz](https://kgd.gov.kz), [egov.kz](https://egov.kz))",
        "pipeline_line":    "**Pipeline:** Query Rewrite → Hybrid Search (Vector DB + BM25) → RRF Fusion → Doc Grader → Cross-Encoder → Generate → Hallucination Check → Reflect | no docs → MCP → Cross-Encoder | retry → Reformulate (loop)",
        "hard_q": [
            "What are the penalties for late tax filing for SMEs in Kazakhstan?",
            "What are the sanitary requirements for opening a food service establishment in Kazakhstan?",
            "How do I check the status of my business registration application on egov.kz?",
            "How do I obtain an electronic digital signature (EDS) for business use in Kazakhstan?",
            "What are the requirements for opening a pharmacy in Kazakhstan?",
        ],
        "easy_q": [
            "What is the statute of limitations under the Civil Code of Kazakhstan?",
            "What is the employer's liability for delayed salary payment under the Labor Code of Kazakhstan?",
            "How many vacation days are employees entitled to in Kazakhstan?",
            "What are the grounds for terminating an employment contract at the employer's initiative under the Labor Code of Kazakhstan?",
            "What are the grounds for declaring a transaction void under the Civil Code of Kazakhstan?",
        ],
        "scenario3":        "**Scenario 3 — Edge Cases**",
        "edge_q": [
            "What are the penalties for late tax filing for SMEs in Kazakhstan?",
            "If a customer slips and falls in my shop, am I liable?",
            "If I want to close my business temporarily for vacation, do I need to notify authorities?",
        ],
    },
}

# ── Node icons ────────────────────────────────────────────────────────────────

NODE_ICONS = {
    "QUERY_REWRITE":       "✏️",
    "VECTOR_DB":           "🔵",
    "BM25_INDEX":          "📑",
    "RRF_FUSION":          "🔀",
    "DOC_GRADER":          "📋",
    "CROSS_ENCODER":       "⚖️",
    "LLM_GENERATE":        "💬",
    "HALLUCINATION_CHECK": "🔍",
    "REFLECT":             "🪞",
    "REFORMULATE":         "🔄",
    "MCP_LEGAL_SEARCH":    "⚖️",
}


# ── Formatters ────────────────────────────────────────────────────────────────

def _format_trace(trace: list, lang: str = "ru") -> str:
    attempt_label = {"ru": "Попытка", "kz": "Әрекет", "en": "Attempt"}
    regen_label   = {"ru": "перегенерация", "kz": "қайта генерация", "en": "regeneration"}
    lines = []
    attempt = 0
    last_was_reflect = False
    for e in trace:
        if e["node"] == "QUERY_REWRITE":
            attempt += 1
            lines.append(f"\n---\n### {attempt_label[lang]} {attempt}\n")
            last_was_reflect = False
        elif e["node"] == "LLM_GENERATE" and last_was_reflect:
            attempt += 1
            lines.append(f"\n---\n### {attempt_label[lang]} {attempt} ({regen_label[lang]})\n")
            last_was_reflect = False
        last_was_reflect = (e["node"] == "REFLECT")
        icon = NODE_ICONS.get(e["node"], "▸")
        lines.append(f"{icon} **{e['node']}** — {e['detail']}")
        for k, v in list((e.get("data") or {}).items())[:2]:
            val = str(v)[:120]
            lines.append(f"  - `{k}`: {val}")
    return "\n\n".join(lines)


def _format_scores(scores: dict, overall: float) -> str:
    if not scores:
        return ""
    lines = ["### Judge Scores\n"]
    bar_chars = 20
    all_none = all(v is None for v in scores.values())
    if all_none:
        lines.append("_RAGAS computing in background..._")
    else:
        for key, val in scores.items():
            if val is None:
                lines.append(f"**{key}**: —")
            else:
                filled = int((val / 10) * bar_chars)
                bar    = "█" * filled + "░" * (bar_chars - filled)
                lines.append(f"**{key}**: `{bar}` {val}/10")
    lines.append(f"\n**Overall: {overall}/10**")
    return "\n\n".join(lines)


def _format_summary(state: dict, max_attempts: int) -> str:
    attempts   = state.get("attempt", 1)
    decision   = state.get("decision", "accept")
    hall_ok    = state.get("hallucination_ok", True)
    question   = state.get("question", "")
    strategies = state.get("strategies_used", [])
    overall    = state.get("overall_score", 0.0)

    lines = [
        f"**Attempts:** {attempts} / {max_attempts}",
        f"**Decision:** {'✅ accept' if decision == 'accept' else f'🔄 {decision}'}",
        f"**Hallucination check:** {'✅ Grounded' if hall_ok else '⚠️ Issues found'}",
        f"**Overall score:** {overall}/10",
        f"**Strategies:** {', '.join(strategies) if strategies else '—'}",
        f"**Query:** {question}",
    ]
    return "\n\n".join(lines)


def _format_chunks(state: dict) -> str:
    docs = state.get("reranked_docs") or state.get("graded_docs") or []
    web  = state.get("web_context", "")
    lines = []

    if docs:
        lines.append("### Retrieved & Re-ranked Chunks\n")
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            lines.append(
                f"**Chunk {i}** — page {meta.get('page','?')} | "
                f"RRF: {meta.get('rrf_score','—')} | "
                f"Cross-Encoder: {meta.get('cross_encoder_score','—')}\n"
                f"```\n{doc.page_content[:400]}\n```"
            )

    mcp = state.get("mcp_context", "")
    if mcp and "не найдены" not in mcp and "error" not in mcp.lower() and mcp.strip():
        lines.append("\n### MCP — adilet.zan.kz\n")
        lines.append(mcp)

    if web and web not in ("", "Web search disabled", "Нет данных из веба"):
        lines.append("\n### Web Context\n")
        lines.append(web)

    return "\n\n".join(lines) if lines else "_No chunks retrieved_"


# ── Main handler ──────────────────────────────────────────────────────────────

def ask(question: str, max_attempts: int, expansion: str, lang: str):
    """Streaming generator — yields partial updates as graph nodes complete."""
    t = TRANSLATIONS[lang]
    if not question.strip():
        yield t["no_question"], "", "", "", "", "", "", ""
        return

    initial = {
        "question":        question,
        "max_attempts":    max_attempts,
        "current_query":   question,
        "expansion":       expansion,
        "language":        lang,
        "attempt":         0,
        "vector_docs":     [],
        "bm25_docs":       [],
        "fused_docs":      [],
        "graded_docs":     [],
        "reranked_docs":   [],
        "web_context":     "",
        "mcp_context":     "",
        "answer":          "",
        "hallucination_ok": True,
        "decision":        "",
        "reasoning":       "",
        "judge_scores":    {},
        "overall_score":   0.0,
        "strategies_used": [],
        "trace":           [],
    }

    accumulated = dict(initial)
    live_trace  = []

    for chunk in rag_graph.stream(initial):
        for _, node_output in chunk.items():
            accumulated.update(node_output)
            if "trace" in node_output:
                live_trace = node_output["trace"]
            yield t["processing"], "", "", _format_trace(live_trace, lang), "", "", "", ""

    answer_text = accumulated.get("answer", "—")
    trace   = _format_trace(accumulated.get("trace", []), lang)
    chunks  = _format_chunks(accumulated)
    summary = _format_summary(accumulated, max_attempts)

    # Stream answer; show placeholder scores while RAGAS computes
    placeholder_scores = _format_scores(
        {"faithfulness": None, "answer_relevancy": None}, 0.0)
    streamed = ""
    for char in answer_text:
        streamed += char
        yield streamed, placeholder_scores, summary, trace, chunks, "", "", ""

    # Run RAGAS after answer is shown
    docs = accumulated.get("reranked_docs") or accumulated.get("graded_docs") or []
    contexts = [d.page_content for d in docs[:3]] if docs else []
    mcp = accumulated.get("mcp_context", "")
    if mcp and "не найдено" not in mcp and mcp.strip():
        contexts.append(mcp[:1500])
    try:
        ragas_scores, ragas_overall = _run_ragas(question, answer_text, contexts)
        accumulated["judge_scores"]  = ragas_scores
        accumulated["overall_score"] = ragas_overall
    except Exception:
        pass

    scores  = _format_scores(accumulated.get("judge_scores", {}),
                              accumulated.get("overall_score", 0.0))
    summary = _format_summary(accumulated, max_attempts)

    log_trace(
        name="agentic-rag",
        input_data={"question": question, "expansion": expansion, "language": lang},
        output_data={"answer": answer_text, "decision": accumulated.get("decision", "")},
        metadata={
            "attempts":        accumulated.get("attempt", 0),
            "overall_score":   accumulated.get("overall_score", 0.0),
            "judge_scores":    accumulated.get("judge_scores", {}),
            "hallucination_ok": accumulated.get("hallucination_ok", True),
            "strategies_used": accumulated.get("strategies_used", []),
        },
    )
    yield answer_text, scores, summary, trace, chunks, question, answer_text, ""


# ── User feedback handler ─────────────────────────────────────────────────────

def log_feedback(rating: int, question: str, answer: str) -> str:
    """Log 👍/👎 feedback to LangFuse."""
    log_trace(
        name="user-feedback",
        input_data={"question": question},
        output_data={"answer": answer, "rating": rating},
        metadata={"feedback": "positive" if rating == 1 else "negative"},
    )
    return "👍 Thank you!" if rating == 1 else "👎 Feedback received"


# ── Document upload handler ───────────────────────────────────────────────────

def upload_document(file):
    if file is None:
        return "No file selected.", ""

    import shutil
    from agents.document_loader import load_document, list_docs

    docs_dir = Path(__file__).parent / "docs"
    docs_dir.mkdir(exist_ok=True)

    src = Path(file.name)
    dst = docs_dir / src.name
    shutil.copy(src, dst)

    result = load_document(src.name)

    if result["status"] == "ok":
        msg = (f"✅ **{src.name}** uploaded successfully!\n\n"
               f"- Pages/sections: {result['pages']}\n"
               f"- Chunks added: {result['chunks']}\n"
               f"- Total in DB: {result['total_in_db']}")
    else:
        msg = f"❌ Error: {result['message']}"

    docs = list_docs()
    doc_list = "\n".join(f"- {d}" for d in docs) if docs else "_No documents_"
    return msg, doc_list


def get_doc_list():
    try:
        from agents.document_loader import list_docs
        docs = list_docs()
        return "\n".join(f"- {d}" for d in docs) if docs else "_Нет документов_"
    except Exception:
        return "_Error reading docs/ folder_"


def get_chunk_count(lang: str = "ru"):
    try:
        count = vectorstore._collection.count()
        status = "✅" if count > 0 else "⚠️"
        t = TRANSLATIONS.get(lang, TRANSLATIONS["ru"])
        return (f"{t['sources_line']}\n\n"
                f"{t['pipeline_line']}\n\n"
                f"ChromaDB: **{count} chunks** {status} | [LangFuse](http://localhost:3000)")
    except Exception:
        return "ChromaDB: connection error"


def upload_document_and_refresh(file, lang="kz"):
    msg, doc_list = upload_document(file)
    return msg, doc_list, get_chunk_count(lang)


def delete_document(filename: str, lang: str = "kz"):
    import os
    import agents.bm25_index as _bm25
    from agents.document_loader import list_docs

    if not filename or not filename.strip():
        return "Enter a file name.", get_doc_list(), get_chunk_count(lang)

    filename = filename.strip()
    docs_dir = Path(__file__).parent / "docs"
    filepath = docs_dir / filename

    try:
        col = vectorstore._collection
        results = col.get(where={"source": filename}, include=[])
        ids = results.get("ids", [])
        if ids:
            vectorstore.delete(ids=ids)
            db_msg = f"removed {len(ids)} chunks from ChromaDB"
        else:
            db_msg = "no chunks found in ChromaDB"
    except Exception as e:
        db_msg = f"ChromaDB error: {e}"

    if filepath.exists():
        os.remove(filepath)
        file_msg = "file deleted"
    else:
        file_msg = "file not found in docs/ folder"

    _bm25._bm25_cache = None
    _bm25._docs_cache = None

    msg = f"**{filename}**: {file_msg}, {db_msg}"
    docs = list_docs()
    doc_list = "\n".join(f"- {d}" for d in docs) if docs else "_Нет документов_"
    return msg, doc_list, get_chunk_count(lang)


# ── Language switcher ─────────────────────────────────────────────────────────

def switch_language(lang: str):
    """Update all UI text when language changes."""
    t = TRANSLATIONS[lang]
    hard_updates = [gr.update(value=q) for q in t["hard_q"]]
    easy_updates = [gr.update(value=q) for q in t["easy_q"]]
    edge_updates = [gr.update(value=q) for q in t["edge_q"]]
    return (
        gr.update(value=t["title"]),                          # title_md
        gr.update(value=get_chunk_count(lang)),               # header_md
        gr.update(value=t["scenario2"]),                      # scenario2_md
        gr.update(value=t["scenario1"]),                      # scenario1_md
        gr.update(value=t["scenario3"]),                      # scenario3_md
        gr.update(label=t["q_label"], placeholder=t["q_placeholder"]),  # q_input
        gr.update(value=t["submit"]),                         # ask_btn
        *hard_updates,                                        # hard_btns[0..4]
        *easy_updates,                                        # easy_btns[0..4]
        *edge_updates,                                        # edge_btns[0..4]
        gr.update(value=t["upload_title"]),                   # upload_title_md
        gr.update(value=t["upload_section"]),                 # upload_section_md
        gr.update(label=t["file_label"]),                     # file_input
        gr.update(value=t["upload_btn"]),                     # upload_btn
        gr.update(value=t["delete_section"]),                 # delete_section_md
        gr.update(label=t["delete_label"], placeholder=t["delete_ph"]),  # delete_input
        gr.update(value=t["delete_btn"]),                     # delete_btn
        gr.update(value=t["docs_title"]),                     # docs_title_md
        gr.update(value=t["refresh_btn"]),                    # refresh_btn
        gr.update(label=t["lang_label"]),                     # lang_selector
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────

try:
    chunk_count = vectorstore._collection.count()
    print(f"ChromaDB: {chunk_count} chunks loaded")
except Exception:
    chunk_count = 0

_UPDATE_UPLOAD_TEXT_JS = """
(lang) => {
    window.__rag_lang = lang;
    const map = {
        'Drop File Here':        {kz:'Файлды осюда тастаңыз', en:'Drop File Here',        ru:'Перетащите файл сюда'},
        'Click to Upload':       {kz:'Жүктеу үшін басыңыз',  en:'Click to Upload',        ru:'Нажмите для загрузки'},
        '- or -':                {kz:'- немесе -',            en:'- or -',                 ru:'- или -'},
        'Перетащите файл сюда':  {kz:'Файлды осюда тастаңыз', en:'Drop File Here',        ru:'Перетащите файл сюда'},
        'Нажмите для загрузки':  {kz:'Жүктеу үшін басыңыз',  en:'Click to Upload',        ru:'Нажмите для загрузки'},
        '- или -':               {kz:'- немесе -',            en:'- or -',                 ru:'- или -'},
        'Файлды осюда тастаңыз': {kz:'Файлды осюда тастаңыз', en:'Drop File Here',        ru:'Перетащите файл сюда'},
        'Жүктеу үшін басыңыз':   {kz:'Жүктеу үшін басыңыз',  en:'Click to Upload',        ru:'Нажмите для загрузки'},
        '- немесе -':            {kz:'- немесе -',            en:'- or -',                 ru:'- или -'},
    };
    const apply = () => {
        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
        const updates = [];
        let node;
        while ((node = walker.nextNode())) {
            const t = node.textContent.trim();
            if (map[t]) updates.push({node, t});
        }
        updates.forEach(({node, t}) => { node.textContent = map[t][lang] || t; });
    };
    apply();
    setTimeout(apply, 300);
    return lang;
}
"""

_INIT_UPLOAD_TEXT_JS = """
() => {
    window.__rag_lang = window.__rag_lang || 'kz';
    const map = {
        'Drop File Here':        {kz:'Файлды осюда тастаңыз', en:'Drop File Here',        ru:'Перетащите файл сюда'},
        'Click to Upload':       {kz:'Жүктеу үшін басыңыз',  en:'Click to Upload',        ru:'Нажмите для загрузки'},
        '- or -':                {kz:'- немесе -',            en:'- or -',                 ru:'- или -'},
        'Перетащите файл сюда':  {kz:'Файлды осюда тастаңыз', en:'Drop File Here',        ru:'Перетащите файл сюда'},
        'Нажмите для загрузки':  {kz:'Жүктеу үшін басыңыз',  en:'Click to Upload',        ru:'Нажмите для загрузки'},
        '- или -':               {kz:'- немесе -',            en:'- or -',                 ru:'- или -'},
        'Файлды осюда тастаңыз': {kz:'Файлды осюда тастаңыз', en:'Drop File Here',        ru:'Перетащите файл сюда'},
        'Жүктеу үшін басыңыз':   {kz:'Жүктеу үшін басыңыз',  en:'Click to Upload',        ru:'Нажмите для загрузки'},
        '- немесе -':            {kz:'- немесе -',            en:'- or -',                 ru:'- или -'},
    };
    const applyTranslation = () => {
        const lang = window.__rag_lang || 'kz';
        const upload = document.querySelector('.upload-container, .file-preview-holder, [data-testid="file"], .svelte-upload');
        const root = upload ? upload.parentElement : document.body;
        const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
        const updates = [];
        let node;
        while ((node = walker.nextNode())) {
            const t = node.textContent.trim();
            if (map[t] && map[t][lang] !== t) updates.push({node, t});
        }
        updates.forEach(({node, t}) => { node.textContent = map[t][lang]; });
    };
    setInterval(applyTranslation, 2000);
    setTimeout(applyTranslation, 800);
}
"""

_TAB_SELECT_JS = """() => {}"""

CSS = """
.answer-box {
    font-size: 15px;
    border-left: 4px solid #4CAF50;
    padding: 12px 16px;
    background: #f9f9f9;
    border-radius: 4px;
}
.score-box { font-size: 13px; font-family: monospace; }
.trace-box { font-size: 12px; }
.summary-box { font-size: 13px; }
.lang-selector { max-width: 320px; }
"""

_t = TRANSLATIONS["kz"]

with gr.Blocks(title="Legal Assistant for SMB Kazakhstan", css=CSS, theme=gr.themes.Soft()) as demo:

    # ── Language selector ─────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=4):
            title_md = gr.Markdown(value=_t["title"])
        with gr.Column(scale=1, min_width=280):
            lang_selector = gr.Radio(
                choices=[("🇰🇿 Қазақша", "kz"), ("🇬🇧 English", "en"), ("🇷🇺 Русский", "ru")],
                value="kz",
                label=_t["lang_label"],
                elem_classes=["lang-selector"],
            )

    lang_state  = gr.State("kz")
    last_q_state = gr.State("")
    last_a_state = gr.State("")
    header_md   = gr.Markdown(value=get_chunk_count("kz"))

    with gr.Tabs() as tabs:

        # ══════════════════════════════════════
        #  TAB 1: Вопрос-Ответ
        # ══════════════════════════════════════
        with gr.Tab("🗣️ Сұрақ / Ask / Задать вопрос"):

            with gr.Row():
                with gr.Column(scale=3):
                    q_input = gr.Textbox(
                        label=_t["q_label"],
                        placeholder=_t["q_placeholder"],
                        lines=3,
                    )
                    with gr.Row():
                        with gr.Column():
                            scenario2_md = gr.Markdown(_t["scenario2"])
                            hard_btns = [
                                gr.Button(q, size="sm", variant="secondary")
                                for q in _t["hard_q"]
                            ]
                        with gr.Column():
                            scenario1_md = gr.Markdown(_t["scenario1"])
                            easy_btns = [
                                gr.Button(q, size="sm", variant="secondary")
                                for q in _t["easy_q"]
                            ]
                        with gr.Column():
                            scenario3_md = gr.Markdown(_t["scenario3"])
                            edge_btns = [
                                gr.Button(q, size="sm", variant="secondary")
                                for q in _t["edge_q"]
                            ]

                with gr.Column(scale=1):
                    max_att   = gr.Slider(1, 5, value=3, step=1, label="Max Attempts")
                    expansion = gr.Radio(
                        choices=["none", "hyde", "step_back", "keyword"],
                        value="none",
                        label="Query Expansion",
                    )

            ask_btn = gr.Button(_t["submit"], variant="primary", size="lg")

            with gr.Row():
                with gr.Column(scale=2):
                    answer_out = gr.Markdown(label="Answer", elem_classes=["answer-box"])
                    with gr.Row():
                        thumbs_up   = gr.Button("👍", size="sm", variant="secondary")
                        thumbs_down = gr.Button("👎", size="sm", variant="secondary")
                    feedback_msg = gr.Markdown("")
                with gr.Column(scale=1):
                    scores_out  = gr.Markdown(label="Judge Scores", elem_classes=["score-box"])
                    summary_out = gr.Markdown(label="Summary",      elem_classes=["summary-box"])

            with gr.Accordion("Agent Trace (LangGraph)", open=True):
                trace_out = gr.Markdown(elem_classes=["trace-box"])

            with gr.Accordion("Retrieved & Re-ranked Chunks + MCP", open=False):
                chunks_out = gr.Markdown()

            ask_btn.click(
                fn=ask,
                inputs=[q_input, max_att, expansion, lang_state],
                outputs=[answer_out, scores_out, summary_out, trace_out, chunks_out,
                         last_q_state, last_a_state, feedback_msg],
            )
            q_input.submit(
                fn=ask,
                inputs=[q_input, max_att, expansion, lang_state],
                outputs=[answer_out, scores_out, summary_out, trace_out, chunks_out,
                         last_q_state, last_a_state, feedback_msg],
            )
            thumbs_up.click(
                fn=lambda q, a: log_feedback(1, q, a),
                inputs=[last_q_state, last_a_state],
                outputs=[feedback_msg],
            )
            thumbs_down.click(
                fn=lambda q, a: log_feedback(0, q, a),
                inputs=[last_q_state, last_a_state],
                outputs=[feedback_msg],
            )

            # Example question buttons fill q_input on click (language-aware)
            for i, btn in enumerate(hard_btns):
                btn.click(
                    fn=lambda lang, idx=i: TRANSLATIONS[lang]["hard_q"][idx],
                    inputs=[lang_state], outputs=[q_input],
                )
            for i, btn in enumerate(easy_btns):
                btn.click(
                    fn=lambda lang, idx=i: TRANSLATIONS[lang]["easy_q"][idx],
                    inputs=[lang_state], outputs=[q_input],
                )
            for i, btn in enumerate(edge_btns):
                btn.click(
                    fn=lambda lang, idx=i: TRANSLATIONS[lang]["edge_q"][idx],
                    inputs=[lang_state], outputs=[q_input],
                )

        # ══════════════════════════════════════
        #  TAB 2: Загрузка документов
        # ══════════════════════════════════════
        with gr.Tab("📁 Жүктеу / Upload / Загрузить"):

            upload_title_md = gr.Markdown(_t["upload_title"])

            with gr.Row():
                with gr.Column(scale=2):
                    upload_section_md = gr.Markdown(_t["upload_section"])
                    file_input  = gr.File(
                        label=_t["file_label"],
                        file_types=[".pdf", ".docx", ".doc"],
                    )
                    upload_btn  = gr.Button(_t["upload_btn"], variant="primary")
                    upload_msg  = gr.Markdown(label="Status")

                    delete_section_md = gr.Markdown(_t["delete_section"])
                    delete_input = gr.Textbox(
                        label=_t["delete_label"],
                        placeholder=_t["delete_ph"],
                    )
                    delete_btn  = gr.Button(_t["delete_btn"], variant="secondary")
                    delete_msg  = gr.Markdown()

                with gr.Column(scale=1):
                    docs_title_md = gr.Markdown(_t["docs_title"])
                    doc_list_out  = gr.Markdown(value=get_doc_list())
                    refresh_btn   = gr.Button(_t["refresh_btn"], size="sm")

            upload_btn.click(
                fn=upload_document_and_refresh,
                inputs=[file_input, lang_state],
                outputs=[upload_msg, doc_list_out, header_md],
            )
            delete_btn.click(
                fn=lambda name, lang: delete_document(name, lang),
                inputs=[delete_input, lang_state],
                outputs=[delete_msg, doc_list_out, header_md],
            )
            refresh_btn.click(
                fn=lambda lang: (get_doc_list(), get_chunk_count(lang)),
                inputs=[lang_state],
                outputs=[doc_list_out, header_md],
            )

    # ── Language change handler ───────────────────────────────────────────────
    lang_selector.change(
        fn=lambda lang: lang,
        inputs=[lang_selector],
        outputs=[lang_state],
    )
    lang_selector.change(
        fn=switch_language,
        inputs=[lang_selector],
        outputs=[
            title_md, header_md, scenario2_md, scenario1_md, scenario3_md,
            q_input, ask_btn,
            *hard_btns, *easy_btns, *edge_btns,
            upload_title_md, upload_section_md,
            file_input, upload_btn,
            delete_section_md, delete_input, delete_btn,
            docs_title_md, refresh_btn,
            lang_selector,
        ],
    )
    lang_selector.change(
        fn=None,
        inputs=[lang_selector],
        js=_UPDATE_UPLOAD_TEXT_JS,
    )
    tabs.select(
        fn=None,
        inputs=[],
        js=_TAB_SELECT_JS,
    )

    demo.load(
        fn=lambda: (get_chunk_count("kz"), get_doc_list()),
        inputs=[],
        outputs=[header_md, doc_list_out],
    )
    demo.load(fn=None, js=_INIT_UPLOAD_TEXT_JS)

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7861)
