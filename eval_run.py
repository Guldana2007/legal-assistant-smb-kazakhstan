"""
Automated Evaluation Script — Legal RAG Assistant
===================================================
Runs predefined test questions through the RAG pipeline,
collects RAGAS scores, and prints a summary table.

Usage:
    python eval_run.py

Output:
    eval_results/eval_YYYY-MM-DD_HH-MM.txt  — human-readable report
    eval_results/eval_YYYY-MM-DD_HH-MM.json — raw data for further analysis
"""

import os
import sys
import time
import json
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from langgraph_rag import run_graph, _run_ragas

# ── Test questions (English) ──────────────────────────────────────────────────
# Three scenarios cover all pipeline paths:
#   Local RAG  → answer from local law database (ChromaDB + BM25)
#   MCP        → answer from official government portals (adilet, kgd, egov)
#   Hard       → current data not present in the static knowledge base

TEST_CASES = [
    # Scenario 1 — Local RAG (questions from UI easy_q)
    {
        "scenario": "Local RAG",
        "question": "What is the statute of limitations under the Civil Code of Kazakhstan?",
        "expected_source": "local",
    },
    {
        "scenario": "Local RAG",
        "question": "What is the employer's liability for delayed salary payment under the Labor Code of Kazakhstan?",
        "expected_source": "local",
    },
    {
        "scenario": "Local RAG",
        "question": "How many vacation days are employees entitled to in Kazakhstan?",
        "expected_source": "local",
    },
    {
        "scenario": "Local RAG",
        "question": "What are the grounds for terminating an employment contract at the employer's initiative under the Labor Code of Kazakhstan?",
        "expected_source": "local",
    },
    {
        "scenario": "Local RAG",
        "question": "What are the grounds for declaring a transaction void under the Civil Code of Kazakhstan?",
        "expected_source": "local",
    },

    # Scenario 2 — RAG not found → MCP (questions from UI hard_q)
    {
        "scenario": "MCP",
        "question": "What are the penalties for late tax filing for SMEs in Kazakhstan?",
        "expected_source": "mcp",
    },
    {
        "scenario": "MCP",
        "question": "What are the sanitary requirements for opening a food service establishment in Kazakhstan?",
        "expected_source": "mcp",
    },
    {
        "scenario": "MCP",
        "question": "How do I check the status of my business registration application on egov.kz?",
        "expected_source": "mcp",
    },
    {
        "scenario": "MCP",
        "question": "How do I obtain an electronic digital signature (EDS) for business use in Kazakhstan?",
        "expected_source": "mcp",
    },
    {
        "scenario": "MCP",
        "question": "What are the requirements for opening a pharmacy in Kazakhstan?",
        "expected_source": "mcp",
    },

    # Scenario 3 — Edge Cases (questions from UI edge_q)
    {
        "scenario": "Edge Case",
        "question": "If a customer slips and falls in my shop, am I liable?",
        "expected_source": "none",
    },
    {
        "scenario": "Edge Case",
        "question": "If I want to close my business temporarily for vacation, do I need to notify authorities?",
        "expected_source": "local",
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_actual_source(state: dict) -> str:
    """Определяет какой источник реально использовался для ответа."""
    mcp  = state.get("mcp_context", "")
    docs = state.get("reranked_docs") or state.get("graded_docs") or []
    web  = state.get("web_context", "")

    if docs:
        return "local"
    if mcp and "не найдено" not in mcp and "error" not in mcp.lower():
        return "mcp"
    if web and web not in ("", "Web search disabled"):
        return "web"
    return "none"


def _short(text: str, n: int = 80) -> str:
    """Обрезает текст до n символов для таблицы."""
    text = text.replace("\n", " ")
    return text[:n] + "…" if len(text) > n else text


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_eval():
    results = []

    print("\n" + "=" * 90)
    print("  LEGAL RAG ASSISTANT — AUTOMATED EVALUATION")
    print("=" * 90)
    print(f"  Questions: {len(TEST_CASES)} | Max attempts: 3 | Model: gpt-4.1-mini")
    print("=" * 90 + "\n")

    for i, tc in enumerate(TEST_CASES, 1):
        q = tc["question"]
        print(f"[{i}/{len(TEST_CASES)}] {tc['scenario']} — {q[:60]}…")

        t0 = time.time()
        try:
            # Запускаем полный RAG pipeline для одного вопроса
            state = run_graph(q, max_attempts=3, language="en")
            elapsed = round(time.time() - t0, 1)

            score    = state.get("overall_score", 0.0)
            scores   = state.get("judge_scores", {})
            decision = state.get("decision", "?")
            attempts = state.get("attempt", 0)
            answer   = state.get("answer", "")
            actual   = _detect_actual_source(state)

            # Fast-path skips RAGAS (score=0.0) — run it explicitly here
            if score == 0.0 and answer:
                docs = state.get("reranked_docs") or state.get("graded_docs") or []
                ctxs = [d.page_content for d in docs[:3]]
                mcp  = state.get("mcp_context", "")
                if mcp and "не найдено" not in mcp:
                    ctxs.append(mcp[:1500])
                scores, score = _run_ragas(q, answer, ctxs)
            # ✅ источник совпал с ожидаемым, ⚠️ — нет
            match    = "✅" if actual == tc["expected_source"] else "⚠️"

            results.append({
                "n":        i,
                "scenario": tc["scenario"],
                "question": q,
                "score":    score,
                "faith":    scores.get("faithfulness", 0),      # насколько ответ основан на контексте
                "relev":    scores.get("answer_relevancy", 0),  # насколько ответ релевантен вопросу
                "decision": decision,
                "attempts": attempts,
                "source":   actual,
                "match":    match,
                "elapsed":  elapsed,
                "answer":   answer,
            })
            print(f"     Score: {score}/10 | Source: {actual} {match} | "
                  f"Attempts: {attempts} | {elapsed}s\n")

        except Exception as e:
            elapsed = round(time.time() - t0, 1)
            print(f"     ERROR: {e}\n")
            results.append({
                "n": i, "scenario": tc["scenario"], "question": q,
                "score": 0, "faith": 0, "relev": 0,
                "decision": "error", "attempts": 0,
                "source": "error", "match": "❌",
                "elapsed": elapsed, "answer": str(e),
            })

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  RESULTS SUMMARY")
    print("=" * 90)
    header = f"{'#':>2}  {'Scenario':<12} {'Score':>6} {'Faith':>6} {'Relev':>6}  {'Source':<8} {'M':>2}  {'Att':>3}  {'Time':>5}  Question"
    print(header)
    print("-" * 90)

    for r in results:
        print(
            f"{r['n']:>2}  {r['scenario']:<12} {r['score']:>5.1f}  "
            f"{r['faith']:>5.1f}  {r['relev']:>5.1f}  "
            f"{r['source']:<8} {r['match']:>2}  {r['attempts']:>3}  "
            f"{r['elapsed']:>4.1f}s  {_short(r['question'], 45)}"
        )

    print("-" * 90)

    # Считаем средние только по вопросам без ошибок
    valid = [r for r in results if r["score"] > 0]
    if valid:
        avg_score = sum(r["score"] for r in valid) / len(valid)
        avg_faith = sum(r["faith"] for r in valid) / len(valid)
        avg_relev = sum(r["relev"] for r in valid) / len(valid)
        matched   = sum(1 for r in results if r["match"] == "✅")
        total_t   = sum(r["elapsed"] for r in results)

        print(f"{'AVG':>2}  {'':12} {avg_score:>5.1f}  {avg_faith:>5.1f}  {avg_relev:>5.1f}")
        print(f"\n  Source match: {matched}/{len(results)} correct")
        print(f"  Total time:   {total_t:.1f}s")
        print(f"  Pass (≥7.0):  {sum(1 for r in valid if r['score'] >= 7.0)}/{len(valid)} questions")

    print("=" * 90 + "\n")

    # ── Answers detail ────────────────────────────────────────────────────────
    print("ANSWERS DETAIL\n")
    for r in results:
        print(f"[{r['n']}] {r['question']}")
        print(f"    {_short(r['answer'], 120)}")
        print()

    # ── Save results to file ──────────────────────────────────────────────────
    _save_results(results)


def _save_results(results: list):
    """Сохраняет результаты в папку eval_results/ в двух форматах."""
    os.makedirs("eval_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # JSON — для программного анализа или загрузки в таблицы
    json_path = f"eval_results/eval_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # TXT — читабельный отчёт для презентации / скриншота
    txt_path = f"eval_results/eval_{timestamp}.txt"
    valid     = [r for r in results if r["score"] > 0]
    avg_score = sum(r["score"] for r in valid) / len(valid) if valid else 0
    avg_faith = sum(r["faith"] for r in valid) / len(valid) if valid else 0
    avg_relev = sum(r["relev"] for r in valid) / len(valid) if valid else 0
    matched   = sum(1 for r in results if r["match"] == "✅")
    total_t   = sum(r["elapsed"] for r in results)
    passed    = sum(1 for r in valid if r["score"] >= 7.0)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("LEGAL RAG ASSISTANT — EVALUATION REPORT\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Model: gpt-4.1-mini | Questions: {len(results)}\n")
        f.write("=" * 90 + "\n\n")

        # Таблица результатов
        f.write(f"{'#':>2}  {'Scenario':<12} {'Score':>6} {'Faith':>6} {'Relev':>6}  {'Source':<8} {'M':>2}  {'Att':>3}  {'Time':>5}  Question\n")
        f.write("-" * 90 + "\n")
        for r in results:
            f.write(
                f"{r['n']:>2}  {r['scenario']:<12} {r['score']:>5.1f}  "
                f"{r['faith']:>5.1f}  {r['relev']:>5.1f}  "
                f"{r['source']:<8} {r['match']:>2}  {r['attempts']:>3}  "
                f"{r['elapsed']:>4.1f}s  {r['question'][:45]}\n"
            )
        f.write("-" * 90 + "\n")
        f.write(f"{'AVG':>2}  {'':12} {avg_score:>5.1f}  {avg_faith:>5.1f}  {avg_relev:>5.1f}\n\n")
        f.write(f"Source match: {matched}/{len(results)} correct\n")
        f.write(f"Total time:   {total_t:.1f}s\n")
        f.write(f"Pass (>=7.0): {passed}/{len(valid)} questions\n\n")

        # Полные ответы для проверки качества
        f.write("=" * 90 + "\n")
        f.write("ANSWERS DETAIL\n\n")
        for r in results:
            f.write(f"[{r['n']}] {r['question']}\n")
            f.write(f"Score: {r['score']}/10 | Faith: {r['faith']} | Relev: {r['relev']} | Source: {r['source']} {r['match']}\n")
            f.write(f"{r['answer']}\n\n")
            f.write("-" * 60 + "\n\n")

    print(f"Results saved to:\n  {txt_path}\n  {json_path}\n")


if __name__ == "__main__":
    run_eval()
