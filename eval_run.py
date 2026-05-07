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

from langgraph_rag import run_graph

# ── Test questions by scenario ────────────────────────────────────────────────
# Три сценария покрывают все пути в графе:
#   Local RAG  → ответ из локальной базы законов РК (ChromaDB + BM25)
#   MCP        → ответ из официальных сайтов через MCP (adilet, kgd, egov)
#   Hard       → актуальные данные, которых нет в локальной базе

TEST_CASES = [
    # Scenario 1: вопросы из локальной базы законов
    {
        "scenario": "Local RAG",
        "question": "Какой срок исковой давности по Гражданскому кодексу РК?",
        "expected_source": "local",  # ожидаем ответ из ChromaDB
    },
    {
        "scenario": "Local RAG",
        "question": "Сколько дней отпуска положено работнику в Казахстане?",
        "expected_source": "local",
    },
    {
        "scenario": "Local RAG",
        "question": "Что такое НДС по Налоговому кодексу РК?",
        "expected_source": "local",
    },

    # Scenario 2: вопросы которых нет в локальной базе → MCP
    {
        "scenario": "MCP",
        "question": "Какие налоговые ставки КПН для МСБ в Казахстане?",
        "expected_source": "mcp",   # ожидаем ответ из kgd.gov.kz или adilet
    },
    {
        "scenario": "MCP",
        "question": "Какие документы нужны для открытия ТОО в Казахстане?",
        "expected_source": "mcp",
    },
    {
        "scenario": "MCP",
        "question": "Как зарегистрироваться в качестве ИП в Казахстане?",
        "expected_source": "mcp",
    },

    # Scenario 3: актуальные данные 2025-2026 → MCP или веб
    {
        "scenario": "Hard/Recent",
        "question": "Какой минимальный размер заработной платы в Казахстане в 2026 году?",
        "expected_source": "mcp",   # МЗП меняется ежегодно — нужны свежие данные
    },
    {
        "scenario": "Hard/Recent",
        "question": "Какие изменения в Трудовой кодекс РК внесены в 2025 году?",
        "expected_source": "mcp",
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
            state = run_graph(q, max_attempts=3)
            elapsed = round(time.time() - t0, 1)

            score    = state.get("overall_score", 0.0)
            scores   = state.get("judge_scores", {})
            decision = state.get("decision", "?")
            attempts = state.get("attempt", 0)
            answer   = state.get("answer", "")
            actual   = _detect_actual_source(state)
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
