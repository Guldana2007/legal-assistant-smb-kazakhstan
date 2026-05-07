"""
Architecture Diagram — v4 (clean 3-row layout for defense)
Run: python generate_diagram.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(28, 15))
ax.set_xlim(0, 28); ax.set_ylim(0, 15); ax.axis("off")
fig.patch.set_facecolor("#FAFAFA")

# ── Color palette ──────────────────────────────────────────────────────────────
C_AGENT = "#D6E4FF"; B_AGENT = "#4A90D9"   # Blue   — AI Agents
C_TOOL  = "#D4EDDA"; B_TOOL  = "#28A745"   # Green  — Search tools
C_MCP   = "#FFF3CD"; B_MCP   = "#E6A817"   # Yellow — MCP / external
C_DB    = "#E8D5F5"; B_DB    = "#9B59B6"   # Purple — Knowledge base
C_USER  = "#F0F0F0"; B_USER  = "#6C757D"   # Gray   — User
C_ANS   = "#C8F7C5"; B_ANS   = "#1E8449"   # Green  — Answer
C_OBS   = "#FFE4E4"; B_OBS   = "#C0392B"   # Red    — Observability / Retry


# ── Helpers ────────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, lines, fc, ec, fs=8.5):
    """Draw a rounded box centred at (x, y)."""
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.12", facecolor=fc, edgecolor=ec,
        linewidth=2.0, zorder=3,
    )
    ax.add_patch(rect)
    if isinstance(lines, str):
        lines = [lines]
    n = len(lines)
    for i, line in enumerate(lines):
        ax.text(x, y + (n - 1) * 0.23 - i * 0.42, line,
                ha="center", va="center", fontsize=fs,
                fontweight="bold" if i == 0 else "normal", zorder=4)


def region(ax, x0, y0, w, h, label, fc, ec):
    """Draw a dashed region with a label above it."""
    rect = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.15", facecolor=fc, edgecolor=ec,
        linewidth=1.5, linestyle="--", zorder=1, alpha=0.30,
    )
    ax.add_patch(rect)
    ax.text(x0 + w / 2, y0 + h + 0.13, label,
            ha="center", fontsize=9, fontstyle="italic",
            color=ec, fontweight="bold", zorder=5)


def arrow(ax, x1, y1, x2, y2, color, label="", rad=0.0):
    """Draw an arrow with optional mid-point label."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=1.8,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=2,
    )
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        ax.text(mx, my + 0.28, label, fontsize=7.5, color=color,
                ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.90, pad=2.0),
                zorder=6)


# ══════════════════════════════════════════════════════════════════════════════
#  TITLE
# ══════════════════════════════════════════════════════════════════════════════
ax.text(14, 14.65,
        "Юридический ассистент для МСБ Казахстана  —  System Architecture",
        ha="center", va="center", fontsize=13,
        fontweight="bold", color="#1A1A2E")
ax.text(14, 14.30,
        "Agentic RAG Pipeline  ·  LangGraph  ·  Hybrid Search  ·  Self-Reflection  ·  RAGAS",
        ha="center", va="center", fontsize=9,
        fontstyle="italic", color="#4A90D9")

# ══════════════════════════════════════════════════════════════════════════════
#  ROW 1  ─  MAIN AGENTS  (y = 12.3)
# ══════════════════════════════════════════════════════════════════════════════
Y1 = 12.3

# Dashed border around the 6 agents
region(ax, 3.5, 10.9, 21.0, 2.8, "Agentic RAG  —  AI Агенты  (LangGraph StateGraph)", "#EBF3FF", B_AGENT)

box(ax,  1.5, Y1, 2.2, 1.0,  ["Пользователь", "(вопрос)"],                         C_USER,  B_USER)
box(ax,  5.5, Y1, 2.8, 1.5,  ["1. Query Rewrite", "none | HyDE", "step_back | keyword"], C_AGENT, B_AGENT)
box(ax,  9.5, Y1, 2.8, 1.5,  ["3. Doc Grader", "LLM: строгая оценка", "релевантности"], C_AGENT, B_AGENT)
box(ax, 17.5, Y1, 2.8, 1.5,  ["4. LLM Generate", "gpt-4.1-mini", "RAG + MCP контекст"], C_AGENT, B_AGENT)
box(ax, 21.5, Y1, 2.8, 1.5,  ["5. Hallucination", "Check", "Grounded: true / false"],    C_AGENT, B_AGENT)
box(ax, 25.0, Y1, 2.6, 1.5,  ["6. Reflect", "RAGAS judge", "accept / retry"],            C_AGENT, B_AGENT)

# Answer — right of Reflect
box(ax, 27.4, Y1, 1.4, 1.2,  ["ОТВЕТ", "Streaming"],                                    C_ANS,   B_ANS)

# ══════════════════════════════════════════════════════════════════════════════
#  ROW 2  ─  RETRIEVAL TOOLS + CROSS-ENCODER + MCP  (y = 8.6)
# ══════════════════════════════════════════════════════════════════════════════
Y2 = 8.6

# Retrieval subgraph
region(ax, 3.5, 7.3, 9.5, 2.6, "Retrieval Tools", "#EBF9EE", B_TOOL)
box(ax,  5.0, Y2, 2.5, 1.2,  ["2a. Vector DB", "ChromaDB", "text-embedding-3-small"],   C_TOOL, B_TOOL)
box(ax,  8.2, Y2, 2.5, 1.2,  ["2b. BM25 Index", "BM25Okapi", "14 585 chunks"],          C_TOOL, B_TOOL)
box(ax, 11.5, Y2, 2.2, 1.2,  ["2c. RRF Fusion", "Объединение", "рейтингов"],            C_TOOL, B_TOOL)

# Cross-Encoder — between retrieval and Generate
box(ax, 14.8, Y2, 2.8, 1.2,  ["Cross-Encoder", "Re-ranking", "ms-marco"],               C_TOOL, B_TOOL)

# MCP subgraph
region(ax, 18.5, 7.3, 8.0, 2.6, "MCP  —  Внешний источник", "#FFFBEA", B_MCP)
box(ax, 22.5, Y2, 6.5, 1.4,  ["MCP Legal Search",
                                "adilet.zan.kz · kgd.gov.kz · egov.kz",
                                "(DuckDuckGo site filter)"],                              C_MCP, B_MCP)

# ══════════════════════════════════════════════════════════════════════════════
#  ROW 3  ─  RETRY LOOP  (y = 6.0)
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 25.0, 6.0, 2.6, 1.0,  ["Reformulate", "(retry_retrieval)"],                      C_AGENT, B_AGENT)

# ══════════════════════════════════════════════════════════════════════════════
#  LANGFUSE  (right column)
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 27.2, 9.8, 1.8, 2.6,  ["LangFuse", "Observability", "Trace", "Score"],           C_OBS, B_OBS, fs=7.5)

# ══════════════════════════════════════════════════════════════════════════════
#  KNOWLEDGE BASE  (bottom left)
# ══════════════════════════════════════════════════════════════════════════════
region(ax, 0.3, 1.5, 6.8, 5.0, "Локальная база знаний  (ChromaDB)", "#F3E8FF", B_DB)
box(ax, 1.8, 5.7, 2.2, 1.1,  ["Налоговый", "кодекс РК"],            C_DB, B_DB)
box(ax, 4.5, 5.7, 2.2, 1.1,  ["Трудовой", "кодекс РК"],             C_DB, B_DB)
box(ax, 1.8, 4.2, 2.2, 1.1,  ["Гражданский", "кодекс РК"],          C_DB, B_DB)
box(ax, 4.5, 4.2, 2.2, 1.1,  ["Предпринима-", "тельский кодекс"],   C_DB, B_DB)
box(ax, 3.1, 2.7, 3.5, 1.0,  ["PDF Parser", "pypdf / docling", "chunk_size=500"],        C_DB, B_DB, fs=7.5)

# ══════════════════════════════════════════════════════════════════════════════
#  ARROWS
# ══════════════════════════════════════════════════════════════════════════════

# ── Row 1: main pipeline ──────────────────────────────────────────────────────
arrow(ax,  2.6, Y1,   4.1, Y1,   B_AGENT, "вопрос")
arrow(ax,  6.9, Y1,   8.1, Y1,   B_AGENT)
# Doc Grader → Generate (через CrossEncoder — нет прямой стрелки, будет через Row 2)
arrow(ax, 18.9, Y1,  20.1, Y1,   B_AGENT)   # Generate → Hallu
arrow(ax, 22.9, Y1,  23.7, Y1,   B_AGENT)   # Hallu → Reflect
arrow(ax, 26.3, Y1,  26.7, Y1,   B_ANS, "accept")   # Reflect → ОТВЕТ

# ── Row 1 → Row 2: Hybrid Search (параллельный поиск) ────────────────────────
arrow(ax,  5.5, Y1 - 0.75,  5.0, Y2 + 0.6,  B_TOOL)          # QR → VecDB
arrow(ax,  5.5, Y1 - 0.75,  8.2, Y2 + 0.6,  B_TOOL)          # QR → BM25 (параллельно)
# VecDB и BM25 → RRF (оба сходятся в RRF — Hybrid Search)
arrow(ax,  5.0, Y2 - 0.6,  11.5, Y2 - 0.6,  B_TOOL)          # VecDB → RRF (снизу)
arrow(ax,  8.2, Y2 - 0.6,  11.5, Y2 - 0.6,  B_TOOL)          # BM25 → RRF (снизу)
ax.plot(11.5, Y2 - 0.6, 'o', color=B_TOOL, markersize=5, zorder=5)  # merge dot
# Hybrid Search badge
from matplotlib.patches import FancyBboxPatch as FBP
badge = FBP((4.2, Y2 - 1.05), 5.5, 0.38,
            boxstyle="round,pad=0.08", facecolor="#C8E6C9", edgecolor=B_TOOL,
            linewidth=1.5, zorder=4)
ax.add_patch(badge)
ax.text(6.95, Y2 - 0.86, "Hybrid Search  (Semantic + BM25)",
        ha="center", va="center", fontsize=7.5, color="#1B5E20",
        fontweight="bold", zorder=5)
arrow(ax, 11.5,  Y2 + 0.6, 9.5,  Y1 - 0.75, B_TOOL)          # RRF → DocGrader

# ── Doc Grader branching ───────────────────────────────────────────────────────
# ≥2 docs → Cross-Encoder (diagonal down-right)
arrow(ax,  9.5, Y1 - 0.75, 14.8, Y2 + 0.6,  B_TOOL, "≥2 docs",  0.0)
# <2 docs → MCP (diagonal down-right far)
arrow(ax, 10.9, Y1,        19.5, Y2 + 0.7,  B_MCP,  "<2 docs",  0.1)
# MCP → Cross-Encoder (horizontal left in row 2)
arrow(ax, 19.5, Y2,        16.2, Y2,         B_MCP)
# Cross-Encoder → Generate (diagonal up-right)
arrow(ax, 14.8, Y2 + 0.6,  16.1, Y1 - 0.75, B_AGENT)

# ── Retry loops ────────────────────────────────────────────────────────────────
# Reflect → Reformulate (vertical down)
arrow(ax, 25.0, Y1 - 0.75, 25.0, 6.5,       B_OBS)
# Reformulate → QR (long arc back under diagram)
arrow(ax, 23.7, 6.0,        5.5, Y1 - 0.75, B_OBS, "", -0.18)
# Reflect → Generate (retry_generation, arc above row 1)
arrow(ax, 23.7, Y1,         18.9, Y1,        B_OBS, "retry generation", -0.25)

# ── LangFuse ──────────────────────────────────────────────────────────────────
arrow(ax, 26.3, Y1 - 0.3,  27.2, 10.8,      B_OBS, "trace")

# ── KB → VecDB ────────────────────────────────────────────────────────────────
arrow(ax,  4.0,  6.15,  4.5, Y2 + 0.6,      B_DB, "индекс")

# ══════════════════════════════════════════════════════════════════════════════
#  LEGEND  ─  horizontal at bottom
# ══════════════════════════════════════════════════════════════════════════════
legend_items = [
    (C_AGENT, B_AGENT, "AI Агент (LangGraph node)"),
    (C_TOOL,  B_TOOL,  "Инструмент поиска"),
    (C_MCP,   B_MCP,   "MCP / Внешний источник"),
    (C_DB,    B_DB,    "Локальная база знаний"),
    (C_OBS,   B_OBS,   "Observability / Retry"),
    (C_ANS,   B_ANS,   "Ответ пользователю"),
]
lx, ly, step = 1.0, 1.05, 4.3
for i, (fc, ec, label) in enumerate(legend_items):
    xi = lx + i * step
    rect = FancyBboxPatch((xi, ly - 0.18), 0.55, 0.40,
                          boxstyle="round,pad=0.05", facecolor=fc,
                          edgecolor=ec, linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(xi + 0.72, ly + 0.01, label, fontsize=8.5, va="center")

# Arrow legend
ax_x = lx + len(legend_items) * step
for dy, color, txt in [(0.05, "#444", "Основной поток"),
                       (-0.35, B_OBS, "Retry / переформулировка")]:
    ax.annotate("", xy=(ax_x + 0.6, ly + dy), xytext=(ax_x, ly + dy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))
    ax.text(ax_x + 0.75, ly + dy, txt, fontsize=8.5, va="center")

# ── Tech stack ─────────────────────────────────────────────────────────────────
ax.text(14, 0.45,
        "Tech Stack:  LangGraph · ChromaDB · OpenAI gpt-4.1-mini · "
        "text-embedding-3-small · BM25 · Cross-Encoder (ms-marco) · "
        "RAGAS · MCP · LangFuse · Gradio · Python 3.11",
        ha="center", fontsize=8, color="#555", style="italic")

plt.tight_layout(pad=0.3)
plt.savefig("architecture_diagram.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved: architecture_diagram.png")
