"""
MCP Server: Kazakhstan Legal Database
=======================================
Searches for fresh legal information from official Kazakhstan government sources via DuckDuckGo.
Sources: adilet.zan.kz, minfin.gov.kz, kgd.gov.kz, egov.kz

Run standalone:
    python mcp_server/legal_kz_server.py

Tools exposed:
    - search_kazakhstan_law(query, max_results) → fresh results from official KZ sources
"""

import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

app = Server("legal-kz")


OFFICIAL_SITES = [
    "adilet.zan.kz",   # Законы, кодексы, постановления
    "kgd.gov.kz",      # Налоговая служба, сборы, льготы для бизнеса
    "egov.kz",         # Госуслуги, регистрация бизнеса, лицензии
]

BLOCKED_DOMAINS = [
    "google.kz", "translate.google.kz",
    "yandex.kz", "youtube.com", "facebook.com",
    "wikipedia.org", "wikimedia.org",
    "sports.kz", "yadro.kz",
]


def _ddg_search_adilet(query: str, max_results: int = 3, language: str = "ru") -> str:
    """Search official Kazakhstan government sources via DuckDuckGo (single request, domain filter)."""
    try:
        from ddgs import DDGS
        # One request — no rate limiting; then filter by trusted domains
        search_query = f"{query} Казахстан"
        with DDGS() as ddgs:
            raw = list(ddgs.text(search_query, max_results=15))

        # Filter out blocked domains first
        raw = [r for r in raw if not any(b in r.get("href", "") for b in BLOCKED_DOMAINS)]
        # Priority 1: known official sites
        trusted = [r for r in raw if any(d in r.get("href", "") for d in OFFICIAL_SITES)]
        # Priority 2: any .kz domain (Kazakhstan only) — no Russian/foreign sites
        kz_sites = [r for r in raw if r.get("href", "").split("?")[0].endswith(".kz")
                    or "/.kz/" in r.get("href", "")
                    or r.get("href", "").split("/")[2].endswith(".kz")]
        # Merge: official first, then other .kz, deduplicate
        seen_urls = set()
        results = []
        for r in trusted + kz_sites:
            url = r.get("href", "")
            if url not in seen_urls:
                seen_urls.add(url)
                results.append(r)
        results = results[:max_results + 3]

        if not results:
            if language == "en":
                return f"No information found in Kazakhstan sources (.kz) for query: '{query}'."
            elif language == "kz":
                return f"'{query}' сұранысы бойынша қазақстандық дереккөздерден (.kz) ақпарат табылмады."
            else:
                return f"По запросу '{query}' информация в казахстанских источниках (.kz) не найдена."

        # Filter out documents that are no longer in force
        INVALID_MARKERS = [
            "утратил силу", "утратила силу", "утратило силу",
            "признан утратившим", "прекращено действие",
            "истечением срока", "недействующий", "отменен",
        ]
        results = [
            r for r in results
            if not any(m in (r.get("title", "") + r.get("body", "")).lower() for m in INVALID_MARKERS)
        ]

        if not results:
            if language == "en":
                return f"No current data found in official sources for query: '{query}'."
            elif language == "kz":
                return f"'{query}' сұранысы бойынша ресми дереккөздерде өзекті мәліметтер табылмады."
            else:
                return f"По запросу '{query}' актуальных данных в официальных источниках не найдено."

        # Show which sites results came from
        found_sites = list({r.get("href","").split("/")[2] for r in results if r.get("href")})
        if language == "en":
            header = f"Fresh data from official Kazakhstan sources ({', '.join(found_sites)}) for query '{query}':\n"
            no_title = "No title"
        elif language == "kz":
            header = f"Ресми қазақстандық дереккөздерден ({', '.join(found_sites)}) '{query}' сұранысы бойынша жаңа деректер:\n"
            no_title = "Атауы жоқ"
        else:
            header = f"Свежие данные из официальных источников РК ({', '.join(found_sites)}) по запросу '{query}':\n"
            no_title = "Без названия"
        lines = [header]
        for i, r in enumerate(results, 1):
            title = r.get("title", no_title)
            body  = r.get("body", "")[:300]
            url   = r.get("href", "")
            lines.append(
                f"📄 **{title}**\n"
                f"{body}\n"
                f"🔗 {url}"
            )
        return "\n\n---\n\n".join(lines)

    except Exception as e:
        return f"MCP search error: {e}"


async def _search_adilet(query: str, max_results: int = 3, language: str = "ru") -> str:
    """Async wrapper for DuckDuckGo search."""
    return _ddg_search_adilet(query, max_results, language)


# ── Tool definitions ──────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_kazakhstan_law",
            description=(
                "Поиск свежих данных по законодательству и госрегулированию Казахстана. "
                "Источники: adilet.zan.kz (законы), minfin.gov.kz (МЗП/МРП/налоги), "
                "kgd.gov.kz (налоговая служба), egov.kz (госуслуги для бизнеса). "
                "Используй для актуальных данных, которых нет в локальной базе."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос на русском языке"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Максимальное количество результатов (1-5)",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "search_kazakhstan_law":
        result = await _search_adilet(
            arguments["query"],
            arguments.get("max_results", 3)
        )
        return [types.TextContent(type="text", text=result)]
    raise ValueError(f"Unknown tool: {name}")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
