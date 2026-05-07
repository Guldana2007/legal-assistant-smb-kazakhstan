"""
Agent: MCP Legal Search
========================
Searches Kazakhstan legal database (adilet.zan.kz) via MCP server.
Calls the underlying search function from legal_kz_server.py.
The MCP server (legal_kz_server.py) can also be run standalone via stdio.
"""

import asyncio
import sys
from pathlib import Path
from .shared import add_trace

# Add project root to path so we can import mcp_server module
sys.path.insert(0, str(Path(__file__).parent.parent))


async def _call_mcp_search(query: str) -> str:
    """Call MCP server search function for Kazakhstan legal database."""
    try:
        from mcp_server.legal_kz_server import _search_adilet
        result = await _search_adilet(query, max_results=6)
        return result
    except Exception as e:
        return f"MCP search error: {e}"


def node_mcp_legal_search(state: dict) -> dict:
    """LangGraph node: searches Kazakhstan law via MCP → adilet.zan.kz."""
    # Always use original question — more specific than reformulated query
    query = state.get("question", state.get("current_query", ""))

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_call_mcp_search(query))
        loop.close()
    except Exception as e:
        result = f"MCP unavailable: {e}"

    # Extract URLs from result for trace
    import re
    urls = re.findall(r'https://adilet\.zan\.kz/\S+', result)
    url_str = " | ".join(urls[:2]) if urls else "no links"

    return {
        "mcp_context": result,
        "trace": add_trace(
            state, "MCP_LEGAL_SEARCH",
            f"adilet.zan.kz → {url_str}",
            {"urls": urls, "preview": result[:150]},
        ),
    }
