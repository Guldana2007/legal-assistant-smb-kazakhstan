"""
Shared utilities for all agents.
"""

import os, json

# ── LangFuse ──
_lf_client = None

def setup_langfuse():
    """Initialize LangFuse client using Python SDK directly (no LangChain callback)."""
    global _lf_client
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    host       = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

    if not public_key or not secret_key:
        print("  [LangFuse] OFF — add LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to .env")
        return False

    try:
        from langfuse import Langfuse
        _lf_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        print(f"  [LangFuse] ON — dashboard: {host}")
        return True
    except Exception as e:
        print(f"  [LangFuse] init error: {e}")
        return False


LANGFUSE = setup_langfuse()


def get_lf_config() -> dict:
    """Returns empty config (tracing done via SDK, not LangChain callbacks)."""
    return {}


def get_lf_handler():
    """Returns None — tracing done via SDK directly."""
    return None


def log_trace(name: str, input_data: dict, output_data: dict, metadata: dict = None):
    """Log a trace event to LangFuse using the Python SDK."""
    if not LANGFUSE or _lf_client is None:
        return
    try:
        trace = _lf_client.trace(
            name=name,
            input=input_data,
            output=output_data,
            metadata=metadata or {},
        )
        return trace
    except Exception:
        pass


def flush_langfuse():
    """Flush all pending traces to LangFuse before exit."""
    if not LANGFUSE or _lf_client is None:
        return
    try:
        _lf_client.flush()
        print("  [LangFuse] traces flushed")
    except Exception:
        pass


def parse_json(raw: str, fallback: dict) -> dict:
    """Parse LLM JSON response, stripping markdown fences if needed."""
    try:
        text = raw.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception:
        return fallback


def add_trace(state: dict, node: str, detail: str, data: dict = None) -> list:
    entry = {"node": node, "detail": detail, "data": data or {}}
    return list(state.get("trace", [])) + [entry]
