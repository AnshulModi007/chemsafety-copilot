"""Web search fallback via Tavily, used only when the CSB corpus has no
confident answer (see src/generation/crag.py's insufficient-retrieval signal).
A closed corpus always has gaps -- current events, incidents never filed as a
CSB report, chemicals PubChem doesn't cover -- so falling back to the open web
lets the assistant answer those instead of just refusing.
"""
import sys
from pathlib import Path

from tavily import TavilyClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import TAVILY_API_KEY  # noqa: E402


class WebSearchUnavailable(Exception):
    """Raised when TAVILY_API_KEY isn't configured -- callers should fall back
    to the plain refusal message rather than let this bubble up as a 500."""


def web_search(query: str, max_results: int = 5) -> list[dict]:
    if not TAVILY_API_KEY:
        raise WebSearchUnavailable("TAVILY_API_KEY is not set")

    client = TavilyClient(api_key=TAVILY_API_KEY)
    response = client.search(query, max_results=max_results, include_answer=False)
    return [
        {"title": r["title"], "url": r["url"], "content": r["content"]}
        for r in response.get("results", [])
    ]
