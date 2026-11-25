"""A local LLM (Ollama) with live web search, news search and page reading.

Usage:
    export NESTDADDY_API_KEY="your_key"
    python web_agent.py "What happened in AI this week?"

Guide: https://nestdaddy.com/blog/local-llm-web-search-ollama
"""
import json
import os

import ollama
import requests

API = "https://nestdaddy.com/api/v1"
KEY = os.environ.get("NESTDADDY_API_KEY") or os.environ.get("NESTDADDY_KEY", "")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")


def _get(path, **params):
    r = requests.get(f"{API}/{path}", headers={"X-API-Key": KEY},
                     params={k: v for k, v in params.items() if v is not None},
                     timeout=30)
    r.raise_for_status()
    return r.json()


def web_search(query: str, limit: int = 5) -> str:
    """Search the web. Use it for facts, documentation, products and anything
    the model may not know.

    Args:
        query: Short key words, like a search box query.
        limit: Number of results, 1 to 10.

    Returns:
        JSON list of results with title, url and snippet.
    """
    data = _get("search", q=query, limit=max(1, min(int(limit), 10)))
    return json.dumps([
        {"title": x["title"], "url": x["url"], "snippet": x.get("snippet", "")}
        for x in data.get("results", [])
    ])


def news_search(query: str, days: int = 7, country: str = "") -> str:
    """Search recent news articles. Use it for events, prices, launches and
    anything that happened recently.

    Args:
        query: Key words, for example a company, person or topic.
        days: How many days back to search, 1 to 120.
        country: Optional two-letter country code such as US, GB, DE or JP.

    Returns:
        JSON list of articles with date, title, source, url and description.
    """
    data = _get("news/search", q=query, days=max(1, min(int(days), 120)),
                country=country or None, limit=8)
    return json.dumps([
        {"date": a.get("published_timestamp"), "title": a["title"],
         "source": a.get("source_domain"), "url": a["url"],
         "description": a.get("description", "")}
        for a in data.get("results", [])
    ])


def read_page(url: str) -> str:
    """Read the main text of a web page, without menus or ads. Use it after a
    search when a snippet is not enough.

    Args:
        url: The full URL of the page.

    Returns:
        The page text, cut to 6,000 characters.
    """
    data = _get("extract", url=url)
    return (data.get("text") or "")[:6000]


TOOLS = {f.__name__: f for f in (web_search, news_search, read_page)}

SYSTEM = (
    "You are a research assistant with live web access. "
    "Use news_search for recent events and web_search for everything else. "
    "Read the one or two best pages in full with read_page before answering "
    "a detailed question. Answer briefly and list the URLs you used."
)


def ask(question: str, max_steps: int = 6) -> str:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    for _ in range(max_steps):
        response = ollama.chat(model=MODEL, messages=messages,
                               tools=list(TOOLS.values()))
        messages.append(response.message)
        calls = response.message.tool_calls or []
        if not calls:
            return response.message.content
        for call in calls:
            fn = TOOLS.get(call.function.name)
            try:
                output = fn(**call.function.arguments) if fn else "unknown tool"
            except Exception as exc:  # give the error back to the model
                output = f"tool error: {exc}"
            messages.append({"role": "tool", "content": output,
                             "tool_name": call.function.name})
    return "Stopped after too many tool calls."


if __name__ == "__main__":
    import sys
    if not KEY:
        sys.exit("Set NESTDADDY_API_KEY first (every plan starts with 7 days free: https://nestdaddy.com/pricing)")
    print(ask(" ".join(sys.argv[1:]) or "What happened in AI this week?"))
