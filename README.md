# ollama-web-search

Give a local LLM running on [Ollama](https://ollama.com) live web search, news search and page reading,
in one Python file of about 100 lines.

Local models only know their training data. This agent lets the model decide when it needs fresh
information and call one of three tools:

| Tool | What it does |
|---|---|
| `web_search` | Search the web: facts, docs, products |
| `news_search` | Recent news by key words, days back (1-120) and country |
| `read_page` | The readable text of any URL, without menus or ads |

The tool-calling loop works with any Ollama model that supports tools (Qwen 3, Llama 3.1, Mistral and more).

Full walkthrough: [Give a local LLM live web search and news with Ollama and Python](https://nestdaddy.com/blog/local-llm-web-search-ollama).

## Quick start

```bash
ollama pull qwen3:8b
pip install -r requirements.txt
export NESTDADDY_API_KEY="your_key"      # Windows: setx NESTDADDY_API_KEY "your_key"
python web_agent.py "What happened in AI this week?"
```

Pick another model with `OLLAMA_MODEL`, for example `OLLAMA_MODEL=llama3.1 python web_agent.py "..."`.

The search tools use the [NestDaddy API](https://nestdaddy.com/apis), an independent search engine with its own
index. A free key (1,000 calls a month, no card) works. To use another search provider, change only the body of
the three tool functions: the loop stays the same.

## How the loop works

1. The model gets the question, a short system prompt and the three tools.
2. If it answers with tool calls, each tool runs and its result goes back as a `tool` message.
3. Errors and unknown tool names are returned to the model instead of crashing the script.
4. It stops when the model answers with text, or after `max_steps` (default 6).

## Tests

Offline, no Ollama server or key needed:

```bash
python -m unittest discover -s tests -v
```

## More

- [nestdaddy-examples](https://github.com/nestdaddy/nestdaddy-examples): Python, JavaScript, curl and MCP examples
- [news-digest-bot](https://github.com/nestdaddy/news-digest-bot): a daily news digest to Telegram, Slack or email
- [NestDaddy MCP](https://nestdaddy.com/mcp): the same tools for Claude, Cursor and VS Code

MIT License.

## Community

Questions, ideas, or something you built? Join **Agent Builders**, our Discord for people building AI agents and MCP servers: daily MCP news, alerts when MCP servers change their tools, and help from our team.

[Join on Discord](https://discord.gg/CXStfSFC3v)
