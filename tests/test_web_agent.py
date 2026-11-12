"""Offline tests: no Ollama server and no API key needed."""
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ollama import ChatResponse, Message  # noqa: E402

import web_agent  # noqa: E402


def reply(text="", calls=()):
    tool_calls = [Message.ToolCall(function=Message.ToolCall.Function(name=n, arguments=a))
                  for n, a in calls]
    return ChatResponse(message=Message(role="assistant", content=text,
                                        tool_calls=tool_calls or None))


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class ToolTests(unittest.TestCase):
    def test_web_search_returns_small_json_and_clamps_limit(self):
        data = {"results": [{"title": "T", "url": "https://x", "snippet": "S", "score": 9}]}
        with mock.patch.object(web_agent.requests, "get", return_value=FakeResponse(data)) as get:
            out = json.loads(web_agent.web_search("q", limit=50))
        self.assertEqual(out, [{"title": "T", "url": "https://x", "snippet": "S"}])
        self.assertEqual(get.call_args.kwargs["params"]["limit"], 10)

    def test_news_search_drops_empty_country_and_clamps_days(self):
        data = {"results": [{"title": "N", "url": "https://n", "source_domain": "bbc.co.uk",
                             "published_timestamp": "2026-11-12T10:00:00Z", "description": "D"}]}
        with mock.patch.object(web_agent.requests, "get", return_value=FakeResponse(data)) as get:
            out = json.loads(web_agent.news_search("q", days=999))
        params = get.call_args.kwargs["params"]
        self.assertNotIn("country", params)
        self.assertEqual(params["days"], 120)
        self.assertEqual(out[0]["source"], "bbc.co.uk")

    def test_read_page_cuts_long_text(self):
        with mock.patch.object(web_agent.requests, "get", return_value=FakeResponse({"text": "a" * 9000})):
            self.assertEqual(len(web_agent.read_page("https://x")), 6000)


class LoopTests(unittest.TestCase):
    def test_runs_tools_then_returns_answer(self):
        script = [reply(calls=[("web_search", {"query": "q"})]), reply("done")]
        seen = []

        def fake_chat(model, messages, tools):
            seen.append(list(messages))
            return script.pop(0)

        with mock.patch.object(web_agent.ollama, "chat", side_effect=fake_chat), \
             mock.patch.dict(web_agent.TOOLS, {"web_search": lambda **_: "[]"}):
            self.assertEqual(web_agent.ask("hi"), "done")
        last = seen[-1][-1]
        self.assertEqual(last["role"], "tool")
        self.assertEqual(last["tool_name"], "web_search")

    def test_unknown_tool_and_errors_go_back_to_model(self):
        def boom(**_):
            raise RuntimeError("timeout")

        script = [reply(calls=[("nope", {}), ("read_page", {"url": "u"})]), reply("ok")]
        seen = []

        def fake_chat(model, messages, tools):
            seen.append(list(messages))
            return script.pop(0)

        with mock.patch.object(web_agent.ollama, "chat", side_effect=fake_chat), \
             mock.patch.dict(web_agent.TOOLS, {"read_page": boom}):
            self.assertEqual(web_agent.ask("hi"), "ok")
        tool_msgs = [m for m in seen[-1] if isinstance(m, dict) and m.get("role") == "tool"]
        self.assertEqual(tool_msgs[0]["content"], "unknown tool")
        self.assertIn("tool error: timeout", tool_msgs[1]["content"])

    def test_stops_after_max_steps(self):
        with mock.patch.object(web_agent.ollama, "chat",
                               return_value=reply(calls=[("nope", {})])):
            self.assertIn("Stopped", web_agent.ask("hi", max_steps=2))


if __name__ == "__main__":
    unittest.main()
