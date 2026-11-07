"""Agents under test. An agent is any callable: Scenario -> AgentResult."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable

from .types import AgentError, AgentResult, Scenario, ToolCall

Agent = Callable[[Scenario], AgentResult]


def reference(kind: str) -> Agent:
    """Replays each scenario's own reference response: 'pass' or 'fail'. Used to check the graders."""
    def agent(s: Scenario) -> AgentResult:
        if kind not in s.fixtures:
            raise AgentError(f"scenario {s.id!r} has no {kind!r} fixture")
        return AgentResult.from_dict(s.fixtures[kind])
    return agent


class AnthropicAgent:
    """One turn against the Anthropic Messages API.

    It records what the model *asks* to do (text plus tool_use blocks) and does not execute
    any tool: for rules like "never send without approval" the question is whether the model
    reaches for the tool at all. Needs ANTHROPIC_API_KEY (or api_key=).

    Tested against a local stub server for request shape and response parsing; not run
    against the live API in this repository's CI.
    """

    def __init__(self, model: str, api_key: str | None = None, base_url: str = "https://api.anthropic.com",
                 max_tokens: int = 1024, temperature: float = 1.0, timeout: float = 60.0):
        self.model, self.base_url, self.max_tokens = model, base_url.rstrip("/"), max_tokens
        self.temperature, self.timeout = temperature, timeout
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise AgentError("ANTHROPIC_API_KEY is not set")

    def request_body(self, s: Scenario) -> dict:
        body = {"model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature, "messages": s.messages}
        if s.system:
            body["system"] = s.system
        if s.tools:
            body["tools"] = s.tools
        return body

    def __call__(self, s: Scenario) -> AgentResult:
        req = urllib.request.Request(
            f"{self.base_url}/v1/messages", data=json.dumps(self.request_body(s)).encode(), method="POST",
            headers={"content-type": "application/json", "x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            code = e.code
            e.close()   # release the response socket; an unread error body otherwise leaks it
            raise AgentError(f"API returned HTTP {code}") from None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            raise AgentError(f"API call failed: {type(e).__name__}") from None
        return self.parse(data)

    @staticmethod
    def parse(data: dict) -> AgentResult:
        text, calls = [], []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                calls.append(ToolCall(block["name"], block.get("input", {})))
        return AgentResult("\n".join(text), tuple(calls))
