"""Provider-agnostic LLM wrapper (backend: Groq free tier, gpt-oss-120b).

The rest of the codebase calls llm.chat / llm.chat_json only, so switching
providers means editing this one file. Groq's API is OpenAI-compatible, so the
tool-calling agent (built later) also lives behind this wrapper.
"""
from __future__ import annotations

import json

from groq import Groq

from src import config

DEFAULT_MODEL = "openai/gpt-oss-120b"

_client: Groq | None = None


def _get() -> Groq:
    global _client
    if _client is None:
        if not config.GROQ_API_KEY or "PASTE" in config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set. Add your key to the .env file "
                               "(get one free at https://console.groq.com/keys).")
        _client = Groq(api_key=config.GROQ_API_KEY, max_retries=2, timeout=90.0)
    return _client


def chat(system: str, user: str, *, model: str = DEFAULT_MODEL,
         max_tokens: int = 512, temperature: float = 0.0, json_mode: bool = False):
    """One chat turn. Returns (text, usage). json_mode forces a valid JSON object."""
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = _get().chat.completions.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        **kwargs,
    )
    return resp.choices[0].message.content, resp.usage


def chat_json(system: str, user: str, **kw):
    """chat() in JSON mode, parsed to a dict. Returns (dict, usage)."""
    text, usage = chat(system, user, json_mode=True, **kw)
    return json.loads(text), usage


def chat_tools(messages: list, tools: list, *, model: str = DEFAULT_MODEL,
               max_tokens: int = 1500, temperature: float = 0.0):
    """One tool-calling turn. Returns (message, usage). The message may carry
    .content (final text) and/or .tool_calls (requests for us to run tools).
    This is the raw API surface the hand-written agent loop drives."""
    resp = _get().chat.completions.create(
        model=model, messages=messages, tools=tools, tool_choice="auto",
        max_tokens=max_tokens, temperature=temperature,
    )
    return resp.choices[0].message, resp.usage
