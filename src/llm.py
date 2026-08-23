"""Provider-agnostic LLM wrapper (backend: Groq free tier, gpt-oss-120b).

The rest of the codebase calls llm.chat / llm.chat_json / llm.chat_tools only, so
switching providers means editing this one file. Groq's API is OpenAI-compatible,
so the tool-calling agent also lives behind this wrapper.

Rate-limit handling: Groq's free tier caps at 8000 tokens/minute (TPM) and a
daily token budget (TPD). We wait out transient per-minute limits (they clear in
seconds) but fail fast on the daily one (its retry-after is ~40 min).
"""
from __future__ import annotations

import json
import re
import time

from groq import Groq, RateLimitError

from src import config

DEFAULT_MODEL = "openai/gpt-oss-120b"

_client: Groq | None = None


def _get() -> Groq:
    global _client
    if _client is None:
        if not config.GROQ_API_KEY or "PASTE" in config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set. Add your key to the .env file "
                               "(get one free at https://console.groq.com/keys).")
        # max_retries=0: we handle retries ourselves so we can distinguish TPM from TPD.
        _client = Groq(api_key=config.GROQ_API_KEY, max_retries=0, timeout=90.0)
    return _client


def _create(**params):
    """Call Groq, waiting out per-minute (TPM) rate limits but not the daily (TPD) one."""
    for attempt in range(6):
        try:
            return _get().chat.completions.create(**params)
        except RateLimitError as e:
            msg = str(e)
            if "per day" in msg or "TPD" in msg:
                raise                                   # daily budget gone; don't sleep ~40 min
            wait = re.search(r"try again in ([\d.]+)s", msg)
            if attempt == 5:
                raise
            time.sleep(min((float(wait.group(1)) + 1) if wait else 12.0, 25.0))


def chat(system: str, user: str, *, model: str = DEFAULT_MODEL,
         max_tokens: int = 512, temperature: float = 0.0, json_mode: bool = False):
    """One chat turn. Returns (text, usage). json_mode forces a valid JSON object."""
    params = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": user}]}
    if json_mode:
        params["response_format"] = {"type": "json_object"}
    resp = _create(**params)
    return resp.choices[0].message.content, resp.usage


def chat_json(system: str, user: str, **kw):
    """chat() in JSON mode, parsed to a dict. Returns (dict, usage)."""
    text, usage = chat(system, user, json_mode=True, **kw)
    return json.loads(text), usage


def chat_tools(messages: list, tools: list, *, model: str = DEFAULT_MODEL,
               max_tokens: int = 1500, temperature: float = 0.0, tool_choice: str = "auto"):
    """One tool-calling turn. Returns (message, usage). The message may carry
    .content (final text) and/or .tool_calls (requests for us to run tools).
    tool_choice='none' forces a text answer (used to finalize). This is the raw
    API surface the hand-written agent loop drives."""
    resp = _create(model=model, messages=messages, tools=tools, tool_choice=tool_choice,
                   max_tokens=max_tokens, temperature=temperature)
    return resp.choices[0].message, resp.usage
