"""Part B: the agent re-implemented as a LangGraph StateGraph.

Same model (Groq via src.llm), same tools, and same system prompt as the
hand-written loop in src/agent.py. Only the orchestration differs: an explicit
graph of nodes and edges instead of a Python while-loop. Holding everything else
constant makes this a clean "hand-written loop vs framework" comparison.

  python -m src.agent_langgraph "Which architectures report the best Dice on head and neck?"
"""
from __future__ import annotations

import argparse
import json
import operator
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src import agent as A  # reuse SYSTEM, TOOLS, DISPATCH, _msg_to_dict, MAX_ITERS
from src import llm


FINALIZE_PROMPT = (
    "Stop gathering. Using ONLY the evidence already gathered above, give your best "
    "concise final answer now, citing the paper_id for each claim. If the evidence is "
    "insufficient for part of the question, say so explicitly.")


class State(TypedDict):
    messages: list
    trace: Annotated[list, operator.add]   # accumulated across tool steps
    tokens_in: int
    tokens_out: int
    iters: int
    api_key: Optional[str]


def _llm_node(state: State) -> dict:
    """One model turn: it either asks for tools or produces the final answer."""
    msg, usage = llm.chat_tools(state["messages"], A.TOOLS, api_key=state.get("api_key"))
    return {"messages": state["messages"] + [A._msg_to_dict(msg)],
            "tokens_in": state["tokens_in"] + usage.prompt_tokens,
            "tokens_out": state["tokens_out"] + usage.completion_tokens,
            "iters": state["iters"] + 1}


def _finalize_node(state: State) -> dict:
    """Iteration cap reached: force a tool-free answer from the gathered evidence.
    This is what the hand-written loop does with a plain _finalize() call; here it
    needs its own node + edge - a concrete example of the framework trade-off."""
    msgs = state["messages"] + [{"role": "user", "content": FINALIZE_PROMPT}]
    msg, usage = llm.chat_tools(msgs, A.TOOLS, tool_choice="none", api_key=state.get("api_key"))
    return {"messages": state["messages"] + [A._msg_to_dict(msg)],
            "tokens_in": state["tokens_in"] + usage.prompt_tokens,
            "tokens_out": state["tokens_out"] + usage.completion_tokens}


def _tools_node(state: State) -> dict:
    """Run every tool the last assistant message requested, append the results."""
    last = state["messages"][-1]
    msgs, trace = list(state["messages"]), []
    for tc in last["tool_calls"]:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"]["arguments"] or "{}")
            result = A.DISPATCH[name](**args)
        except Exception as ex:  # noqa: BLE001
            args, result = {}, {"error": str(ex)}
        trace.append({"tool": name, "args": args, "result_preview": str(result)[:200]})
        msgs.append({"role": "tool", "tool_call_id": tc["id"],
                     "content": json.dumps(result, default=str)[:2000]})
    return {"messages": msgs, "trace": trace}


def _route(state: State) -> str:
    """Conditional edge: run tools if the model asked and we're under the iteration
    cap; force a finalize at the cap; otherwise the model answered, so stop."""
    if not state["messages"][-1].get("tool_calls"):
        return END
    return "tools" if state["iters"] < A.MAX_ITERS else "finalize"


_GRAPH = None


def _graph():
    global _GRAPH
    if _GRAPH is None:
        g = StateGraph(State)
        g.add_node("llm", _llm_node)
        g.add_node("tools", _tools_node)
        g.add_node("finalize", _finalize_node)
        g.set_entry_point("llm")
        g.add_conditional_edges("llm", _route,
                                {"tools": "tools", "finalize": "finalize", END: END})
        g.add_edge("tools", "llm")
        g.add_edge("finalize", END)
        _GRAPH = g.compile()
    return _GRAPH


def answer(question: str, verbose: bool = False, api_key: str | None = None) -> dict:
    """Same return shape as src.agent.answer, so evals and the app can swap them."""
    init: State = {"messages": [{"role": "system", "content": A.SYSTEM},
                                {"role": "user", "content": question}],
                   "trace": [], "tokens_in": 0, "tokens_out": 0, "iters": 0, "api_key": api_key}
    # recursion_limit bounds super-steps (llm+tools ~ 2 per iteration). Give room for
    # MAX_ITERS iterations plus the finalize node so the graph never raises on a normal run.
    try:
        final = _graph().invoke(init, config={"recursion_limit": A.MAX_ITERS * 2 + 4})
    except Exception as ex:  # noqa: BLE001 - recursion limit or LLM unavailable
        return {"answer": f"(stopped: {str(ex)[:120]})", "trace": [], "steps": 0, "tokens": (0, 0)}
    ans = final["messages"][-1].get("content") or "(no answer produced)"
    steps = sum(1 for m in final["messages"] if m.get("role") == "assistant")
    if verbose:
        for t in final["trace"]:
            print(f"  {t['tool']}({t['args']}) -> {t['result_preview'][:100]}")
    return {"answer": ans, "trace": final["trace"], "steps": steps,
            "tokens": (final["tokens_in"], final["tokens_out"])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LangGraph research agent (Part B).")
    ap.add_argument("question")
    args = ap.parse_args()
    print(f"\nQ: {args.question}\n")
    out = answer(args.question, verbose=True)
    print(f"\n=== ANSWER ({out['steps']} steps, {sum(out['tokens'])} tokens) ===\n{out['answer']}")
