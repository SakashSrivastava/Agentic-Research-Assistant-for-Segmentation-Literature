# Hand-written agent vs LangGraph: head-to-head

**Setup.** Identical model (Groq `gpt-oss-120b`), identical tools, identical system
prompt. The only variable is orchestration: a hand-written Python while-loop
(`src/agent.py`) vs a LangGraph `StateGraph` (`src/agent_langgraph.py`). Lower
steps/tokens is better. Run on a curated custom question set via
`python -m src.compare_agents --custom`. Raw run log: `agent_comparison.csv`.

## Per-question results

| Question | Type | Hand-written (steps / tokens) | LangGraph (steps / tokens) |
|---|---|--:|--:|
| head_neck_dice | comparative metric | 4 / 7,858 | 4 / 7,959 |
| pancreas_dice | metric lookup | 2 / 2,737 | 2 / 2,980 |
| uncertainty | conceptual (search-heavy) | 8 / 29,496 | pending* |
| brain_unet | comparative concept | 8 / 30,343 | pending* |

\* The conceptual questions need ~8 tool-calling iterations. The LangGraph re-run to
capture these numbers is pending a fresh daily token budget (the run that would
produce them hit Groq's per-day rate limit). See the finding below.

## Findings

**1. On matched questions the two agents are near-identical.** Same step count, and
tokens within ~1-8%. This is the expected and reassuring result: with the same model,
tools, and prompt, swapping a hand-written loop for a framework graph does not change
the agent's behaviour - it changes only how the control flow is expressed.

**2. The one real difference is graceful finalize-on-cap, and it favours the
hand-written loop's simplicity.** When a question needs more than the 8-iteration cap,
the hand-written loop simply calls `_finalize()` - a plain function that asks the model
for a best-effort answer from the evidence already gathered. The first LangGraph port
had no such step, so on the two conceptual questions it hit LangGraph's `recursion_limit`
and raised a `GraphRecursionError`, producing no answer at all.

Matching the hand-written behaviour in LangGraph required adding an explicit
`finalize` node, a `finalize` edge, and an iteration counter in the graph state
(now implemented in `agent_langgraph.py`). This is the concrete, measured version of
the "hand-written vs framework" thesis: **the framework makes the happy-path control
flow declarative, but a guardrail behaviour that is trivial in a hand-written loop
becomes extra machinery (a node + edge + state field) in the graph.**

## The interview takeaway

"They behave identically on the same model, tools, and prompt - which is the point. The
difference is engineering ergonomics: my hand-written loop got a graceful finalize for
free as a function call; making LangGraph do the same needed a dedicated node, edge, and
counter. The framework pays off as agents grow (branching, retries, checkpointing,
visualization); for a single linear plan-act loop it is more machinery for the same
behaviour. Building the loop by hand first is what let me see exactly what the framework
adds and what it costs."

## To finish (fresh budget)

Re-run the two conceptual questions through the fixed LangGraph agent to fill in the
`pending*` cells:
```
python -m src.compare_agents --custom
```
