# Hand-written agent vs LangGraph: head-to-head

**Setup.** Identical model (Groq `gpt-oss-120b`), identical tools, identical system
prompt. The only variable is orchestration: a hand-written Python while-loop
(`src/agent.py`) vs a LangGraph `StateGraph` (`src/agent_langgraph.py`). Lower
steps/tokens/latency is better; `Answer` is whether a real (non-stopped) answer was
produced. Run on a curated custom question set via `python -m src.compare_agents
--custom`. Raw run log: `agent_comparison.csv`.

## Per-question results

| Question | Type | Hand-written (steps / tokens) | LangGraph (steps / tokens) | Both answered |
|---|---|--:|--:|:--:|
| head_neck_dice | comparative metric | 7 / 17,689 | 4 / 7,952 | yes |
| pancreas_dice | metric lookup | 2 / 2,661 | 2 / 2,738 | yes |
| uncertainty | conceptual (search-heavy) | 8 / 26,481 | 9 / 27,016 | yes |
| brain_unet | comparative concept | 7 / 18,700 | 9 / 26,049 | yes |

## Averages

| Impl | Avg steps | Avg tokens | Avg latency (s) | Answered |
|---|--:|--:|--:|--:|
| hand-written | 6.0 | 16,383 | 107.7 | 4/4 |
| langgraph | 6.0 | 15,939 | 104.9 | 4/4 |

## Findings

**1. With the same model, tools, and prompt, the two implementations behave the
same.** Both answer all 4/4 questions, and the averages are within ~3% on every axis
(identical mean step count, 16,383 vs 15,939 tokens, 107.7s vs 104.9s). This is the
expected and reassuring result: swapping a hand-written loop for a framework graph does
not change *what* the agent does - it changes only how the control flow is expressed.

**2. Per-question step/token differences are sampling noise, not orchestration.** The
model samples non-deterministically, so the same question can take a different number of
tool-calling rounds on different runs (e.g. `head_neck_dice` took the hand-written loop
7 steps here but 4 in an earlier run). The two agents fan out over the same tools in the
same way; where one uses an extra round on a given run, it is the LLM's choice, not the
harness's. Averaged over the set, neither implementation is systematically leaner.

**3. The one real, reproducible difference is graceful finalize-on-cap, and it is a
clean illustration of the hand-written-vs-framework trade-off.** When a question needs
more than the iteration cap, the hand-written loop calls `_finalize()` - a plain function
that asks the model for a best-effort answer from the evidence already gathered. The
*first* LangGraph port had no equivalent, so on the two conceptual questions (uncertainty,
brain_unet) it hit LangGraph's `recursion_limit` and raised `GraphRecursionError`,
producing no answer at all. Matching the hand-written behaviour required adding an
explicit `finalize` node, a `finalize` edge, and an iteration counter to the graph state
(now in `agent_langgraph.py`). With that in place, LangGraph answers all 4/4 - which is
why every row above says "yes".

This is the concrete, measured version of the thesis: **the framework makes the
happy-path control flow declarative, but a guardrail behaviour that is a one-line
function call in a hand-written loop becomes extra machinery (a node + an edge + a state
field) in the graph.**

## The interview takeaway

"They behave identically on the same model, tools, and prompt - both answer 4/4 with
averages within 3% - which is exactly the point. The difference is engineering
ergonomics: my hand-written loop got a graceful finalize for free as a function call;
making LangGraph do the same needed a dedicated node, edge, and counter, and until I
added it the graph errored out on the two hardest questions instead of answering. The
framework pays off as agents grow - branching, retries, checkpointing, visualization -
but for a single linear plan-act loop it is more machinery for the same behaviour.
Building the loop by hand first is what let me see exactly what the framework adds and
what it costs."
