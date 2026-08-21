"""Flask UI (Day 8): query box -> the agent's cited answer -> collapsible trace.

  python -m src.app      # then open http://localhost:5000

The trace shows exactly which tools the hand-written agent called, so the UI
doubles as a window into how the answer was reached.
"""
from __future__ import annotations

import re

import markdown as md
from flask import Flask, render_template, request

from src import agent

app = Flask(__name__, template_folder="../templates")

EXAMPLES = [
    "Which architectures report the best Dice on head and neck segmentation, on which datasets, and how many cases each?",
    "Which methods report the highest Dice for pancreas segmentation, and on how many patients?",
    "What Dice scores are reported for brain tumor / glioma segmentation, and by which methods?",
    "Which methods use uncertainty estimation in medical image segmentation?",
]


def _linkify(html: str) -> str:
    """Turn arxiv_<id> citations into links to the paper's arXiv page."""
    return re.sub(
        r"arxiv_(\d+\.\d+)",
        r'<a href="https://arxiv.org/abs/\1" target="_blank" rel="noopener">arxiv_\1</a>',
        html)


@app.route("/", methods=["GET", "POST"])
def index():
    question = request.form.get("question", "").strip() if request.method == "POST" else ""
    result = None
    if question:
        try:
            out = agent.answer(question, verbose=False)
            result = {
                "answer": _linkify(md.markdown(out["answer"], extensions=["tables", "fenced_code"])),
                "trace": out["trace"],
                "steps": out["steps"],
                "tokens": sum(out["tokens"]),
            }
        except Exception as ex:  # noqa: BLE001 - most likely a Groq rate limit
            result = {"error": str(ex)}
    return render_template("index.html", question=question, result=result, examples=EXAMPLES)


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
