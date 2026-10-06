# %%
"""Deprecation state benchmark: does the model know what still works?

Three states, not two. A feature can be removed, deprecated-but-functional, or
current. Most evals of this shape collapse the middle state, which hides two
opposite failures: treating a live-but-deprecated feature as broken, and treating
a removed one as fine.

Two question forms per item:
  knowledge  a neutral question about the feature's state
  premise    a request that assumes the feature is usable

The gap between them is the point. A model can know a feature is gone and still
write the code when asked directly.

Run locally with `python task.py` to validate before pushing.
"""

import os
import sys

import kaggle_benchmarks as kbench

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classify import (  # noqa: E402
    CURRENT,
    DEPRECATED_LIVE,
    REMOVED,
    UNCLEAR,
    classify,
)
from items import ITEMS  # noqa: E402

# The proxy reserves worst-case cost per call from the output-token limit. Left
# unset, a single call to an expensive model can reserve several dollars against
# a $10 daily quota and then be refused. This cap is not a nicety.
MAX_TOKENS = 1024

# Asking for a short answer keeps cost down and makes classification cleaner:
# long answers hedge in both directions and land in UNCLEAR.
PREAMBLE = (
    "Answer in at most three sentences. If the feature you are asked about is "
    "removed, deprecated, or current, say so explicitly."
)


# Candidate spellings for the output-token cap, in the order we try them.
#
# The Kaggle proxy fronts every vendor with one OpenAI-compatible surface
# (`Completions.create()`), so the native Google spelling `max_output_tokens` is
# rejected even for Gemini models. `max_tokens` is the OpenAI SDK parameter;
# `max_completion_tokens` is its newer name. Trying both covers either SDK
# version without caring which vendor is behind the proxy.
#
# This is not a tuning knob. The proxy reserves the worst-case cost of every
# call from the output limit and refuses the call when that exceeds the
# remaining daily quota. Unset, one call to a large model can reserve several
# dollars against a $10 cap.
_CAP_KEYS = ("max_tokens", "max_completion_tokens")

# Set once per process, after the first successful call, so we do not pay for a
# failed attempt on every single prompt.
_WORKING_CAP_KEY: list = []


def _ask(llm, question):
    text = f"{PREAMBLE}\n\n{question}"

    if _WORKING_CAP_KEY:
        key = _WORKING_CAP_KEY[0]
        if key is None:
            return llm.prompt(text)
        return llm.prompt(text, extra_api_params={key: MAX_TOKENS})

    for key in _CAP_KEYS:
        try:
            out = llm.prompt(text, extra_api_params={key: MAX_TOKENS})
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            continue
        _WORKING_CAP_KEY.append(key)
        print(f"  output cap set via extra_api_params={{{key!r}: {MAX_TOKENS}}}")
        return out

    # No spelling accepted. Running uncapped risks the reservation trap, so stop
    # rather than quietly spending the daily quota a few calls at a time.
    raise RuntimeError(
        "Could not set an output-token cap: none of "
        f"{_CAP_KEYS} was accepted. Running uncapped risks the proxy reserving "
        "several dollars per call against the daily quota. Inspect "
        "`inspect.signature(llm.prompt)` and the proxy's accepted parameters "
        "before continuing."
    )


@kbench.task(name="deprecation_state")
def deprecation_state(llm) -> dict:
    """Score every item on both question forms. Returns a flat result dict."""
    rows = []

    for item in ITEMS:
        for form in ("knowledge", "premise"):
            question = item.knowledge_q if form == "knowledge" else item.premise_q
            response = _ask(llm, question)
            predicted = classify(response)

            rows.append(
                {
                    "item": item.id,
                    "vendor": item.vendor,
                    "bucket": item.bucket,
                    "form": form,
                    "truth": item.truth,
                    "predicted": predicted,
                    "correct": predicted == item.truth,
                    "response": response,
                }
            )

    scored = [r for r in rows if r["predicted"] != UNCLEAR]
    correct = [r for r in rows if r["correct"]]

    # The two failures worth naming separately.
    missed_removal = [
        r for r in rows
        if r["truth"] == REMOVED and r["predicted"] in (CURRENT, DEPRECATED_LIVE)
    ]
    over_warned = [
        r for r in rows
        if r["truth"] in (DEPRECATED_LIVE, CURRENT) and r["predicted"] == REMOVED
    ]

    def rate(subset, pool):
        return round(len(subset) / len(pool), 4) if pool else 0.0

    by_form = {}
    for form in ("knowledge", "premise"):
        pool = [r for r in rows if r["form"] == form]
        by_form[form] = rate([r for r in pool if r["correct"]], pool)

    by_bucket = {}
    for bucket in sorted({r["bucket"] for r in rows}):
        pool = [r for r in rows if r["bucket"] == bucket]
        by_bucket[bucket] = rate([r for r in pool if r["correct"]], pool)

    by_truth = {}
    for state in (REMOVED, DEPRECATED_LIVE, CURRENT):
        pool = [r for r in rows if r["truth"] == state]
        by_truth[state] = rate([r for r in pool if r["correct"]], pool)

    kbench.assertions.assert_true(
        len(scored) >= len(rows) * 0.6,
        expectation=(
            "At least 60 percent of responses should be classifiable. A high "
            "UNCLEAR rate means the prompt or classifier needs work, not that "
            "the model failed."
        ),
    )

    return {
        "accuracy": rate(correct, rows),
        "accuracy_knowledge": by_form["knowledge"],
        "accuracy_premise": by_form["premise"],
        # The headline number: knowing minus acting on it.
        "knowledge_premise_gap": round(
            by_form["knowledge"] - by_form["premise"], 4
        ),
        "accuracy_by_bucket": by_bucket,
        "accuracy_by_truth": by_truth,
        "missed_removal_rate": rate(missed_removal, rows),
        "over_warn_rate": rate(over_warned, rows),
        "unclear_rate": rate([r for r in rows if r["predicted"] == UNCLEAR], rows),
        "n_calls": len(rows),
        "rows": rows,
    }


deprecation_state.run(llm=kbench.llm)
