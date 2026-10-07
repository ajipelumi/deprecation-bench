# %%
"""Deprecation state benchmark: does the model know what still works?

Four states. A feature is removed, deprecated-but-functional, or current;
DENIES_EXISTENCE is the model's state, not the feature's, for when it reports
something never existed. Collapsing the middle feature state hides two opposite
failures: calling a live feature broken, and calling a removed one fine.

Each item is asked twice, neutrally and in a form that assumes the feature
works, and each question is sampled N_SAMPLES times. The headline is stability:
the library's temperature setting does not reach the provider, so a single
response is one draw from a distribution rather than a measurement.

Run locally with `python task.py` to validate before pushing.
"""

import os
import sys
from collections import Counter

import kaggle_benchmarks as kbench

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classify import (  # noqa: E402
    CURRENT,
    DENIES_EXISTENCE,
    DEPRECATED_LIVE,
    REMOVED,
    UNCLEAR,
    classify,
)
from items import ITEMS  # noqa: E402

# The proxy reserves worst-case cost per call from this limit and refuses the
# call when the reservation exceeds the remaining daily quota. Unset, one call
# to a large model can reserve several dollars against a $10 cap. 1024 was too
# low: a reasoning model spent the whole budget thinking and returned nothing.
MAX_TOKENS = 4096

# prompt() advertises temperature=0, but actors/llms.py sends
# `temperature if self.support_temperature else None` and support_temperature
# defaults to False, so the provider default applies. Runs are not
# deterministic, and one sample per question is not a measurement.
N_SAMPLES = 3

PREAMBLE = (
    "Answer in at most three sentences. If the feature you are asked about is "
    "removed, deprecated, or current, say so explicitly."
)

# The Kaggle proxy fronts every vendor with one OpenAI-compatible
# Completions.create(), so the native Google spelling max_output_tokens is
# rejected even for Gemini. Try both OpenAI spellings, old and new.
_CAP_KEYS = ("max_tokens", "max_completion_tokens")
_WORKING_CAP_KEY: list = []  # negotiated once, then reused


def _ask(llm, question):
    """One question, one isolated conversation.

    prompt() accumulates history by default, which breaks this two ways. Items
    would see each other's answers, making scores depend on item order. And an
    empty response stays in history, so the next call ships a message with no
    parts and the provider rejects it, killing the run.
    """
    text = f"{PREAMBLE}\n\n{question}"

    def _send():
        if _WORKING_CAP_KEY:
            return llm.prompt(
                text, extra_api_params={_WORKING_CAP_KEY[0]: MAX_TOKENS}
            )

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

        # Stop rather than spend the daily quota a few uncapped calls at a time.
        raise RuntimeError(
            f"Could not set an output-token cap: none of {_CAP_KEYS} was "
            "accepted. Running uncapped risks the proxy reserving several "
            "dollars per call. Inspect inspect.signature(llm.prompt) first."
        )

    with kbench.chats.new("item"):
        try:
            return _send() or ""
        except Exception as exc:
            if "parts field" in str(exc) or "message=None" in str(exc):
                print(f"  empty response, scored UNCLEAR: {str(exc)[:90]}")
                return ""
            raise


@kbench.task(name="deprecation_state")
def deprecation_state(llm) -> dict:
    """Score every item on both question forms. Returns a flat result dict."""
    rows = []

    for item in ITEMS:
        for form in ("knowledge", "premise"):
            question = item.knowledge_q if form == "knowledge" else item.premise_q

            samples = []
            for _ in range(N_SAMPLES):
                response = _ask(llm, question)
                samples.append((classify(response), response))

            verdicts = [p for p, _ in samples]
            tally = Counter(verdicts)
            modal, modal_n = tally.most_common(1)[0]

            rows.append(
                {
                    "item": item.id,
                    "vendor": item.vendor,
                    "bucket": item.bucket,
                    "form": form,
                    "truth": item.truth,
                    "predicted": modal,
                    "correct": modal == item.truth,
                    # 1.0 = same answer every sample. 0.333 at N=3 = three
                    # different answers.
                    "consistency": round(modal_n / len(verdicts), 4),
                    "verdicts": verdicts,
                    "distinct_verdicts": len(tally),
                    # CURRENT and REMOVED are opposites, not degrees.
                    "self_contradictory": bool(
                        {CURRENT, REMOVED} <= set(verdicts)
                        or {CURRENT, DENIES_EXISTENCE} <= set(verdicts)
                    ),
                    # Denied on every sample: the model's training almost
                    # certainly predates the feature, so scoring it against
                    # truth measures the cutoff, not deprecation awareness.
                    "cutoff_bound": all(v == DENIES_EXISTENCE for v in verdicts),
                    "response": samples[0][1],
                    "all_responses": [r for _, r in samples],
                }
            )

    scored = [r for r in rows if r["predicted"] != UNCLEAR]
    correct = [r for r in rows if r["correct"]]

    # missed_removal hands you code that cannot run. over_warn sends you on a
    # migration you did not need. Scored as one number, a uniformly pessimistic
    # model looks good.
    missed_removal = [
        r for r in rows
        if r["truth"] == REMOVED and r["predicted"] in (CURRENT, DEPRECATED_LIVE)
    ]
    over_warned = [
        r for r in rows
        if r["truth"] in (DEPRECATED_LIVE, CURRENT) and r["predicted"] == REMOVED
    ]
    denied = [r for r in rows if r["predicted"] == DENIES_EXISTENCE]

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

    engaged = [r for r in rows if not r["cutoff_bound"]]

    kbench.assertions.assert_true(
        len(scored) >= len(rows) * 0.6,
        expectation=(
            "At least 60 percent of responses should be classifiable. A high "
            "UNCLEAR rate means the prompt or classifier needs work, not that "
            "the model failed."
        ),
    )

    return {
        # Stability. mean_consistency_engaged is the headline: a model that
        # denies everything scores 1.0 on the unfiltered version while
        # answering nothing, so read it with n_engaged.
        "mean_consistency_engaged": round(
            sum(r["consistency"] for r in engaged) / max(1, len(engaged)), 4
        ),
        "n_engaged": len(engaged),
        "mean_consistency": round(
            sum(r["consistency"] for r in rows) / len(rows), 4
        ) if rows else 0.0,
        "unstable_rate": rate([r for r in rows if r["consistency"] < 1.0], rows),
        "self_contradictory_rate": rate(
            [r for r in rows if r["self_contradictory"]], rows
        ),
        "n_samples_per_question": N_SAMPLES,

        "cutoff_bound_rate": rate([r for r in rows if r["cutoff_bound"]], rows),
        "denies_existence_rate": rate(denied, rows),

        "accuracy": rate(correct, rows),
        # Excludes UNCLEAR and the cutoff artefact. Reported alongside raw
        # accuracy, not instead of it: dropping inconvenient rows would flatter
        # every model with an early cutoff.
        "accuracy_committed": rate(
            [r for r in scored
             if r["correct"] and r["predicted"] != DENIES_EXISTENCE],
            [r for r in scored if r["predicted"] != DENIES_EXISTENCE],
        ),
        "accuracy_knowledge": by_form["knowledge"],
        "accuracy_premise": by_form["premise"],
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
