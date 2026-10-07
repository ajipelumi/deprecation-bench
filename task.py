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

# The proxy reserves worst-case cost per call from the output-token limit. Left
# unset, a single call to an expensive model can reserve several dollars against
# a $10 daily quota and then be refused. This cap is not a nicety.
MAX_TOKENS = 4096

# Samples per question. One is not a measurement.
#
# The library's prompt() signature reads `temperature: float = 0`, but
# actors/llms.py sends `temperature if self.support_temperature else None`, and
# support_temperature defaults to False. So temperature is never sent and the
# provider's own default applies, which for Gemini is around 1.0. Two runs of
# this benchmark produced opposite answers to the same question: cf-audit-ssh
# went from "is current, you can still enable this" to "no, you can no longer
# create these rules".
#
# With sampling in play, a single response per item measures one draw from a
# distribution, not the model's knowledge. Repeating each question lets the
# instability itself be reported, which is the more interesting result: being
# wrong is ordinary, being wrong in both directions at once is not.
N_SAMPLES = 3

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
    """One question, one isolated conversation.

    `llm.prompt()` accumulates history by default, which would break this
    benchmark two ways. Items would see each other's answers, so a model primed
    by three deprecation questions would answer the fourth differently and the
    measurement would depend on item order. Worse, a reasoning model that spends
    its whole output budget thinking returns an empty message, and that empty
    turn stays in history: the next call then ships a message with no parts and
    the provider rejects the whole request. One empty response would kill the
    run.

    `kbench.chats.new()` gives each call a clean slate, so items are independent
    and an empty response costs one row instead of the run.
    """
    text = f"{PREAMBLE}\n\n{question}"

    def _send():
        if _WORKING_CAP_KEY:
            key = _WORKING_CAP_KEY[0]
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

        # No spelling accepted. Running uncapped risks the reservation trap, so
        # stop rather than quietly spending the daily quota a few calls at a time.
        raise RuntimeError(
            "Could not set an output-token cap: none of "
            f"{_CAP_KEYS} was accepted. Running uncapped risks the proxy "
            "reserving several dollars per call against the daily quota. "
            "Inspect `inspect.signature(llm.prompt)` and the proxy's accepted "
            "parameters before continuing."
        )

    with kbench.chats.new("item"):
        try:
            return _send() or ""
        except Exception as exc:
            # An empty completion from a reasoning model that spent its budget
            # thinking surfaces here. Record it as an unusable response rather
            # than losing every item after it.
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
                    # The modal verdict across samples, not a single draw.
                    "predicted": modal,
                    "correct": modal == item.truth,
                    # Fraction of samples agreeing with the mode. 1.0 means the
                    # model said the same thing every time; 0.34 at N=3 means it
                    # said three different things.
                    "consistency": round(modal_n / len(verdicts), 4),
                    "verdicts": verdicts,
                    "distinct_verdicts": len(tally),
                    # Did the model contradict itself about whether the feature
                    # works at all? CURRENT and REMOVED are not degrees of the
                    # same answer, they are opposites.
                    "self_contradictory": bool(
                        {CURRENT, REMOVED} <= set(verdicts)
                        or {CURRENT, DENIES_EXISTENCE} <= set(verdicts)
                    ),
                    # Every sample denied the feature ever existed. Almost
                    # certainly the model's training predates it, so scoring it
                    # against the item's truth measures the cutoff rather than
                    # anything about deprecation awareness. Reported separately
                    # rather than dropped: a model that cannot know is a fact
                    # about the lineup, not an inconvenience to hide.
                    "cutoff_bound": all(v == DENIES_EXISTENCE for v in verdicts),
                    "response": samples[0][1],
                    "all_responses": [r for _, r in samples],
                }
            )

    scored = [r for r in rows if r["predicted"] != UNCLEAR]
    correct = [r for r in rows if r["correct"]]

    # The failures worth naming separately.
    #
    # missed_removal is the dangerous one: the model hands you code for a feature
    # that no longer works.
    #
    # over_warn is its mirror: calling a live feature dead. Cheaper, but a model
    # that cries deprecation at everything cannot be trusted when it is right.
    #
    # denies_existence is neither. It means the model's training predates the
    # feature, so it reports the feature was never real. That is a cutoff
    # artefact, not a knowledge failure, and conflating it with either of the
    # above would misreport what the model got wrong.
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
        "denies_existence_rate": rate(denied, rows),
        "unclear_rate": rate([r for r in rows if r["predicted"] == UNCLEAR], rows),
        # Stability. mean_consistency is the headline: 1.0 means every question
        # got the same answer on every sample. Anything well below that means
        # single-sample numbers from this benchmark, or any like it, are noise.
        # Instability measured only over questions the model engaged with.
        # DENIES_EXISTENCE answers are driven by training cutoff, so a model
        # that consistently denies everything would otherwise look admirably
        # stable. This is the honest version of the headline number.
        "mean_consistency_engaged": round(
            sum(r["consistency"] for r in rows if not r["cutoff_bound"])
            / max(1, len([r for r in rows if not r["cutoff_bound"]])), 4
        ),
        "n_engaged": len([r for r in rows if not r["cutoff_bound"]]),
        "cutoff_bound_rate": rate([r for r in rows if r["cutoff_bound"]], rows),
        "mean_consistency": round(
            sum(r["consistency"] for r in rows) / len(rows), 4
        ) if rows else 0.0,
        "unstable_rate": rate([r for r in rows if r["consistency"] < 1.0], rows),
        "self_contradictory_rate": rate(
            [r for r in rows if r["self_contradictory"]], rows
        ),
        "n_samples_per_question": N_SAMPLES,
        # Accuracy over responses the model actually committed on, excluding
        # both UNCLEAR and the cutoff artefact. Reported alongside raw accuracy
        # rather than instead of it: dropping inconvenient rows and quoting the
        # remainder would flatter every model with an early cutoff.
        "accuracy_committed": rate(
            [r for r in scored if r["correct"] and r["predicted"] != DENIES_EXISTENCE],
            [r for r in scored if r["predicted"] != DENIES_EXISTENCE],
        ),
        "n_calls": len(rows),
        "rows": rows,
    }


deprecation_state.run(llm=kbench.llm)
