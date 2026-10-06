# %%
"""Deprecation state benchmark: does the model know what still works?

Three states, not two. A feature can be REMOVED (end of life passed),
DEPRECATED_LIVE (announced deprecated, still functional), or CURRENT. Collapsing
the middle state hides two opposite failures: calling a removed feature current
(the model hands you code that cannot work) and calling a live feature removed
(a model that cries deprecation at everything cannot be trusted when it is right).

Each item is asked twice: neutrally, and in a form that assumes the feature works.
The gap between the two is the headline metric. A model can know a feature is gone
and write the code anyway.

Ground truth is vendor-published and quoted inline. Announcement date and shutdown
date are separate fields, because conflating them is the exact error being measured.

Sources:
  https://developers.cloudflare.com/fundamentals/api/reference/deprecations/
  https://developers.openai.com/api/docs/deprecations
"""

import re
from dataclasses import dataclass, field

import kaggle_benchmarks as kbench


# ------------------------------------------------------------------------
# GROUND TRUTH
# ------------------------------------------------------------------------


from dataclasses import dataclass, field

AS_OF = "2026-10-06"

REMOVED = "REMOVED"
DEPRECATED_LIVE = "DEPRECATED_LIVE"
CURRENT = "CURRENT"

CF_DEPRECATIONS = "https://developers.cloudflare.com/fundamentals/api/reference/deprecations/"
OAI_DEPRECATIONS = "https://developers.openai.com/api/docs/deprecations"


@dataclass
class Item:
    id: str
    vendor: str
    feature: str
    truth: str
    # Neutral question. Does the model know the state?
    knowledge_q: str
    # Question that asserts the feature is usable. Does the model push back?
    premise_q: str
    replacement: str
    announced: str
    end_of_life: str
    bucket: str
    source: str
    quote: str
    notes: str = ""
    aliases: list = field(default_factory=list)


# Buckets by how settled the fact is, relative to AS_OF.
#   established  end of life well before 2026, every model should know
#   recent       end of life in H1 2026
#   fresh        end of life Q3 2026 or later, likely past some training cutoffs
ITEMS = [
    # ---------- REMOVED ----------
    Item(
        id="cf-service-key",
        vendor="Cloudflare",
        feature="Service Key authentication (X-Auth-User-Service-Key header)",
        truth=REMOVED,
        knowledge_q=(
            "Can I still authenticate to the Cloudflare API using Service Key "
            "authentication? Answer in one short paragraph."
        ),
        premise_q=(
            "Show me how to authenticate a request to the Cloudflare API using "
            "the X-Auth-User-Service-Key header."
        ),
        replacement="API Tokens",
        announced="2026-03-19",
        end_of_life="2026-09-30",
        bucket="fresh",
        source=CF_DEPRECATIONS,
        quote=(
            "Service Key authentication for the Cloudflare API is deprecated and "
            "will be removed on September 30, 2026."
        ),
        aliases=["service key", "x-auth-user-service-key"],
    ),
    Item(
        id="cf-audit-ssh",
        vendor="Cloudflare",
        feature='Gateway rules with action: "audit_ssh"',
        truth=REMOVED,
        knowledge_q=(
            'Can I still create Cloudflare Gateway rules with action "audit_ssh"? '
            "Answer in one short paragraph."
        ),
        premise_q=(
            'Write the API call to create a Cloudflare Gateway rule with '
            'action "audit_ssh".'
        ),
        replacement="SSH with Access for Infrastructure",
        announced="2025-11-03",
        end_of_life="2026-07-15",
        bucket="recent",
        source=CF_DEPRECATIONS,
        quote=(
            "On July 15, 2026, all remaining Audit SSH rules will stop working."
        ),
        aliases=["audit_ssh", "audit ssh"],
    ),
    Item(
        id="cf-dns-type-change",
        vendor="Cloudflare",
        feature="Changing the type of an existing DNS record via the API",
        truth=REMOVED,
        knowledge_q=(
            "Can I change the type of an existing Cloudflare DNS record via the "
            "API, for example from A to CNAME? Answer in one short paragraph."
        ),
        premise_q=(
            "Show me the Cloudflare API call to change an existing DNS record "
            "from type A to type CNAME, keeping the same record id."
        ),
        replacement="delete and recreate the record",
        announced="2026-01-23",
        end_of_life="2026-06-30",
        bucket="recent",
        source=CF_DEPRECATIONS,
        quote=(
            "Changing the type of an existing DNS record via the API is deprecated "
            "and will no longer be supported after June 30, 2026."
        ),
        aliases=["record type", "delete and recreate"],
    ),
    Item(
        id="oai-realtime-beta",
        vendor="OpenAI",
        feature="Realtime API Beta interface",
        truth=REMOVED,
        knowledge_q=(
            "Is the OpenAI Realtime API Beta interface still available? Answer in "
            "one short paragraph."
        ),
        premise_q=(
            "Show me how to connect to the OpenAI Realtime API Beta interface."
        ),
        replacement="the GA Realtime API",
        announced="2025-09-15",
        end_of_life="2026-05-12",
        bucket="recent",
        source=OAI_DEPRECATIONS,
        quote=(
            "The interfaces in the Realtime beta API and the released GA API have "
            "a few key differences."
        ),
        aliases=["realtime", "ga"],
    ),
    Item(
        id="oai-cyber-model",
        vendor="OpenAI",
        feature="gpt-5.4-cyber model",
        truth=REMOVED,
        knowledge_q=(
            "Is the gpt-5.4-cyber model still available in the OpenAI API? Answer "
            "in one short paragraph."
        ),
        premise_q=(
            "Write a Python call to the OpenAI API using the gpt-5.4-cyber model."
        ),
        replacement="the most capable cyber model available to you",
        announced="2026-09-11",
        end_of_life="2026-10-01",
        bucket="fresh",
        source=OAI_DEPRECATIONS,
        quote="Shutdown date October 1, 2026.",
        notes=(
            "Shut down five days before AS_OF. No model is expected to know this. "
            "Kept deliberately: it isolates premise resistance from knowledge."
        ),
        aliases=["gpt-5.4-cyber", "cyber"],
    ),

    # ---------- DEPRECATED_LIVE ----------
    # These are the controls. They are not invented: the vendor states in plain
    # words that the feature still works. A model that cries "removed" here is
    # over-warning, which is a real failure mode and one no leaderboard reports.
    Item(
        id="cf-foundation-dns",
        vendor="Cloudflare",
        feature="foundation_dns boolean in DNS settings endpoints",
        truth=DEPRECATED_LIVE,
        knowledge_q=(
            "Can I still read and write the foundation_dns boolean on the "
            "Cloudflare DNS settings endpoints? Answer in one short paragraph."
        ),
        premise_q=(
            "Show me the Cloudflare API call that sets foundation_dns to true on "
            "a zone."
        ),
        replacement='nameservers.type: "cloudflare.advanced"',
        announced="2026-07-27",
        end_of_life="2026-11-23",
        bucket="fresh",
        source=CF_DEPRECATIONS,
        quote=(
            "The foundation_dns boolean remains available as a compatibility alias "
            "during this transition. You can continue to read and write it."
        ),
        notes="End of life is 2026-11-23, after AS_OF. Still functional today.",
        aliases=["foundation_dns", "cloudflare.advanced"],
    ),
    Item(
        id="cf-zone-settings-batch",
        vendor="Cloudflare",
        feature="Zone Settings Batch API (GET/PATCH /zones/{id}/settings)",
        truth=DEPRECATED_LIVE,
        knowledge_q=(
            "Can I still use the Cloudflare batch zone settings endpoints, "
            "GET and PATCH /zones/{zone_id}/settings? Answer in one short "
            "paragraph."
        ),
        premise_q=(
            "Show me how to read several Cloudflare zone settings in one call "
            "using PATCH /zones/{zone_id}/settings."
        ),
        replacement="per-setting endpoints /zones/{zone_id}/settings/{setting_id}",
        announced="2025-04-23",
        end_of_life="2027-03-31",
        bucket="established",
        source=CF_DEPRECATIONS,
        quote=(
            "Integrations that read or edit multiple settings in a single call "
            "must migrate to per-setting requests before March 31, 2027 to ensure "
            "uninterrupted service."
        ),
        notes="End of life 2027. Comfortably live. Strong control item.",
        aliases=["per-setting", "setting_id"],
    ),
    Item(
        id="oai-reusable-prompts",
        vendor="OpenAI",
        feature="Reusable prompts and the v1/prompts API",
        truth=DEPRECATED_LIVE,
        knowledge_q=(
            "Is the OpenAI v1/prompts API for reusable prompts still available? "
            "Answer in one short paragraph."
        ),
        premise_q=(
            "Show me how to create a reusable prompt with the OpenAI v1/prompts "
            "API."
        ),
        replacement="move reusable prompt content into your application code",
        announced="2026-06-03",
        end_of_life="2026-11-30",
        bucket="fresh",
        source=OAI_DEPRECATIONS,
        quote=(
            "The v1/prompts API and reusable prompt objects are scheduled to shut "
            "down."
        ),
        notes=(
            "Announced 2026-06-03 but shutdown is 2026-11-30, after AS_OF. This is "
            "the item where the announced-vs-shutdown confusion shows up most."
        ),
        aliases=["v1/prompts", "prompts api"],
    ),
    Item(
        id="oai-agent-builder",
        vendor="OpenAI",
        feature="Agent Builder",
        truth=DEPRECATED_LIVE,
        knowledge_q=(
            "Is OpenAI Agent Builder still available to use? Answer in one short "
            "paragraph."
        ),
        premise_q="Walk me through building an agent with OpenAI Agent Builder.",
        replacement="Agents SDK or ChatGPT Workspace Agents",
        announced="2026-06-03",
        end_of_life="2026-11-30",
        bucket="fresh",
        source=OAI_DEPRECATIONS,
        quote="ChatKit remains available.",
        notes="Shutdown 2026-11-30, after AS_OF.",
        aliases=["agents sdk", "workspace agents"],
    ),

    # ---------- CURRENT ----------
    # The replacements. A model that warns about these is manufacturing a
    # deprecation, which is the mirror failure of missing a real one.
    Item(
        id="cf-api-tokens",
        vendor="Cloudflare",
        feature="API Tokens",
        truth=CURRENT,
        knowledge_q=(
            "Are Cloudflare API Tokens the current recommended way to "
            "authenticate to the Cloudflare API? Answer in one short paragraph."
        ),
        premise_q=(
            "Show me how to authenticate a request to the Cloudflare API using "
            "an API Token."
        ),
        replacement="n/a, this is the current approach",
        announced="",
        end_of_life="",
        bucket="established",
        source=CF_DEPRECATIONS,
        quote=(
            "Create an API Token with the appropriate permissions for your use "
            "case. API Tokens support fine-grained scoping, expiration, and "
            "revocation."
        ),
        aliases=["api token", "bearer"],
    ),
    Item(
        id="cf-per-setting",
        vendor="Cloudflare",
        feature="Per-setting zone endpoints /zones/{id}/settings/{setting_id}",
        truth=CURRENT,
        knowledge_q=(
            "Is GET /zones/{zone_id}/settings/{setting_id} the current way to read "
            "a single Cloudflare zone setting? Answer in one short paragraph."
        ),
        premise_q=(
            "Show me how to read one Cloudflare zone setting using "
            "GET /zones/{zone_id}/settings/{setting_id}."
        ),
        replacement="n/a, this is the current approach",
        announced="",
        end_of_life="",
        bucket="established",
        source=CF_DEPRECATIONS,
        quote=(
            "Get zone setting GET /zones/{zone_id}/settings/{setting_id}"
        ),
        aliases=["setting_id"],
    ),
    Item(
        id="cf-advanced-nameservers",
        vendor="Cloudflare",
        feature='nameservers.type: "cloudflare.advanced"',
        truth=CURRENT,
        knowledge_q=(
            'Is nameservers.type set to "cloudflare.advanced" the current way to '
            "configure Advanced Nameservers on Cloudflare? Answer in one short "
            "paragraph."
        ),
        premise_q=(
            'Show me the Cloudflare API call that configures Advanced Nameservers '
            'using nameservers.type "cloudflare.advanced".'
        ),
        replacement="n/a, this is the current approach",
        announced="",
        end_of_life="",
        bucket="fresh",
        source=CF_DEPRECATIONS,
        quote=(
            'Use nameservers.type: "cloudflare.advanced" to configure Advanced '
            "Nameservers instead."
        ),
        notes=(
            "Rollout began 2026-10-26 per the vendor page, which is after AS_OF. "
            "Treated as CURRENT because it is the documented forward path and the "
            "API accepts it for entitled accounts during rollout. Flagged here "
            "because it is the softest ground truth in the set."
        ),
        aliases=["cloudflare.advanced", "advanced nameservers"],
    ),
]


def by_truth(state):
    return [i for i in ITEMS if i.truth == state]


def summary():
    from collections import Counter
    t = Counter(i.truth for i in ITEMS)
    b = Counter(i.bucket for i in ITEMS)
    v = Counter(i.vendor for i in ITEMS)
    return {"total": len(ITEMS), "by_truth": dict(t), "by_bucket": dict(b),
            "by_vendor": dict(v)}


# ------------------------------------------------------------------------
# CLASSIFIER
# ------------------------------------------------------------------------


import re

REMOVED = "REMOVED"
DEPRECATED_LIVE = "DEPRECATED_LIVE"
CURRENT = "CURRENT"
UNCLEAR = "UNCLEAR"

# Claims the feature is gone / no longer works.
_REMOVED_PAT = [
    r"\b(?:was|has been|have been|is|are|been)\s+(?:fully\s+|completely\s+|now\s+)?removed\b",
    r"\bno longer (?:available|supported|works?|work|accepted|possible|exists?)\b",
    r"\b(?:has|have|was|is)\s+(?:been\s+)?(?:shut down|sunset|retired|discontinued)\b",
    r"\breached (?:its )?end[- ]of[- ]life\b",
    r"\b(?:end of life|eol) (?:has )?(?:passed|already passed)\b",
    r"\bcannot (?:be used|use|still use|authenticate|create)\b",
    r"\bnot (?:available|supported) (?:any ?more|anymore|any longer)\b",
    r"\bthis (?:endpoint|api|feature|model|header|method) (?:is|was) gone\b",
    r"\bstopped working\b",
    r"\bwill (?:fail|be rejected|return an error)\b",
]

# Claims the feature is deprecated but still usable.
_DEPRECATED_PAT = [
    r"\bdeprecated\b",
    r"\bscheduled (?:to|for) (?:shut down|shutdown|removal|be removed)\b",
    r"\bwill be removed\b",
    r"\bwill (?:no longer|cease to) (?:be|work|function)\b",
    r"\bslated for removal\b",
    r"\bsunsetting\b",
    r"\bbeing phased out\b",
    r"\bcompatibility alias\b",
    r"\blegacy\b",
]

# Claims it still works / is fine.
_LIVE_PAT = [
    r"\b(?:still|currently) (?:available|supported|works?|work|functional|valid|accepted)\b",
    r"\byes,? (?:you can|it is|it works|this works)\b",
    r"\bcontinues? to (?:work|function|be supported)\b",
    r"\bremains? (?:available|supported|functional)\b",
    r"\byou can (?:still|continue to)\b",
    r"\b(?:is|are)\s+(?:the\s+)?(?:current|recommended|preferred)\b",
    r"\bfully supported\b",
    r"\bno(?:t)? deprecated\b",
    r"\bworks? (?:fine|as expected|normally)\b",
]

# Negation guards. If a deprecation/removal word sits inside one of these, the
# claim is the opposite of what the bare keyword suggests.
_NEG_PREFIX = r"(?:is|are|was|were|has|have|been|it|this|that|which|and|but)?\s*"
_NEGATED_DEPRECATION = [
    r"\bnot\s+(?:been\s+)?deprecated\b",
    r"\bn't\s+(?:been\s+)?deprecated\b",
    r"\bno\s+deprecation\b",
    r"\bisn't deprecated\b",
    r"\bnot\s+(?:been\s+)?removed\b",
    r"\bn't\s+(?:been\s+)?removed\b",
    r"\bnot\s+(?:yet\s+)?(?:been\s+)?(?:shut down|retired|discontinued)\b",
]


def _any(pats, text):
    return any(re.search(p, text) for p in pats)


def _strip_negated(text):
    """Remove negated-deprecation spans so their keywords do not fire."""
    out = text
    for p in _NEGATED_DEPRECATION:
        out = re.sub(p, " __NEG__ ", out)
    return out


def classify(response: str) -> str:
    """Map a free-text response to a state. Deterministic, order matters."""
    if not response or not response.strip():
        return UNCLEAR

    t = response.lower()
    # Collapse whitespace so patterns spanning newlines still match.
    t = re.sub(r"\s+", " ", t)

    negation_present = _any(_NEGATED_DEPRECATION, t)
    t_clean = _strip_negated(t)

    removed = _any(_REMOVED_PAT, t_clean)
    deprecated = _any(_DEPRECATED_PAT, t_clean)
    live = _any(_LIVE_PAT, t_clean)

    # A removal claim is also, trivially, a deprecation claim. Check removal
    # first so "deprecated and has now been removed" lands on REMOVED.
    if removed and not live:
        return REMOVED
    if removed and live:
        # Mixed signals. Common real shape: "X was removed, use Y which is
        # still supported." The removal claim is about the asked-about feature,
        # so it wins, but only when deprecation language backs it up.
        return REMOVED if deprecated or not negation_present else UNCLEAR

    if deprecated and not negation_present:
        # Deprecated and the model did not claim it is gone.
        return DEPRECATED_LIVE

    if live:
        return CURRENT

    if negation_present:
        # "not deprecated" with no other signal is a currency claim.
        return CURRENT

    return UNCLEAR


# --------------------------------------------------------------------------
# Tests. Run `python classify.py` to check. These are hand-written responses in
# the shapes models actually produce, including the awkward ones.
# --------------------------------------------------------------------------
CASES = [
    # Clear removals
    ("Service Key authentication was removed on September 30, 2026. Use API Tokens.", REMOVED),
    ("No, that header is no longer supported. You should migrate to API Tokens.", REMOVED),
    ("This model has been shut down and is not available in the API.", REMOVED),
    ("That endpoint reached end-of-life in June 2026, so the call will fail.", REMOVED),
    ("You cannot use audit_ssh rules any more; they stopped working in July 2026.", REMOVED),
    # Removal plus a live replacement mentioned. Should still be REMOVED.
    ("The Realtime beta was removed in May 2026. The GA Realtime API is still available and works fine.", REMOVED),
    ("Service Keys are no longer accepted. API Tokens are the current recommended approach.", REMOVED),
    # Deprecated but live
    ("The foundation_dns boolean is deprecated but remains available as a compatibility alias.", DEPRECATED_LIVE),
    ("This is deprecated and will be removed on November 23, 2026, so migrate soon.", DEPRECATED_LIVE),
    ("The batch settings endpoints are legacy; migrate to per-setting requests before March 2027.", DEPRECATED_LIVE),
    ("v1/prompts is scheduled to shut down, though you can still use it today.", DEPRECATED_LIVE),
    # Current
    ("Yes, API Tokens are the current recommended way to authenticate.", CURRENT),
    ("That endpoint is fully supported and works as expected.", CURRENT),
    ("This is not deprecated. You can use it normally.", CURRENT),
    ("Yes, you can still read a single zone setting that way.", CURRENT),
    # Negation traps. A naive matcher calls all three DEPRECATED_LIVE.
    ("API Tokens are not deprecated and remain the recommended approach.", CURRENT),
    ("This has not been removed; it continues to work.", CURRENT),
    ("There is no deprecation notice for this endpoint.", CURRENT),
    # Unclear
    ("", UNCLEAR),
    ("Here is a code sample.", UNCLEAR),
    ("It depends on your account configuration.", UNCLEAR),
]


def _run_tests():
    failures = []
    for text, expected in CASES:
        got = classify(text)
        if got != expected:
            failures.append((text[:70], expected, got))
    print(f"classifier: {len(CASES) - len(failures)}/{len(CASES)} passed")
    for text, exp, got in failures:
        print(f"  FAIL expected={exp:<16} got={got:<16} {text!r}")
    return not failures


# ------------------------------------------------------------------------
# TASK
# ------------------------------------------------------------------------

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


def _token_cap(model_name: str) -> dict:
    """Output-token cap, in whichever spelling the provider expects.

    `prompt()` has no max-tokens parameter of its own; the limit goes through
    `extra_api_params`, and the key is provider-specific. Google models take
    `max_output_tokens`, OpenAI-compatible ones take `max_tokens`.

    This is not a tuning knob. The Kaggle proxy reserves the worst-case cost of
    every call from the output limit and refuses the call when that exceeds the
    remaining daily quota. Unset, a single call to a large model can reserve
    several dollars against a $10 cap.
    """
    name = (model_name or "").lower()
    if name.startswith("google/") or "gemini" in name or "gemma" in name:
        return {"max_output_tokens": MAX_TOKENS}
    return {"max_tokens": MAX_TOKENS}


def _ask(llm, question):
    model_name = getattr(llm, "name", "") or getattr(llm, "model", "") or ""
    text = f"{PREAMBLE}\n\n{question}"
    try:
        return llm.prompt(text, extra_api_params=_token_cap(str(model_name)))
    except Exception as exc:
        # A provider that rejects the key would otherwise kill the whole run
        # after burning most of it. Fall back uncapped and say so loudly.
        if "max_" not in str(exc):
            raise
        print(f"  token cap rejected ({exc}); retrying uncapped")
        return llm.prompt(text)


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


# First pass: one cheap model. Swap the slug to compare others.
# Registry keys carry a vendor prefix ("google/...", "meta/..."), not bare
# slugs. Run `print(sorted(kbench.llms.keys()))` to list what is available.
deprecation_state.run(llm=kbench.llms["google/gemini-2.5-flash"])
