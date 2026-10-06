"""Classify a model's free-text answer into REMOVED / DEPRECATED_LIVE / CURRENT.

This is the weakest link in the benchmark, so it is a separate module with its own
tests. If the classifier is wrong, every number downstream is wrong, and "I used an
LLM judge" invites the obvious objection. So: deterministic rules, tested against
hand-written responses, and an explicit UNCLEAR bucket rather than a forced guess.

Design choices worth defending in the writeup:

1. Three real states plus UNCLEAR. Forcing a fourth-state answer into one of three
   would inflate whichever bucket the tie-break favours.
2. Removal language outranks deprecation language. "This was deprecated and has
   now been removed" is a removal claim. Checking removal first gets that right.
3. Negation is handled explicitly. "is not deprecated" must not count as a
   deprecation claim, and that is the single most common way a naive keyword
   matcher goes wrong.
"""

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


if __name__ == "__main__":
    ok = _run_tests()
    raise SystemExit(0 if ok else 1)
