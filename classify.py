"""Classify a model's free-text answer about an API feature's state.

This is the weakest link in the benchmark, so it lives in its own module with its
own tests, and the tests run against real captured model output rather than
hand-written guesses. The first version passed 21 invented cases and then scored
58 percent of real Gemini responses UNCLEAR, almost entirely because of the bare
phrase "is **current**", which no pattern covered. Invented test cases encode
what you expect a model to say; they cannot tell you what it does say.

Four states, not three:

    REMOVED           claims the feature existed and no longer works
    DEPRECATED_LIVE   claims it is deprecated but still functional
    CURRENT           claims it is fine to use
    DENIES_EXISTENCE  claims the feature never existed at all

The fourth is not a nicety. A model whose training predates a feature says "that
is not a real OpenAI product", which is a different failure from "the old version
is still live". Scoring both as REMOVED would make a cutoff artefact look like
knowledge, and scoring both as CURRENT would hide a real refusal. It needs its
own bucket so the writeup can separate "wrong about the world" from "has not
heard of the world yet".

UNCLEAR remains for responses that genuinely commit to nothing.
"""

import re

REMOVED = "REMOVED"
DEPRECATED_LIVE = "DEPRECATED_LIVE"
CURRENT = "CURRENT"
DENIES_EXISTENCE = "DENIES_EXISTENCE"
UNCLEAR = "UNCLEAR"

# Markdown emphasis is stripped before matching, so patterns never need to
# account for **bold** or *italic* around the load-bearing word.
_EMPHASIS = re.compile(r"[*_`]+")

# ---------------------------------------------------------------------------
# Denies the feature ever existed. Checked FIRST: these responses often also
# contain currency language about some replacement ("there is no Agent Builder,
# but the Assistants API is current"), and the denial is the claim being made
# about the asked-about feature.
# ---------------------------------------------------------------------------
_DENIES_PAT = [
    r"\bis fictional\b",
    r"\bdoes not exist\b",
    r"\bdoesn't exist\b",
    r"\bnever existed\b",
    r"\bnot a real\b",
    r"\bno such (?:model|api|endpoint|feature|product|setting|header)\b",
    r"\bis not recognized\b",
    r"\bnot recognized (?:as|in)\b",
    r"\bdoes not correspond to any\b",
    r"\bthere is no (?:specific )?(?:product|model|api|endpoint|feature)\b",
    r"\bno (?:publicly )?(?:announced|documented) (?:distinct )?(?:api|product|model)\b",
    r"\bdoes not (?:currently )?offer a public product\b",
    r"\bis not (?:a )?current(?:ly)? (?:feature|way|functionality)\b",
    r"\bnot a current feature\b",
    r"\bis not a current\b",
    # Bare "is not current" with no trailing qualifier. Models end sentences
    # here constantly and the earlier patterns all required something after it.
    r"\bis\s+(?:\*\*)?not current(?:\*\*)?\s*[.,;]",
    r"\bis\s+not current\s*$",
    r"\bnon-?existent\b",
    r"\bis not a (?:standard|recognized|documented|distinct|named)\b",
    r"\bnot a (?:standard|recognized|documented|publicly branded)\b",
    r"\bnot a current, (?:distinct|user-configurable|directly settable)\b",
    r"\bis not current for\b",
    r"\bneither current, deprecated, nor removed\b",
    r"\bdoes not currently offer\b",
    r"\bare not current for\b",
    r"\bnot current for\b",
    r"\bis not current;",
    r"\bno publicly announced\b",
    r"\bnot (?:a )?(?:publicly )?(?:documented|announced)\b",
]

# ---------------------------------------------------------------------------
# Claims the feature is gone.
# ---------------------------------------------------------------------------
_REMOVED_PAT = [
    r"\b(?:was|has been|have been|is|are|been)\s+(?:fully\s+|completely\s+|now\s+)?removed\b",
    r"\bno longer (?:available|supported|works?|work|accepted|possible|offered)\b",
    r"\bno longer a\b",
    r"\bthere is no longer\b",
    r"\b(?:has|have|was|is)\s+(?:been\s+)?(?:shut down|sunset|retired|discontinued)\b",
    r"\breached (?:its )?end[- ]of[- ]life\b",
    r"\b(?:end of life|eol) (?:has )?(?:passed|already passed)\b",
    r"\bcannot (?:be used|use|still use|authenticate|create)\b",
    r"\bnot (?:available|supported) (?:any ?more|anymore|any longer)\b",
    r"\bstopped working\b",
    r"\bwill (?:fail|be rejected|return an error)\b",
    r"\bfeature has been removed\b",
]

# ---------------------------------------------------------------------------
# Claims it is deprecated but usable.
# ---------------------------------------------------------------------------
_DEPRECATED_PAT = [
    r"\bdeprecated\b",
    r"\bscheduled (?:to|for) (?:shut down|shutdown|removal|be removed)\b",
    r"\bwill be removed\b",
    r"\bwill (?:no longer|cease to) (?:be|work|function)\b",
    r"\bslated for removal\b",
    r"\bsunsetting\b",
    r"\bbeing phased out\b",
    r"\bcompatibility alias\b",
]

# ---------------------------------------------------------------------------
# Claims it works. The bare "is current" forms are the ones that matter: real
# models use them constantly and the first version of this file had none.
# ---------------------------------------------------------------------------
_LIVE_PAT = [
    r"\b(?:is|are|remains?|stays?)\s+(?:still\s+)?current\b",
    r"\bis current(?:ly)? (?:functionality|supported|available|the)\b",
    r"\bthis is current\b",
    r"\b(?:is|are)\s+(?:the\s+)?(?:current|recommended|preferred)\b",
    r"\b(?:still|currently) (?:available|supported|works?|work|functional|valid|accepted|enabled)\b",
    r"\byes,? (?:you can|it is|it works|this works|this is|the)\b",
    r"\bcontinues? to (?:work|function|be supported)\b",
    r"\bremains? (?:available|supported|functional|a supported)\b",
    r"\byou can (?:still|continue to)\b",
    r"\bfully supported\b",
    r"\bworks? (?:fine|as expected|normally)\b",
    r"\bactively supported\b",
]

# Negation guards: a deprecation/removal keyword inside one of these means the
# opposite of what the bare keyword suggests.
_NEGATED = [
    r"\bnot\s+(?:been\s+)?deprecated\b",
    r"\bn't\s+(?:been\s+)?deprecated\b",
    r"\bno\s+deprecation\b",
    r"\bnot\s+(?:been\s+)?removed\b",
    r"\bn't\s+(?:been\s+)?removed\b",
    r"\bnot\s+(?:yet\s+)?(?:been\s+)?(?:shut down|retired|discontinued)\b",
]


def _any(pats, text):
    return any(re.search(p, text) for p in pats)


def _normalise(response):
    t = response.lower()
    t = _EMPHASIS.sub("", t)          # drop markdown emphasis and code ticks
    t = re.sub(r"\s+", " ", t)        # collapse newlines so patterns span lines
    return t


def classify(response: str) -> str:
    """Map a free-text response to a state. Order matters; see module docstring."""
    if not response or not response.strip():
        return UNCLEAR

    t = _normalise(response)

    negated = _any(_NEGATED, t)
    t_clean = t
    for p in _NEGATED:
        t_clean = re.sub(p, " __neg__ ", t_clean)

    # 1. Denial of existence wins. These responses frequently also praise some
    #    replacement as current, and that currency claim is about a different
    #    thing than the feature asked about.
    if _any(_DENIES_PAT, t_clean):
        return DENIES_EXISTENCE

    removed = _any(_REMOVED_PAT, t_clean)
    deprecated = _any(_DEPRECATED_PAT, t_clean)
    live = _any(_LIVE_PAT, t_clean)

    # 2. Removal outranks deprecation: "deprecated and now removed" is a removal.
    if removed and not live:
        return REMOVED
    if removed and live:
        return REMOVED if deprecated or not negated else UNCLEAR

    # 3. Deprecated, not claimed gone.
    if deprecated and not negated:
        return DEPRECATED_LIVE

    if live:
        return CURRENT

    if negated:
        return CURRENT

    return UNCLEAR


# --------------------------------------------------------------------------
# Tests. Two suites: invented cases for edge shapes, and real captured output.
# --------------------------------------------------------------------------
SYNTHETIC_CASES = [
    ("Service Key authentication was removed on September 30, 2026. Use API Tokens.", REMOVED),
    ("No, that header is no longer supported. You should migrate to API Tokens.", REMOVED),
    ("That endpoint reached end-of-life in June 2026, so the call will fail.", REMOVED),
    ("The Realtime beta was removed in May 2026. The GA Realtime API is still available.", REMOVED),
    ("The foundation_dns boolean is deprecated but remains available as a compatibility alias.", DEPRECATED_LIVE),
    ("This is deprecated and will be removed on November 23, 2026, so migrate soon.", DEPRECATED_LIVE),
    ("v1/prompts is scheduled to shut down, though you can still use it today.", DEPRECATED_LIVE),
    ("Yes, API Tokens are the current recommended way to authenticate.", CURRENT),
    ("That endpoint is fully supported and works as expected.", CURRENT),
    ("This is not deprecated. You can use it normally.", CURRENT),
    ("API Tokens are not deprecated and remain the recommended approach.", CURRENT),
    ("This has not been removed; it continues to work.", CURRENT),
    ("That model is fictional and does not exist.", DENIES_EXISTENCE),
    ("There is no such endpoint in the Cloudflare API.", DENIES_EXISTENCE),
    ("", UNCLEAR),
    ("Here is a code sample.", UNCLEAR),
    ("It depends on your account configuration.", UNCLEAR),
]


def _run_tests():
    ok = True

    failures = [
        (text[:60], exp, classify(text))
        for text, exp in SYNTHETIC_CASES
        if classify(text) != exp
    ]
    print(f"synthetic: {len(SYNTHETIC_CASES) - len(failures)}/{len(SYNTHETIC_CASES)} passed")
    for text, exp, got in failures:
        print(f"  FAIL expected={exp:<17} got={got:<17} {text!r}")
    ok = ok and not failures

    try:
        from fixtures.gemini_2_5_flash import CASES as REAL
    except ImportError:
        try:
            from gemini_2_5_flash import CASES as REAL  # notebook: same namespace
        except ImportError:
            print("real-output fixtures not importable, skipping")
            return ok

    rfail = [
        (item, form, exp, classify(text))
        for item, form, text, exp in REAL
        if classify(text) != exp
    ]
    print(f"real output: {len(REAL) - len(rfail)}/{len(REAL)} passed")
    for item, form, exp, got in rfail:
        print(f"  FAIL {item}/{form} expected={exp:<17} got={got}")
    return ok and not rfail


if __name__ == "__main__":
    raise SystemExit(0 if _run_tests() else 1)
