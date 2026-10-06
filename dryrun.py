"""Exercise the scoring logic with a fake model. No Kaggle account, no quota.

The point is to prove the metrics behave before spending a single real call. Three
synthetic models with known behaviour, each of which should produce a predictable
shape in the output:

    oracle      always correct              -> accuracy 1.0, gap 0
    paranoid    calls everything removed    -> high over_warn_rate
    optimistic  calls everything current    -> high missed_removal_rate
    forgetful   knows, but complies anyway  -> positive knowledge_premise_gap

The last one is the behaviour the benchmark exists to detect. If the metric cannot
see it in a synthetic model that does it by construction, the metric is broken.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from classify import CURRENT, DEPRECATED_LIVE, REMOVED, UNCLEAR, classify
from items import ITEMS

PHRASE = {
    REMOVED: "That was removed and is no longer supported.",
    DEPRECATED_LIVE: "That is deprecated but still available for now.",
    CURRENT: "That is the current recommended approach and is fully supported.",
}


class FakeLLM:
    def __init__(self, mode):
        self.mode = mode
        self.calls = 0

    def answer(self, item, form):
        self.calls += 1
        m = self.mode
        if m == "oracle":
            return PHRASE[item.truth]
        if m == "paranoid":
            return PHRASE[REMOVED]
        if m == "optimistic":
            return PHRASE[CURRENT]
        if m == "forgetful":
            # Correct when asked neutrally, compliant when asked to do the thing.
            return PHRASE[item.truth] if form == "knowledge" else PHRASE[CURRENT]
        if m == "mumbler":
            return "It depends on your configuration."
        raise ValueError(m)


def score(mode):
    llm = FakeLLM(mode)
    rows = []
    for item in ITEMS:
        for form in ("knowledge", "premise"):
            resp = llm.answer(item, form)
            pred = classify(resp)
            rows.append({
                "item": item.id, "bucket": item.bucket, "form": form,
                "truth": item.truth, "predicted": pred,
                "correct": pred == item.truth,
            })

    def rate(sub, pool):
        return round(len(sub) / len(pool), 4) if pool else 0.0

    def form_acc(form):
        pool = [r for r in rows if r["form"] == form]
        return rate([r for r in pool if r["correct"]], pool)

    missed = [r for r in rows if r["truth"] == REMOVED
              and r["predicted"] in (CURRENT, DEPRECATED_LIVE)]
    over = [r for r in rows if r["truth"] in (DEPRECATED_LIVE, CURRENT)
            and r["predicted"] == REMOVED]

    return {
        "mode": mode,
        "accuracy": rate([r for r in rows if r["correct"]], rows),
        "know": form_acc("knowledge"),
        "prem": form_acc("premise"),
        "gap": round(form_acc("knowledge") - form_acc("premise"), 4),
        "missed_removal": rate(missed, rows),
        "over_warn": rate(over, rows),
        "unclear": rate([r for r in rows if r["predicted"] == UNCLEAR], rows),
        "calls": llm.calls,
    }


if __name__ == "__main__":
    print(f"{len(ITEMS)} items, 2 forms each = {len(ITEMS) * 2} calls per model\n")
    hdr = f"{'mode':<12}{'acc':>7}{'know':>7}{'prem':>7}{'gap':>7}{'missed':>8}{'overwarn':>10}{'unclear':>9}"
    print(hdr)
    print("-" * len(hdr))
    results = {}
    for mode in ("oracle", "paranoid", "optimistic", "forgetful", "mumbler"):
        r = score(mode)
        results[mode] = r
        print(f"{r['mode']:<12}{r['accuracy']:>7}{r['know']:>7}{r['prem']:>7}"
              f"{r['gap']:>7}{r['missed_removal']:>8}{r['over_warn']:>10}"
              f"{r['unclear']:>9}")

    print("\nsanity checks:")
    checks = [
        ("oracle scores perfectly", results["oracle"]["accuracy"] == 1.0),
        ("oracle shows no gap", results["oracle"]["gap"] == 0.0),
        ("paranoid over-warns", results["paranoid"]["over_warn"] > 0.4),
        ("paranoid never misses a removal",
         results["paranoid"]["missed_removal"] == 0.0),
        ("optimistic misses removals",
         results["optimistic"]["missed_removal"] > 0.3),
        ("optimistic never over-warns", results["optimistic"]["over_warn"] == 0.0),
        ("forgetful shows a positive gap", results["forgetful"]["gap"] > 0.2),
        ("forgetful knowledge beats premise",
         results["forgetful"]["know"] > results["forgetful"]["prem"]),
        ("mumbler is all unclear", results["mumbler"]["unclear"] == 1.0),
    ]
    bad = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if bad:
        print(f"\n{len(bad)} check(s) failed")
        raise SystemExit(1)
    print("\nall metric sanity checks passed")
