# Run this as a SECOND cell, after the main benchmark cell has been run once.
# It reuses everything defined there and sweeps several models in one go.
#
# Each model costs len(ITEMS) * 2 * N_SAMPLES calls. At 13 items that is 78
# calls per model, roughly 16 cents on Gemini Flash. Larger models cost more.
# Watch the quota panel: the proxy reserves worst-case cost per call and
# refuses once the reservation exceeds what is left of the $10 daily budget.

import json
import traceback

SWEEP = [
    # One per vendor, mixing dated pins and @default tags. Dated pins have
    # knowable cutoffs, which matters when the finding is about what a model
    # does outside its training window.
    #
    # gemini-2.5-flash is listed twice on purpose, as the first and last entry.
    # Three separate runs of it produced unstable_rate 0.731, 0.654 and 0.769
    # with cutoff_bound_rate swinging 0.385 to 0.077, so the aggregate metrics
    # are themselves unstable across runs. Running the same model at both ends
    # of the sweep gives a within-sweep estimate of that noise floor, and no
    # cross-model difference smaller than the gap between those two entries
    # should be treated as real.
    "google/gemini-2.5-flash",
    "google/gemini-3.8-flash",
    "anthropic/claude-haiku-4-5@20251001",
    "anthropic/claude-sonnet-5@default",
    "openai/gpt-5.4-mini-2026-03-17",
    "qwen/qwen3-next-80b-a3b-instruct",
    "google/gemini-2.5-flash#repeat",
]

results = {}

for slug in SWEEP:
    # "#repeat" marks a deliberate re-run of a model already in the sweep. The
    # registry knows nothing about it, so strip it for lookup and keep it for
    # labelling and the output filename.
    model = slug.split("#")[0]

    if model not in kbench.llms:
        print(f"SKIP {slug}: not in registry")
        continue

    # The cap spelling is negotiated once per process and cached. Clearing it
    # lets each vendor negotiate its own.
    _WORKING_CAP_KEY.clear()

    print(f"\n=== {slug} ===", flush=True)
    try:
        run = deprecation_state.run(llm=kbench.llms[model])
        res = run.result if hasattr(run, "result") else run
    except Exception as exc:
        # One vendor failing should not cost the whole sweep.
        print(f"  FAILED: {type(exc).__name__}: {str(exc)[:160]}")
        traceback.print_exc(limit=1)
        continue

    res["model"] = model
    res["label"] = slug
    res["as_of"] = AS_OF
    results[slug] = res

    _fname = slug.replace("/", "-").replace("#", "-")
    with open(f"/kaggle/working/result-{_fname}.json", "w") as fh:
        json.dump(res, fh, indent=2)

    print(f"  engaged={res['n_engaged']}/{res['n_calls']} "
          f"cutoff={res['cutoff_bound_rate']} "
          f"consist={res['mean_consistency_engaged']} "
          f"unstable={res['unstable_rate']} "
          f"acc={res['accuracy']}", flush=True)

with open("/kaggle/working/sweep.json", "w") as fh:
    json.dump(results, fh, indent=2)

print("\n" + "=" * 112)
hdr = (f"{'model':<38}{'cutoff':>8}{'engag':>7}{'consist':>9}{'unstab':>8}"
       f"{'contra':>8}{'acc':>7}{'commit':>8}{'missed':>8}{'overwarn':>9}")
print(hdr)
print("-" * len(hdr))
for slug, r in sorted(results.items(), key=lambda kv: -kv[1]["mean_consistency_engaged"]):
    print(f"{slug:<38}{r['cutoff_bound_rate']:>8}{r['n_engaged']:>7}"
          f"{r['mean_consistency_engaged']:>9}{r['unstable_rate']:>8}"
          f"{r['self_contradictory_rate']:>8}{r['accuracy']:>7}"
          f"{r['accuracy_committed']:>8}{r['missed_removal_rate']:>8}"
          f"{r['over_warn_rate']:>9}")

print("\nper-item instability across models (how many models were inconsistent):")
if results:
    counts = {}
    for r in results.values():
        for row in r["rows"]:
            if row["consistency"] < 1.0:
                key = f"{row['item']}/{row['form']}"
                counts[key] = counts.get(key, 0) + 1
    for key, n in sorted(counts.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {n}/{len(results)}  {key}")
