"""Multi-model sweep. Run as a second cell, after the main cell has run once.

Each model costs len(ITEMS) * 2 * N_SAMPLES calls (78 at 13 items), roughly 13
cents on a Flash-class model. The proxy reserves worst-case cost per call and
refuses once the reservation exceeds the remaining daily budget, so watch the
quota panel on a long sweep.

gemini-2.5-flash runs twice, first and last. Three separate runs of it gave
unstable_rate 0.731 / 0.654 / 0.769, so the aggregate metrics move run to run.
The gap between the two entries is the noise floor, and no smaller cross-model
difference is a finding.
"""

import json
import traceback

SWEEP = [
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
    model = slug.split("#")[0]

    if model not in kbench.llms:
        print(f"SKIP {slug}: not in registry")
        continue

    _WORKING_CAP_KEY.clear()  # let each vendor negotiate its own cap spelling

    print(f"\n=== {slug} ===", flush=True)
    try:
        run = deprecation_state.run(llm=kbench.llms[model])
        res = run.result if hasattr(run, "result") else run
    except Exception as exc:
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
for slug, r in sorted(results.items(),
                      key=lambda kv: -kv[1]["mean_consistency_engaged"]):
    print(f"{slug:<38}{r['cutoff_bound_rate']:>8}{r['n_engaged']:>7}"
          f"{r['mean_consistency_engaged']:>9}{r['unstable_rate']:>8}"
          f"{r['self_contradictory_rate']:>8}{r['accuracy']:>7}"
          f"{r['accuracy_committed']:>8}{r['missed_removal_rate']:>8}"
          f"{r['over_warn_rate']:>9}")

print("\nper-item instability across models:")
counts = {}
for r in results.values():
    for row in r["rows"]:
        if row["consistency"] < 1.0:
            key = f"{row['item']}/{row['form']}"
            counts[key] = counts.get(key, 0) + 1
for key, n in sorted(counts.items(), key=lambda kv: -kv[1])[:15]:
    print(f"  {n}/{len(results)}  {key}")
