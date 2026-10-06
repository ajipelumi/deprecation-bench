# Deprecation state benchmark

Does a model know what still works?

A Kaggle Benchmarks task for the [DEV Kaggle Benchmarking Challenge](https://dev.to/devteam/join-the-kaggle-benchmarking-challenge-2500-in-prizes-for-five-winners-18ml).

## The idea

Most evals of this shape ask a binary question: is this feature deprecated or not.
Real API surfaces have **three** states, and the middle one is where the
interesting failures live.

| State | Meaning | Example |
| --- | --- | --- |
| `REMOVED` | End of life has passed. Using it fails. | Cloudflare Service Key auth, gone 30 Sep 2026 |
| `DEPRECATED_LIVE` | Announced deprecated, still functional. | `foundation_dns`, functional until 23 Nov 2026 |
| `CURRENT` | The documented replacement. | Cloudflare API Tokens |

Collapsing the middle state hides two opposite failures:

- **Missed removal.** Calling a removed feature current. The dangerous one: the
  model hands you code that cannot work.
- **Over-warning.** Calling a live feature removed. Cheaper but corrosive, because
  a model that cries deprecation at everything cannot be trusted when it is right.

Neither shows up in a binary score. A model can look good on a two-state eval by
being uniformly pessimistic.

## Two question forms

Each item is asked twice.

- **knowledge** — a neutral question. "Can I still authenticate using Service Key
  authentication?"
- **premise** — a request that assumes the feature works. "Show me how to
  authenticate using the `X-Auth-User-Service-Key` header."

The headline metric is `knowledge_premise_gap`: accuracy when asked neutrally,
minus accuracy when asked to just do the thing. A positive gap means the model
knows the feature is gone and writes the code anyway.

## Bucketing by how settled the fact is

Models have different training cutoffs, so a model that cannot know a late-2026
change is not worse than one that can. Treating that as a capability difference
would be a measurement error, so items carry a `bucket`:

| Bucket | Meaning |
| --- | --- |
| `established` | End of life well before 2026. Everything should know. |
| `recent` | End of life in H1 2026. |
| `fresh` | End of life Q3 2026 or later. Likely past some cutoffs. |

Reporting per bucket turns the confound into a designed axis. A model that does
not know a fact but still refuses a false premise is showing something different
from one that knows the answer, and the split is what makes that visible.

`gpt-5.4-cyber` is deliberately included: it shut down 1 Oct 2026, five days
before the item set was written. No model is expected to know. It isolates premise
resistance from knowledge.

## Layout

| File | What it is |
| --- | --- |
| `items.py` | Ground truth. Every item quotes its vendor source. Self-validating. |
| `classify.py` | Free-text to state classifier. Deterministic, 21 unit tests. |
| `task.py` | The Kaggle Benchmarks task. |
| `dryrun.py` | Metric sanity checks against synthetic models. No account needed. |

## Running the checks

No Kaggle account or API quota required for any of this:

```bash
python items.py      # validate ground truth against AS_OF
python classify.py   # 21 classifier unit tests
python dryrun.py     # metric behaviour against 5 synthetic models
```

`items.py` asserts that every `REMOVED` item has an end-of-life date **before**
`AS_OF` and every `DEPRECATED_LIVE` item has one **after** it. So the file fails
loudly when a date passes rather than going quietly stale. `AS_OF` is currently
`2026-10-06`; `foundation_dns` and the OpenAI items flip state in November 2026.

`dryrun.py` checks that the metrics detect behaviour they are supposed to detect:

| Synthetic model | Behaviour | Expected signal |
| --- | --- | --- |
| `oracle` | always right | accuracy 1.0, gap 0 |
| `paranoid` | everything is removed | high `over_warn_rate`, zero missed removals |
| `optimistic` | everything is current | high `missed_removal_rate`, zero over-warns |
| `forgetful` | knows, but complies anyway | **positive gap** |
| `mumbler` | never commits | `unclear_rate` 1.0 |

`forgetful` is the important one. It exhibits the target behaviour by
construction, so if the gap metric cannot see it there, the metric is broken.

## Running on Kaggle

The task needs a Kaggle notebook, since `kaggle_benchmarks` and the model proxy
come preinstalled there.

1. Open https://www.kaggle.com/benchmarks/tasks/new
2. Paste `items.py`, `classify.py` and `task.py` as cells, in that order (or
   attach them as a utility script).
3. Run. `%choose deprecation_state` before submitting, since the leaderboard
   accepts one task per notebook.

### Quota

The Kaggle model proxy **reserves the worst-case cost of each call** based on the
output-token limit, then refuses the call if the reservation exceeds the remaining
daily quota. With no limit set, published reports show reservations of $6.40 per
call for the largest models against a **$10 daily cap**: two calls before lockout.

`task.py` sets `max_completion_tokens = 1024` on every call for this reason. Do
not remove it.

Current item set: 12 items x 2 forms = **24 calls per model**. Comfortable.

### Leaderboard behaviour

The latest run replaces earlier results, and **an incomplete run overwrites a
complete one**. Do not re-run casually once a good result is recorded.

## Sources

All ground truth is vendor-published and quoted inline in `items.py`:

- [Cloudflare API deprecations](https://developers.cloudflare.com/fundamentals/api/reference/deprecations/) (49 dated entries)
- [OpenAI deprecations](https://developers.openai.com/api/docs/deprecations)

Announcement date and shutdown date are tracked as separate fields. Conflating
them is the exact error this benchmark measures, so the item set does not make it.
