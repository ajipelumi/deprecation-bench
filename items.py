"""Ground truth, every item quoting a vendor deprecation page.

    REMOVED          end of life passed, using it fails
    DEPRECATED_LIVE  announced, still functional
    CURRENT          the documented replacement

Announcement date and end-of-life date are separate fields. Conflating them is
the error this benchmark measures, so the item set does not make it.

`bucket` splits items by how settled the fact is, so a model that cannot know a
recent change is scored apart from one that should know an old one. Without it,
differing training cutoffs read as differing capability.
"""

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
        id="cf-registrar-api",
        vendor="Cloudflare",
        feature="Registrar API (GET /accounts/{id}/registrar/registrations)",
        truth=CURRENT,
        knowledge_q=(
            "Is GET /accounts/{account_id}/registrar/registrations the current "
            "Cloudflare API for listing domain registrations? Answer in one "
            "short paragraph."
        ),
        premise_q=(
            "Show me how to list Cloudflare domain registrations using "
            "GET /accounts/{account_id}/registrar/registrations."
        ),
        replacement="n/a, this is the current approach",
        announced="",
        end_of_life="",
        bucket="recent",
        source=CF_DEPRECATIONS,
        quote=(
            "Replacement: Registrar API ... List registrations - "
            "GET /accounts/{account_id}/registrar/registrations"
        ),
        notes=(
            "Named on the deprecation page as the replacement for the Legacy "
            "Registrar Domain Management API, which ended 27 Sep 2026. No "
            "pending rollout, unlike cf-advanced-nameservers."
        ),
        aliases=["registrar", "registrations"],
    ),
    Item(
        id="cf-access-for-infrastructure",
        vendor="Cloudflare",
        feature="SSH with Access for Infrastructure",
        truth=CURRENT,
        knowledge_q=(
            "Is SSH with Access for Infrastructure the current Cloudflare way "
            "to manage and audit SSH access? Answer in one short paragraph."
        ),
        premise_q=(
            "Show me how to audit SSH access in Cloudflare using Access for "
            "Infrastructure."
        ),
        replacement="n/a, this is the current approach",
        announced="",
        end_of_life="",
        bucket="recent",
        source=CF_DEPRECATIONS,
        quote=(
            "SSH with Access for Infrastructure is the replacement for managing "
            "and auditing SSH access."
        ),
        notes=(
            "Pairs with cf-audit-ssh: the model should call the old action dead "
            "and this one live. Getting both right needs the distinction, not "
            "just a blanket posture toward Cloudflare SSH features."
        ),
        aliases=["access for infrastructure", "infrastructure"],
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


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), indent=2))
    ids = [i.id for i in ITEMS]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    for i in ITEMS:
        assert i.truth in (REMOVED, DEPRECATED_LIVE, CURRENT), i.id
        assert i.quote, f"{i.id} has no source quote"
        assert i.knowledge_q and i.premise_q, i.id
        if i.truth == REMOVED:
            assert i.end_of_life and i.end_of_life < AS_OF, (
                f"{i.id} marked REMOVED but end_of_life {i.end_of_life} "
                f"is not before {AS_OF}")
        if i.truth == DEPRECATED_LIVE:
            assert i.end_of_life and i.end_of_life > AS_OF, (
                f"{i.id} marked DEPRECATED_LIVE but end_of_life "
                f"{i.end_of_life} is not after {AS_OF}")
    print("\nall ground-truth assertions passed")


# WITHDRAWN 2026-10-07: cf-advanced-nameservers. Scored CURRENT, but the vendor
# page says the rollout representing Advanced Nameservers with
# nameservers.type: "cloudflare.advanced" only BEGINS 26 Oct 2026, and until it
# reaches an account the API still returns "cloudflare.standard". Gemini said at
# consistency 1.0 that the value is returned, not configurable, which is
# defensible on today's page. Marking a model wrong against a fact that is not
# yet true is the error this benchmark exists to measure.
