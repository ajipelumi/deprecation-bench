"""Real responses from google/gemini-2.5-flash, captured 2026-10-06.

These are verbatim model outputs from the first full run, used as classifier test
cases. Hand-written test cases encode what I guessed models would say; these
encode what one actually said. The first version of the classifier scored 58
percent of them UNCLEAR, almost all because of the bare phrase "is **current**"
which no pattern covered.

Each tuple is (item_id, form, response_prefix, expected_classification).

The expected values are what a careful human reader would say the response
CLAIMS, which is not the same as whether the claim is true. The classifier's job
is to read the claim; scoring against ground truth happens afterwards.
"""

CURRENT = "CURRENT"
REMOVED = "REMOVED"
DEPRECATED_LIVE = "DEPRECATED_LIVE"
UNCLEAR = "UNCLEAR"
DENIES_EXISTENCE = "DENIES_EXISTENCE"

CASES = [
    # Bare "is **current**" with markdown bold. The single biggest miss.
    (
        "cf-service-key", "knowledge",
        "Yes, Service Key authentication is still **current** for the Cloudflare "
        "API. While it remains a supported method, it is considered an older "
        "approach and Cloudflare now strongly recommends using API Tokens for "
        "enhanced security and fine-grained control over permissions.",
        CURRENT,
    ),
    (
        "cf-audit-ssh", "knowledge",
        "Yes, the functionality to audit SSH commands in Cloudflare Gateway is "
        "**current**. You can still enable this feature by creating Gateway SSH "
        "policies and selecting the \"Log SSH commands\" action.",
        CURRENT,
    ),
    (
        "cf-audit-ssh", "premise",
        "The `audit_ssh` action for Cloudflare Gateway rules is **current**.\n"
        "To create such a rule, use `POST /accounts/{account_id}/gateway/rules`.",
        CURRENT,
    ),
    (
        "cf-dns-type-change", "knowledge",
        "Yes, this is **current** functionality. You can change the type of an "
        "existing Cloudflare DNS record via the API by performing an update (PUT) "
        "operation on the specific record, providing the new record type.",
        CURRENT,
    ),
    (
        "cf-foundation-dns", "knowledge",
        "Yes, the `foundation_dns` boolean on the Cloudflare DNS settings "
        "endpoints is **current**. You can still both read and write this setting "
        "using the respective GET and PATCH API endpoints for zone settings.",
        CURRENT,
    ),
    (
        "cf-per-setting", "premise",
        "This feature is **current**. To read a single Cloudflare zone setting, "
        "you would make a `GET` request to "
        "`https://api.cloudflare.com/client/v4/zones/{zone_id}/settings/{setting_id}`.",
        CURRENT,
    ),
    # Deprecation claims.
    (
        "cf-service-key", "premise",
        "The `X-Auth-User-Service-Key` header for Cloudflare API authentication is "
        "**deprecated**. It was used with User Service Keys, which have been "
        "superseded by more granular and secure API Tokens.",
        DEPRECATED_LIVE,
    ),
    # Removal claim.
    (
        "oai-reusable-prompts", "premise",
        "The OpenAI `v1/prompts` API feature has been removed. There is no longer "
        "a dedicated endpoint for creating reusable prompts directly within the "
        "OpenAI API ecosystem.",
        REMOVED,
    ),
    # Denies the thing ever existed. This is NOT a removal claim and NOT a
    # currency claim: it is a cutoff artefact, where the model's training
    # predates the feature entirely. Scoring it as anything else conflates
    # "I think the old version is still live" with "I have never heard of this".
    (
        "oai-cyber-model", "knowledge",
        "The `gpt-5.4-cyber` model is **not available** in the OpenAI API. This "
        "model name does not correspond to any publicly announced or accessible "
        "model offered by OpenAI. OpenAI's current model offerings include "
        "various versions of GPT-3.5 and GPT-4.",
        DENIES_EXISTENCE,
    ),
    (
        "oai-cyber-model", "premise",
        "The `gpt-5.4-cyber` model is fictional and does not currently exist as an "
        "OpenAI API offering. If it were available, a Python call would resemble "
        "`client.chat.completions.create(model=\"gpt-5.4-cyber\", ...)`.",
        DENIES_EXISTENCE,
    ),
    (
        "oai-realtime-beta", "knowledge",
        "The \"OpenAI Realtime API Beta interface,\" as a distinct and publicly "
        "documented product or feature, is not recognized as an official offering. "
        "OpenAI provides capabilities like streaming responses from its standard "
        "APIs for real-time applications, but not under that specific product name.",
        DENIES_EXISTENCE,
    ),
    (
        "oai-realtime-beta", "premise",
        "The \"OpenAI Realtime API Beta interface\" you're asking about is not "
        "current; there is no publicly announced or documented distinct API "
        "offering by OpenAI specifically named as such.",
        DENIES_EXISTENCE,
    ),
    (
        "oai-agent-builder", "knowledge",
        "There is no specific product officially named \"OpenAI Agent Builder.\" "
        "However, the ability to build AI agents using OpenAI's technologies is "
        "**current** and actively supported through the Assistants API and custom "
        "GPTs within ChatGPT.",
        DENIES_EXISTENCE,
    ),
    (
        "oai-agent-builder", "premise",
        "OpenAI does not currently offer a public product explicitly named "
        "\"OpenAI Agent Builder.\" However, you can build custom AI agents using "
        "ChatGPT's \"GPTs\" feature (for Plus subscribers) or programmatically via "
        "the OpenAI Assistant API.",
        DENIES_EXISTENCE,
    ),
    (
        "cf-advanced-nameservers", "knowledge",
        "No, `nameservers.type` set to `\"cloudflare.advanced\"` is not a current "
        "way to configure Advanced Nameservers on Cloudflare. This specific "
        "configuration syntax is not recognized in Cloudflare's public APIs or "
        "dashboard settings for managing nameservers.",
        DENIES_EXISTENCE,
    ),
    (
        "cf-advanced-nameservers", "premise",
        "The concept of configuring a zone's nameservers using "
        "`nameservers.type: cloudflare.advanced` is **not a current feature** in "
        "the Cloudflare API. Cloudflare automatically assigns its nameservers to a "
        "zone upon activation.",
        DENIES_EXISTENCE,
    ),
    (
        "cf-zone-settings-batch", "knowledge",
        "The Cloudflare API does not currently offer `GET /zones/{zone_id}/settings` "
        "or `PATCH /zones/{zone_id}/settings` as **batch** endpoints to manage all "
        "zone settings simultaneously. These specific paths are not current for "
        "retrieving or updating multiple settings at once.",
        DENIES_EXISTENCE,
    ),
    # Mixed: asserts current, qualifies heavily. Reads as a currency claim.
    (
        "cf-zone-settings-batch", "premise",
        "`PATCH /zones/{zone_id}/settings` is **current**, but it is used to "
        "*update* multiple zone settings, not to *read* them. To read several "
        "Cloudflare zone settings, you would use `GET /zones/{zone_id}/settings`, "
        "which returns all current zone settings.",
        CURRENT,
    ),
    # Says the setting is not publicly exposed. Not a deprecation claim, not a
    # denial that the name exists: it claims restricted access.
    (
        "cf-foundation-dns", "premise",
        "The `foundation_dns` setting is an **Enterprise feature** and is not "
        "directly exposed as a publicly modifiable API setting. It is typically "
        "managed internally by Cloudflare support or your account team based on "
        "your plan.",
        UNCLEAR,
    ),
    # Plain currency claim with a recommendation. Earlier version misread the
    # word "recommended" context and returned DEPRECATED_LIVE.
    (
        "cf-api-tokens", "knowledge",
        "Cloudflare API Tokens are the **current** recommended way to authenticate "
        "to the Cloudflare API. They provide significantly enhanced security by "
        "allowing granular permissions and scoped access.",
        CURRENT,
    ),
]
