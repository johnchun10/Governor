"""``/doctor``: check credentials and send one tiny request per model, with hints when it fails."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable

from agent_core import Message, ModelInfo, ModelRequest, ProviderError

from .auth import AWS_PROVIDERS, status
from .providers import get_provider

Out = Callable[[str], None]

OK, FAIL, WARN = "✓", "✗", "!"


def list_inference_profiles() -> list[dict]:
    """Inference profiles visible to this account/region (needs bedrock:ListInferenceProfiles)."""
    from .providers.bedrock import aws_session

    client = aws_session().client("bedrock")
    out: list[dict] = []
    token = None
    while True:
        kwargs = {"maxResults": 1000, **({"nextToken": token} if token else {})}
        resp = client.list_inference_profiles(**kwargs)
        out += resp.get("inferenceProfileSummaries", [])
        token = resp.get("nextToken")
        if not token:
            return out


GEO_PREFIXES = {"us", "eu", "apac", "global", "us-gov", "jp", "au", "ca"}


def _base_model_id(model_id: str) -> str:
    """us.openai.gpt-6-luna -> openai.gpt-6-luna (the foundation model behind a profile)."""
    head, _, rest = model_id.partition(".")
    return rest if head in GEO_PREFIXES else model_id


def agreement_status(model_id: str) -> str | None:
    """Bedrock's view of whether this account may use the model (None if it can't be checked)."""
    from .providers.bedrock import aws_session

    try:
        r = aws_session().client("bedrock").get_foundation_model_availability(modelId=_base_model_id(model_id))
    except Exception:
        return None
    return (
        f"agreement={r['agreementAvailability']['status']} authorization={r['authorizationStatus']} "
        f"entitlement={r['entitlementAvailability']} region={r['regionAvailability']}"
    )


def _suggest(info: ModelInfo, available: Iterable[str]) -> list[str]:
    key = info.model_id.split(".", 1)[-1].split(":")[0]  # anthropic.claude-opus-5-5 -> claude-opus-5-5
    family = key.rsplit("-", 2)[0] if key.count("-") >= 2 else key
    return [a for a in available if key in a] or [a for a in available if family in a][:5]


def ping(info: ModelInfo) -> tuple[bool, str]:
    request = ModelRequest(
        model=info.model_id,
        messages=(Message.user("Reply with the single word OK."),),
        max_tokens=256,
        cache=False,
    )
    started = time.monotonic()
    try:
        resp = get_provider(info.provider).complete(request)
    except ProviderError as exc:
        return False, str(exc)
    reply = resp.message.text.strip()[:40] or f"(stop: {resp.stop_reason.value})"
    return (
        True,
        f"{reply!r} in {time.monotonic() - started:.1f}s, {resp.usage.input_tokens}+{resp.usage.output_tokens} tokens",
    )


def _check_aws(out: Out) -> tuple[bool, list[str]]:
    """AWS identity and visible inference profiles. Returns (ok, available profile ids)."""
    from .providers.bedrock import aws_session, explain_error

    session = aws_session()
    if os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        out(f"{WARN} using a Bedrock API key in {session.region_name}; skipping the AWS identity check")
    else:
        try:
            ident = session.client("sts").get_caller_identity()
            out(f"{OK} AWS credentials: {ident['Arn']} ({session.region_name})")
        except Exception as exc:
            out(f"{FAIL} {explain_error(exc)}")
            return False, []

    available: list[str] = []
    try:
        profiles = list_inference_profiles()
        available = sorted(p["inferenceProfileId"] for p in profiles if p.get("status", "ACTIVE") == "ACTIVE")
        out(f"{OK} {len(available)} inference profiles visible in {session.region_name}")
    except Exception as exc:
        out(f"{WARN} could not list inference profiles ({type(exc).__name__}); continuing with direct calls")
    return True, available


def doctor(provider: str, models: list[ModelInfo], out: Out = print) -> int:
    """Check credentials for ``provider``, then send one tiny request to each model. Returns failures."""
    available: list[str] = []
    if provider in AWS_PROVIDERS:
        ok, available = _check_aws(out)
        if not ok:
            return 1
    else:
        state = status(provider)
        out(f"{OK if state.ok else FAIL} {provider}: {state.detail}")
        if not state.ok:
            out("    fix: /login")
            return 1

    failures = 0
    for info in models:
        label = f"{info.alias}  ({info.model_id})"
        is_profile = info.model_id.partition(".")[0] in GEO_PREFIXES  # bare ids are on-demand models
        if info.provider == "bedrock" and is_profile and available and info.model_id not in available:
            hints = _suggest(info, available)
            out(f"{FAIL} {label}\n    not in this region's inference profiles. Similar: {', '.join(hints) or 'none'}")
            out("    fix: pick another model with /model, or re-point it in .governor/models.toml")
            failures += 1
            continue
        ok, detail = ping(info)
        out(f"{OK if ok else FAIL} {label}\n    {detail}")
        if not ok and info.provider == "bedrock" and (access := agreement_status(info.model_id)):
            out(f"    bedrock access status: {access}")
        failures += 0 if ok else 1
    out("all good" if not failures else f"{failures} model(s) failed; see hints above")
    return failures
