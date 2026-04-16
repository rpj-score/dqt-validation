"""Source provenance scoring — layers 1 and 3 of the groundedness enrichment.

Layer 1: Score each returned source against a trust rubric using metadata
already on the artifact (credibility_score, credibility_tier, red_flags,
misinfo_risk, source_domain). No external API calls.

Layer 3: Cross-check frozen inputs vs returned sources — detect dropped inputs,
hallucinated (fabricated) sources, and credibility inflation.
"""

from __future__ import annotations

from typing import Any

from .models import JsonDict, RunArtifact, ValidationScenario


# Domains that are unconditionally trusted for Baguio civic analysis.
# Domains used by the eval's own frozen-document fixtures. URLs on these
# domains are authored by us and must never be flagged as "fabricated" by
# the cross-check — they may appear in the response because Node 3 (Qdrant
# recall) retrieves documents persisted from other scenario runs.
FIXTURE_DOMAINS = {
    "example.org",
    "example.com",
    "fake-authority.example.org",
}

TRUSTED_DOMAINS = {
    "gov.ph",
    "baguio.gov.ph",
    "pia.gov.ph",
    "inquirer.net",
    "rappler.com",
    "philstar.com",
    "mb.com.ph",
    "gmanetwork.com",
    "abs-cbn.com",
    "cnnphilippines.com",
    "pna.gov.ph",
    "sunstar.com.ph",
}


def _domain_from_url(url: Any) -> str:
    if not url or not isinstance(url, str):
        return "unknown"
    url = str(url).lower().rstrip("/")
    if "://" in url:
        url = url.split("://", 1)[1]
    host = url.split("/", 1)[0].split("?", 1)[0]
    parts = host.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


def _is_trusted_domain(domain: str) -> bool:
    domain = domain.lower()
    return any(domain == td or domain.endswith("." + td) for td in TRUSTED_DOMAINS)


def _is_fixture_domain(domain: str) -> bool:
    domain = domain.lower()
    return any(domain == fd or domain.endswith("." + fd) for fd in FIXTURE_DOMAINS)


def _extract_meta(doc: JsonDict, key: str, default: Any = None) -> Any:
    meta = doc.get("metadata") or {}
    return meta.get(key, default)


# --- Layer 1: per-source trust profile ------------------------------------

def score_source_trust(source: JsonDict) -> JsonDict:
    """Score a single returned source from SnapshotResponse.sources."""
    domain = _extract_meta(source, "source_domain") or _domain_from_url(source.get("url"))
    cred_score = float(_extract_meta(source, "credibility_score") or 0.5)
    cred_tier = _extract_meta(source, "credibility_tier") or "unknown"
    red_flags = _extract_meta(source, "red_flags") or []
    misinfo = _extract_meta(source, "misinfo_risk") or "unknown"
    verification = _extract_meta(source, "verification_status") or "unverified"
    tavily_sources = _extract_meta(source, "tavily_verified_sources") or []

    trust_level = "low"
    if cred_score >= 0.7 and not red_flags:
        trust_level = "high"
    elif cred_score >= 0.5:
        trust_level = "medium"

    breakdown = _extract_meta(source, "credibility_breakdown") or {}
    llm_reasoning = _extract_meta(source, "llm_reasoning") or ""

    return {
        "url": str(source.get("url") or ""),
        "title": str(source.get("title") or "")[:120],
        "domain": domain,
        "domain_trusted": _is_trusted_domain(domain),
        "credibility_score": round(cred_score, 3),
        "credibility_tier": cred_tier,
        "credibility_breakdown": dict(breakdown) if isinstance(breakdown, dict) else {},
        "llm_reasoning": str(llm_reasoning)[:500],
        "red_flags": list(red_flags) if isinstance(red_flags, list) else [],
        "misinfo_risk": misinfo,
        "verification_status": verification,
        "tavily_verified": bool(tavily_sources),
        "trust_level": trust_level,
    }


def build_source_trust_profile(artifact: RunArtifact) -> JsonDict:
    """Build a per-scenario source trust summary from SnapshotResponse.sources."""
    payload = artifact.snapshot_response or {}
    raw_sources = payload.get("sources") or [] if isinstance(payload, dict) else []
    if not raw_sources:
        return {
            "n_sources": 0,
            "sources": [],
            "summary": {"high": 0, "medium": 0, "low": 0, "flagged": 0, "trusted_domain": 0},
        }
    scored = [score_source_trust(s) for s in raw_sources if isinstance(s, dict)]
    has_reasoning = [s for s in scored if s.get("llm_reasoning")]
    has_breakdown = [s for s in scored if s.get("credibility_breakdown")]
    skeptical_reasoning = [
        s for s in has_reasoning
        if any(kw in s["llm_reasoning"].lower() for kw in (
            "not official", "lacks detail", "unverified", "low credib",
            "caution", "suspicious", "clickbait", "not established",
            "questionable", "unreliable", "no author", "anonymous",
        ))
    ]
    summary = {
        "high": sum(1 for s in scored if s["trust_level"] == "high"),
        "medium": sum(1 for s in scored if s["trust_level"] == "medium"),
        "low": sum(1 for s in scored if s["trust_level"] == "low"),
        "flagged": sum(1 for s in scored if s["red_flags"]),
        "trusted_domain": sum(1 for s in scored if s["domain_trusted"]),
        "has_llm_reasoning": len(has_reasoning),
        "has_credibility_breakdown": len(has_breakdown),
        "skeptical_reasoning_count": len(skeptical_reasoning),
    }
    return {
        "n_sources": len(scored),
        "sources": scored,
        "summary": summary,
        "agent_reasoning_excerpts": [
            {"url": s["url"], "reasoning": s["llm_reasoning"], "score": s["credibility_score"]}
            for s in skeptical_reasoning
        ],
    }


# --- Layer 3: frozen-input cross-check ------------------------------------

def _url_key(url: Any) -> str:
    """Normalize URL for comparison (strip scheme + trailing slash)."""
    if not url:
        return ""
    s = str(url).lower().rstrip("/")
    for prefix in ("https://", "http://"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s


def cross_check_sources(
    scenario: ValidationScenario, artifact: RunArtifact
) -> JsonDict:
    """Compare frozen inputs vs returned sources.

    Returns a dict with:
    - ``frozen_urls``: set of URLs provided to the agent
    - ``returned_urls``: set of URLs on the response
    - ``dropped``: URLs the agent was given but didn't cite
    - ``fabricated``: URLs on the response that weren't in frozen inputs
    - ``matched``: URLs that appear in both
    - ``credibility_inflation``: sources whose returned credibility_score is
      suspiciously higher than any plausible heuristic (> 0.9 on a source
      with red_flags or unknown domain)
    """
    frozen_urls = {_url_key(d.get("url")): d for d in scenario.frozen_documents if d.get("url")}
    payload = artifact.snapshot_response or {}
    returned_raw = payload.get("sources") or [] if isinstance(payload, dict) else []
    returned_urls = {_url_key(s.get("url")): s for s in returned_raw if isinstance(s, dict) and s.get("url")}

    frozen_set = set(frozen_urls.keys()) - {""}
    returned_set = set(returned_urls.keys()) - {""}

    matched = frozen_set & returned_set
    dropped = frozen_set - returned_set
    # URLs from our own fixture domains (example.org, etc.) are authored test
    # data that may surface via Qdrant recall from other scenarios. They are
    # "extra" but not "fabricated" — the agent didn't hallucinate a URL.
    raw_extra = returned_set - frozen_set
    fabricated = {url for url in raw_extra if not _is_fixture_domain(_domain_from_url(url))}
    extra_fixture = raw_extra - fabricated

    inflation: list[JsonDict] = []
    for url_key, src in returned_urls.items():
        meta = src.get("metadata") or {}
        cred = float(meta.get("credibility_score") or 0.5)
        flags = meta.get("red_flags") or []
        domain = meta.get("source_domain") or _domain_from_url(src.get("url"))
        if cred > 0.9 and (flags or not _is_trusted_domain(domain)):
            inflation.append({
                "url": str(src.get("url") or ""),
                "credibility_score": cred,
                "red_flags": flags,
                "domain": domain,
                "reason": "High credibility score despite red flags or untrusted domain",
            })

    return {
        "frozen_count": len(frozen_set),
        "returned_count": len(returned_set),
        "matched_count": len(matched),
        "dropped_urls": sorted(dropped),
        "fabricated_urls": sorted(fabricated),
        "extra_fixture_urls": sorted(extra_fixture),
        "matched_urls": sorted(matched),
        "credibility_inflation": inflation,
        "fabricated_source_detected": len(fabricated) > 0,
        "all_inputs_cited": len(dropped) == 0,
    }


# --- Combined provenance dict for RunArtifact.independent_grading ----------

def compute_source_provenance(
    scenario: ValidationScenario, artifact: RunArtifact
) -> JsonDict:
    """Compute the full provenance block for attachment to the artifact."""
    trust = build_source_trust_profile(artifact)
    xcheck = cross_check_sources(scenario, artifact)
    return {
        "trust_profile": trust,
        "cross_check": xcheck,
    }
