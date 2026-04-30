"""Classify persisted error strings for resume vs fair-metrics exclusion.

* :mod:`swerouter.harness.run_eval` — **resume** only uses
  :func:`is_transport_or_infra_failure` (HTTP/TLS/quota / transient transport).
  Broad harness/provider glitches must **not** force infinite re-queue (e.g.
  ``argument list too long``).
* :mod:`swerouter.leaderboard.score` — ``--exclude-infra-failures`` uses
  :func:`is_excluded_from_fair_metrics`, which adds environment / harness /
  upstream-provider failures that are not the router's task competence.
"""

from __future__ import annotations

# Substring probes (not instance-id hacks). Anything that does not match is
# treated as a normal completed outcome for resume, and as a countable outcome
# for scoring unless ``exclude_infra_failures`` is enabled.
TRANSPORT_INFRA_ERROR_SUBSTRINGS: tuple[str, ...] = (
    "ProviderMissingUsageError",
    "MalformedResponseError",
    "TransientResponseError",
    "ConnectionError",
    "Timeout",
    "HTTPError",
    "ReadError",
    "ChunkedEncodingError",
    "HTTP 5",  # HTTP 500/502/503/504 after retries exhausted
    "HTTP 408",
    "HTTP 409",
    "HTTP 425",
    "HTTP 429",
    "missing usage object",
    "response is not JSON",
    "unexpected response shape",
    # OpenRouter / provider account limits (not model routing capability).
    "Key limit exceeded",
    "insufficient_quota",
    # TLS / network between harness and API.
    "SSLError",
    "SSLCertVerificationError",
    "CertificateError",
    "ConnectTimeout",
    "NameResolutionError",
    "Temporary failure in name resolution",
)


def is_transport_or_infra_failure(message: str | None) -> bool:
    """True when ``message`` looks like HTTP/TLS/quota infra, not agent logic."""

    if not message:
        return False
    return any(pat in message for pat in TRANSPORT_INFRA_ERROR_SUBSTRINGS)


# Excluded from fair leaderboard totals when ``--exclude-infra-failures`` is
# on, but **not** used for resume re-queue (see module docstring).
_SCORING_ONLY_INFRA_SUBSTRINGS: tuple[str, ...] = (
    # Linux / docker exec ARG_MAX style failures when applying large patches.
    "argument list too long",
    # SWE-bench harness did not write report.json (disk, timeout, etc.).
    "eval report not produced",
    # OpenRouter wraps upstream 400 (e.g. Gemini "Corrupted thought signature").
    "Provider returned error",
)


def is_excluded_from_fair_metrics(
    agent_error: str | None, eval_error: str | None
) -> bool:
    """True if either error string should drop the instance from fair metrics."""

    for msg in (agent_error, eval_error):
        if not msg:
            continue
        if is_transport_or_infra_failure(msg):
            return True
        if any(pat in msg for pat in _SCORING_ONLY_INFRA_SUBSTRINGS):
            return True
    return False


__all__ = [
    "TRANSPORT_INFRA_ERROR_SUBSTRINGS",
    "is_transport_or_infra_failure",
    "is_excluded_from_fair_metrics",
]
