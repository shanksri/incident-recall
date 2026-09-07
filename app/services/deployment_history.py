"""Deployment-history lookup — "what deployed near this failure window".

# Why this exists

A root cause proposed for an incident is far more credible when it can point
at a concrete change that landed shortly before the symptom appeared. This
answers exactly that question against GitHub's commit history: given a
repository and a failure timestamp, return the commits that landed in a
window around it.

# What this deliberately is, and is not

This is a lookup, not a diagnosis. It does not decide that a given commit
CAUSED the incident — it returns candidates a human or a downstream
hypothesis can weigh, in the same spirit as ``HypothesisEvaluator`` returning
supporting evidence rather than a verdict. Correlation in time is not
causation, and the caller (a human reviewing an RCA draft, or a future
hypothesis-scoring step) is responsible for that judgment, not this module.

# Window shape

``lookback`` (default 24h) and ``lookahead`` (default 15m) are asymmetric on
purpose: a deployment that causes a failure almost always PRECEDES it by
minutes to hours, so the lookback window is wide. A short lookahead exists
only to catch the case where the failure timestamp itself is slightly behind
the deploy that caused it (clock skew between systems, or a rollout that
takes a few minutes to reach the instance that alerted) — it is not meant to
catch deployments that happened well after the failure, which cannot have
caused it.

# Read-only, and fails open like other diagnostic reads

This is read-only diagnostics (see ``app.services.action_policy`` — it is
catalogued as READ_ONLY), so an API failure here should degrade the
*context* available to an investigation, not fail the investigation itself.
A timeout or transient error returns an empty list with a warning logged,
mirroring ``GitHubCollector``'s "preserve partial results, don't raise"
posture for the same failure. A caller that needs to distinguish "no
deployments happened" from "the lookup failed" should check the logs, not
branch on an empty list meaning something it doesn't.

Reuses the same authenticated ``httpx.Client`` construction as
``GitHubCollector`` rather than introducing a second way to talk to GitHub.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_LOOKBACK = timedelta(hours=24)
_DEFAULT_LOOKAHEAD = timedelta(minutes=15)
_DEFAULT_LIMIT = 10


class DeploymentHistoryError(RuntimeError):
    """Raised only for caller-fixable configuration errors (unknown
    repository, bad credentials) — never for a transient/network failure,
    which degrades to an empty result instead. See module docstring.
    """


@dataclass(frozen=True)
class DeploymentEvent:
    sha: str
    short_sha: str
    author: str | None
    message: str
    committed_at: datetime
    url: str


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_commit(item: dict) -> DeploymentEvent | None:
    commit = item.get("commit") or {}
    # GitHub's commit payload carries two distinct "author" concepts: the raw
    # git trailer info under commit.author/commit.committer (name, email,
    # date -- no GitHub identity), and the resolved GitHub user under the
    # top-level item.author/item.committer (has "login", may be null if the
    # commit's email doesn't match a GitHub account). Prefer the resolved
    # login; fall back to whichever raw trailer has a name.
    commit_author = commit.get("author") or {}
    commit_committer = commit.get("committer") or {}
    raw_date = commit_committer.get("date") or commit_author.get("date")
    sha = item.get("sha")
    if not sha or not raw_date:
        return None
    try:
        committed_at = datetime.strptime(raw_date, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None
    author = (
        (item.get("author") or {}).get("login")
        or commit_committer.get("name")
        or commit_author.get("name")
    )
    message = (commit.get("message") or "").splitlines()[0] if commit.get("message") else ""
    return DeploymentEvent(
        sha=sha, short_sha=sha[:7], author=author, message=message,
        committed_at=committed_at, url=item.get("html_url", ""),
    )


class DeploymentHistoryClient:
    """Looks up recent commits on a GitHub repository. Mirrors
    ``GitHubCollector``'s construction: pass an authenticated token, or
    inject a fake ``httpx.Client`` for tests.
    """

    def __init__(
        self,
        token: str | None = None,
        timeout_seconds: float = 10.0,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        if client is not None:
            self._client = client  # injected (tests)
        else:
            headers = {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "enterprise-incident-intelligence-platform",
            }
            if token:
                headers["Authorization"] = f"Bearer {token}"
            self._client = httpx.Client(
                base_url="https://api.github.com",
                headers=headers,
                timeout=timeout_seconds,
                follow_redirects=True,
            )

    def find_near(
        self,
        owner: str,
        repo: str,
        around: datetime,
        *,
        lookback: timedelta = _DEFAULT_LOOKBACK,
        lookahead: timedelta = _DEFAULT_LOOKAHEAD,
        limit: int = _DEFAULT_LIMIT,
    ) -> list[DeploymentEvent]:
        """Return commits to ``owner/repo`` in ``[around - lookback, around +
        lookahead]``, most recent first. Empty on any transient failure —
        see module docstring's "fails open" section. Raises
        ``DeploymentHistoryError`` only for a 404 (repository does not
        exist / token cannot see it), since that is a caller mistake worth
        surfacing rather than silently returning no context.
        """
        params = {
            "since": _iso(around - lookback),
            "until": _iso(around + lookahead),
            "per_page": min(limit, 100),
        }
        try:
            response = self._client.get(f"/repos/{owner}/{repo}/commits", params=params)
        except (httpx.ReadTimeout, httpx.TimeoutException):
            logger.warning(
                "deployment_history.timeout",
                extra={"owner": owner, "repo": repo},
            )
            return []
        except httpx.HTTPError:
            logger.warning(
                "deployment_history.request_failed",
                exc_info=True,
                extra={"owner": owner, "repo": repo},
            )
            return []

        if response.status_code == 404:
            raise DeploymentHistoryError(
                f"repository {owner}/{repo} was not found, or the configured token "
                "cannot see it"
            )
        if response.status_code != 200:
            logger.warning(
                "deployment_history.non_200",
                extra={"owner": owner, "repo": repo, "status": response.status_code},
            )
            return []

        events = [
            event for item in response.json()[:limit]
            if (event := _parse_commit(item)) is not None
        ]
        logger.info(
            "deployment_history.lookup_complete",
            extra={"owner": owner, "repo": repo, "found": len(events)},
        )
        return events

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> DeploymentHistoryClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
