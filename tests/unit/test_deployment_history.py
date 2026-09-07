"""Tests for the deployment-history lookup ("what deployed near this failure
window"). Uses ``httpx.MockTransport`` exactly like ``GitHubCollector``'s
tests — no real network, no token.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest

from app.services.deployment_history import (
    DeploymentHistoryClient,
    DeploymentHistoryError,
)

AROUND = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def _client(handler) -> DeploymentHistoryClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport, base_url="https://api.github.com")
    return DeploymentHistoryClient(client=http_client)


def _commit(sha: str, message: str, when: datetime, *, author: str = "octocat") -> dict:
    return {
        "sha": sha,
        "html_url": f"https://github.com/o/r/commit/{sha}",
        "author": {"login": author},
        "commit": {
            "message": message,
            "committer": {"date": when.strftime("%Y-%m-%dT%H:%M:%SZ")},
        },
    }


# ── Happy path ─────────────────────────────────────────────────────────────


def test_returns_parsed_commits_in_the_window() -> None:
    commits = [
        _commit("a" * 40, "Deploy: bump config", AROUND - timedelta(hours=2)),
        _commit("b" * 40, "Fix flaky test", AROUND - timedelta(hours=20)),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=commits)

    events = _client(handler).find_near("o", "r", AROUND)
    assert len(events) == 2
    assert events[0].sha == "a" * 40
    assert events[0].short_sha == "a" * 7
    assert events[0].message == "Deploy: bump config"
    assert events[0].author == "octocat"


def test_multiline_commit_message_keeps_only_the_first_line() -> None:
    commits = [_commit("c" * 40, "Rollback bad deploy\n\nDetails here", AROUND)]
    events = _client(lambda r: httpx.Response(200, json=commits)).find_near("o", "r", AROUND)
    assert events[0].message == "Rollback bad deploy"


def test_falls_back_to_committer_name_when_no_github_login() -> None:
    # Realistic shape: the raw git committer name lives under commit.committer,
    # not commit.author -- item.author is null when the commit email doesn't
    # match a GitHub account.
    commit = _commit("d" * 40, "msg", AROUND)
    commit["author"] = None
    commit["commit"]["committer"]["name"] = "Jane Doe"
    events = _client(lambda r: httpx.Response(200, json=[commit])).find_near("o", "r", AROUND)
    assert events[0].author == "Jane Doe"


def test_result_limit_is_applied() -> None:
    commits = [_commit(f"{i:040d}", f"c{i}", AROUND) for i in range(5)]
    events = _client(lambda r: httpx.Response(200, json=commits)).find_near(
        "o", "r", AROUND, limit=2
    )
    assert len(events) == 2


def test_malformed_commit_entries_are_skipped_not_fatal() -> None:
    good = _commit("e" * 40, "good", AROUND)
    bad_no_sha = {"commit": {"message": "x", "committer": {"date": "2026-01-01T00:00:00Z"}}}
    bad_no_date = {"sha": "f" * 40, "commit": {"message": "x", "committer": {}}}
    events = _client(
        lambda r: httpx.Response(200, json=[good, bad_no_sha, bad_no_date])
    ).find_near("o", "r", AROUND)
    assert len(events) == 1
    assert events[0].sha == "e" * 40


# ── Window parameters reach the API correctly ───────────────────────────────


def test_since_and_until_reflect_lookback_and_lookahead() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["since"] = request.url.params.get("since")
        captured["until"] = request.url.params.get("until")
        return httpx.Response(200, json=[])

    _client(handler).find_near(
        "o", "r", AROUND, lookback=timedelta(hours=1), lookahead=timedelta(minutes=10)
    )
    assert captured["since"] == "2026-06-15T11:00:00Z"
    assert captured["until"] == "2026-06-15T12:10:00Z"


def test_default_lookback_is_wider_than_lookahead() -> None:
    """A deploy that causes a failure precedes it -- the window must be
    asymmetric, weighted toward the past."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["since"] = request.url.params.get("since")
        captured["until"] = request.url.params.get("until")
        return httpx.Response(200, json=[])

    _client(handler).find_near("o", "r", AROUND)
    since = datetime.strptime(captured["since"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    until = datetime.strptime(captured["until"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    assert (AROUND - since) > (until - AROUND)


# ── Fails open on transient failure, raises only on 404 ─────────────────────


def test_timeout_returns_empty_list_not_an_exception() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    assert _client(handler).find_near("o", "r", AROUND) == []


def test_network_error_returns_empty_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    assert _client(handler).find_near("o", "r", AROUND) == []


def test_non_200_non_404_status_returns_empty_list() -> None:
    events = _client(lambda r: httpx.Response(503)).find_near("o", "r", AROUND)
    assert events == []


def test_404_raises_the_typed_error() -> None:
    with pytest.raises(DeploymentHistoryError, match="o/r"):
        _client(lambda r: httpx.Response(404)).find_near("o", "r", AROUND)


def test_empty_result_set_is_not_an_error() -> None:
    assert _client(lambda r: httpx.Response(200, json=[])).find_near("o", "r", AROUND) == []


# ── Construction / context manager ──────────────────────────────────────────


def test_context_manager_closes_the_client() -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=[]))
    http_client = httpx.Client(transport=transport, base_url="https://api.github.com")
    with DeploymentHistoryClient(client=http_client) as dh:
        dh.find_near("o", "r", AROUND)
    assert http_client.is_closed


def test_token_is_sent_as_a_bearer_header() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=[])

    # Construct without an injected client so the real header-building path runs,
    # but swap in a mock transport afterward so no network call is made.
    client = DeploymentHistoryClient(token="secret-token")
    client._client._transport = httpx.MockTransport(handler)
    client.find_near("o", "r", AROUND)
    assert captured["auth"] == "Bearer secret-token"
