"""Alert-webhook normalization — turns a monitoring provider's payload shape
into one investigation-ready problem statement.

# Why this mirrors app.ingestion.adapters, not the other way around

``SourceAdapter`` (see ``app.ingestion.adapters.base``) already solved "many
providers, one canonical shape" for incident sources. This is the same
problem for a different input: a Prometheus Alertmanager webhook and a
PagerDuty event webhook have nothing in common syntactically, but both are
answering "something is on fire, here's what and where." ``AlertAdapter``
is deliberately the same ABC-plus-registry shape as ``SourceAdapter`` rather
than a bespoke design, because the problem is genuinely the same one.

The one structural difference: ingestion adapters are selected by an
explicit ``source_type`` the caller already knows (an ingestion run is
configured for "github" or "jira"). A webhook receiver does not get to ask
the sender which shape it used — Prometheus and PagerDuty both just POST
JSON — so adapters here are tried by ``matches()`` against the payload
itself, first match wins, exactly the "first matching rule wins" posture
``RuleBasedPlanner`` and ``DefaultRuleBasedRoutingPolicy`` already use
elsewhere in this codebase.

# Where this hands off

``build_problem_statement()`` is the seam: it renders a ``NormalizedAlert``
into the exact same kind of free-text string a human would type into
``POST /agent/investigate``. Nothing downstream of that call needs to know
an alert exists — the orchestrator, planner, generator, evaluator and critic
are completely unmodified. A webhook is just another way to produce the one
input the investigation pipeline has always accepted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


class UnrecognizedAlertPayloadError(ValueError):
    """Raised when no registered adapter's ``matches()`` accepts a payload.
    A 422 at the API boundary, not a 500 — an unrecognized shape is a caller
    error (wrong provider, malformed payload), not a server fault.
    """


@dataclass(frozen=True)
class NormalizedAlert:
    """The canonical, provider-agnostic shape every ``AlertAdapter`` must
    produce. Deliberately small — just enough to build a problem statement
    and to know which repository to check deployment history against.
    """

    source_type: str
    alert_name: str
    severity: str
    service: str | None
    summary: str
    description: str
    started_at: datetime | None
    raw_metadata: dict[str, Any] = field(default_factory=dict)


class AlertAdapter(ABC):
    source_type: str

    @abstractmethod
    def matches(self, payload: dict[str, Any]) -> bool:
        """Return True if this adapter recognizes ``payload``'s shape.
        Must not raise on a malformed or unrelated payload — return False.
        """

    @abstractmethod
    def normalize(self, payload: dict[str, Any]) -> NormalizedAlert:
        """Convert a payload this adapter has already accepted via
        ``matches()`` into a ``NormalizedAlert``. May assume the shape
        ``matches()`` checked for is present.
        """


def _parse_iso8601(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        # Alertmanager and PagerDuty both emit RFC3339 with a literal "Z".
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


class PrometheusAlertAdapter(AlertAdapter):
    """Alertmanager webhook payload. Accepts both the real envelope
    (``{"alerts": [...], ...}``, possibly batching several alerts under one
    webhook call) and a single bare alert (``{"labels": ..., "annotations":
    ...}``) for convenience when simulating payloads by hand.
    """

    source_type = "prometheus"

    def matches(self, payload: dict[str, Any]) -> bool:
        if isinstance(payload.get("alerts"), list) and payload["alerts"]:
            first = payload["alerts"][0]
            return isinstance(first, dict) and "labels" in first
        return "labels" in payload and "annotations" in payload

    def normalize(self, payload: dict[str, Any]) -> NormalizedAlert:
        alert = payload["alerts"][0] if isinstance(payload.get("alerts"), list) else payload
        labels = alert.get("labels") or {}
        annotations = alert.get("annotations") or {}
        return NormalizedAlert(
            source_type=self.source_type,
            alert_name=labels.get("alertname", "unknown_alert"),
            severity=labels.get("severity", "unknown"),
            service=labels.get("service") or labels.get("job"),
            summary=annotations.get("summary", labels.get("alertname", "")),
            description=annotations.get("description", ""),
            started_at=_parse_iso8601(alert.get("startsAt")),
            raw_metadata={"labels": labels, "annotations": annotations},
        )


class PagerDutyAlertAdapter(AlertAdapter):
    """PagerDuty webhook v3 event payload:
    ``{"event": {"event_type": ..., "data": {...}}}``.
    """

    source_type = "pagerduty"

    def matches(self, payload: dict[str, Any]) -> bool:
        event = payload.get("event")
        return isinstance(event, dict) and isinstance(event.get("data"), dict)

    def normalize(self, payload: dict[str, Any]) -> NormalizedAlert:
        data = payload["event"]["data"]
        service = data.get("service")
        service_name = service.get("summary") if isinstance(service, dict) else None
        priority = data.get("priority")
        severity = (
            priority.get("summary") if isinstance(priority, dict) else None
        ) or data.get("urgency", "unknown")
        return NormalizedAlert(
            source_type=self.source_type,
            alert_name=data.get("title", "unknown_alert"),
            severity=str(severity),
            service=service_name,
            summary=data.get("title", ""),
            description=data.get("description", data.get("title", "")),
            started_at=_parse_iso8601(data.get("created_at")),
            raw_metadata={"event_type": payload["event"].get("event_type")},
        )


# First match wins, same posture as RuleBasedPlanner's keyword priority order.
_ADAPTERS: tuple[AlertAdapter, ...] = (PrometheusAlertAdapter(), PagerDutyAlertAdapter())


def normalize_alert_payload(payload: dict[str, Any]) -> NormalizedAlert:
    for adapter in _ADAPTERS:
        if adapter.matches(payload):
            return adapter.normalize(payload)
    raise UnrecognizedAlertPayloadError(
        "payload did not match any registered alert adapter "
        f"({', '.join(a.source_type for a in _ADAPTERS)})"
    )


def build_problem_statement(alert: NormalizedAlert) -> str:
    """Render a NormalizedAlert as the free-text problem statement the
    investigation pipeline already accepts from a human. Deterministic
    string assembly, no LLM call -- the same "cheap and explainable first"
    posture as every other rendering function in this codebase (e.g.
    ``planner_agent._render_plan_context``).
    """
    parts = [f"{alert.alert_name} firing at {alert.severity} severity"]
    if alert.service:
        parts.append(f"on service {alert.service}")
    statement = " ".join(parts) + "."
    if alert.summary and alert.summary != alert.alert_name:
        statement += f" {alert.summary}."
    if alert.description and alert.description != alert.summary:
        statement += f" {alert.description}"
    return statement.strip()
