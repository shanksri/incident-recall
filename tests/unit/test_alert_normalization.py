"""Tests for alert-webhook normalization: turning Prometheus/PagerDuty
payload shapes into one canonical NormalizedAlert, and rendering that into
the free-text problem statement the investigation pipeline already accepts.
"""
from __future__ import annotations

from datetime import timezone

import pytest

from app.services.alert_normalization import (
    NormalizedAlert,
    PagerDutyAlertAdapter,
    PrometheusAlertAdapter,
    UnrecognizedAlertPayloadError,
    build_problem_statement,
    normalize_alert_payload,
)

# ── Prometheus / Alertmanager ────────────────────────────────────────────────


def _alertmanager_envelope(**label_overrides) -> dict:
    labels = {"alertname": "HighErrorRate", "severity": "critical", "service": "checkout-api"}
    labels.update(label_overrides)
    return {
        "alerts": [
            {
                "labels": labels,
                "annotations": {
                    "summary": "Error rate above 5%",
                    "description": "checkout-api 5xx rate exceeded threshold for 5m",
                },
                "startsAt": "2026-06-15T12:00:00Z",
            }
        ],
    }


def test_prometheus_envelope_is_recognized_and_normalized() -> None:
    alert = normalize_alert_payload(_alertmanager_envelope())
    assert alert.source_type == "prometheus"
    assert alert.alert_name == "HighErrorRate"
    assert alert.severity == "critical"
    assert alert.service == "checkout-api"
    assert alert.summary == "Error rate above 5%"
    assert "5xx rate exceeded" in alert.description


def test_prometheus_bare_single_alert_shape_is_also_recognized() -> None:
    payload = {
        "labels": {"alertname": "DiskFull", "severity": "warning"},
        "annotations": {"summary": "Disk usage above 90%"},
    }
    alert = normalize_alert_payload(payload)
    assert alert.source_type == "prometheus"
    assert alert.alert_name == "DiskFull"


def test_prometheus_falls_back_to_job_label_when_no_service_label() -> None:
    payload = _alertmanager_envelope()
    del payload["alerts"][0]["labels"]["service"]
    payload["alerts"][0]["labels"]["job"] = "worker-pool"
    alert = normalize_alert_payload(payload)
    assert alert.service == "worker-pool"


def test_prometheus_parses_starts_at_as_utc() -> None:
    alert = normalize_alert_payload(_alertmanager_envelope())
    assert alert.started_at is not None
    assert alert.started_at.tzinfo == timezone.utc
    assert alert.started_at.hour == 12


def test_prometheus_missing_annotations_does_not_crash() -> None:
    payload = _alertmanager_envelope()
    del payload["alerts"][0]["annotations"]
    alert = normalize_alert_payload(payload)
    assert alert.summary == "HighErrorRate"  # falls back to alert name
    assert alert.description == ""


# ── PagerDuty ────────────────────────────────────────────────────────────────


def _pagerduty_envelope(**data_overrides) -> dict:
    data = {
        "title": "Payment service down",
        "description": "5xx spike on payment-service",
        "urgency": "high",
        "service": {"summary": "payment-service"},
        "priority": {"summary": "P1"},
        "created_at": "2026-06-15T11:55:00Z",
    }
    data.update(data_overrides)
    return {"event": {"event_type": "incident.triggered", "data": data}}


def test_pagerduty_envelope_is_recognized_and_normalized() -> None:
    alert = normalize_alert_payload(_pagerduty_envelope())
    assert alert.source_type == "pagerduty"
    assert alert.alert_name == "Payment service down"
    assert alert.severity == "P1"
    assert alert.service == "payment-service"
    assert alert.description == "5xx spike on payment-service"


def test_pagerduty_falls_back_to_urgency_when_no_priority() -> None:
    payload = _pagerduty_envelope()
    del payload["event"]["data"]["priority"]
    alert = normalize_alert_payload(payload)
    assert alert.severity == "high"


def test_pagerduty_falls_back_to_title_when_no_description() -> None:
    payload = _pagerduty_envelope()
    del payload["event"]["data"]["description"]
    alert = normalize_alert_payload(payload)
    assert alert.description == "Payment service down"


def test_pagerduty_parses_created_at() -> None:
    alert = normalize_alert_payload(_pagerduty_envelope())
    assert alert.started_at is not None
    assert alert.started_at.minute == 55


# ── Adapter isolation and dispatch ───────────────────────────────────────────


def test_prometheus_adapter_does_not_match_pagerduty_shape() -> None:
    assert PrometheusAlertAdapter().matches(_pagerduty_envelope()) is False


def test_pagerduty_adapter_does_not_match_prometheus_shape() -> None:
    assert PagerDutyAlertAdapter().matches(_alertmanager_envelope()) is False


def test_unrecognized_payload_raises_typed_error() -> None:
    with pytest.raises(UnrecognizedAlertPayloadError):
        normalize_alert_payload({"totally": "unrelated", "shape": True})


def test_empty_payload_raises_rather_than_crashing() -> None:
    with pytest.raises(UnrecognizedAlertPayloadError):
        normalize_alert_payload({})


def test_matches_never_raises_on_malformed_input() -> None:
    weird_inputs = [{"alerts": "not-a-list"}, {"alerts": []}, {"event": "not-a-dict"},
                    {"event": {"data": "not-a-dict"}}]
    for payload in weird_inputs:
        assert PrometheusAlertAdapter().matches(payload) is False
        assert PagerDutyAlertAdapter().matches(payload) is False


# ── build_problem_statement ──────────────────────────────────────────────────


def test_problem_statement_includes_name_severity_and_service() -> None:
    alert = NormalizedAlert(
        source_type="prometheus", alert_name="HighErrorRate", severity="critical",
        service="checkout-api", summary="Error rate above 5%",
        description="5xx rate exceeded for 5m", started_at=None,
    )
    statement = build_problem_statement(alert)
    assert "HighErrorRate" in statement
    assert "critical" in statement
    assert "checkout-api" in statement
    assert "Error rate above 5%" in statement
    assert "5xx rate exceeded" in statement


def test_problem_statement_omits_service_when_absent() -> None:
    alert = NormalizedAlert(
        source_type="prometheus", alert_name="X", severity="warning", service=None,
        summary="X", description="", started_at=None,
    )
    statement = build_problem_statement(alert)
    assert "on service" not in statement


def test_problem_statement_does_not_duplicate_identical_fields() -> None:
    """When summary/description equal the alert name, don't repeat it three
    times in a row."""
    alert = NormalizedAlert(
        source_type="prometheus", alert_name="X", severity="warning", service=None,
        summary="X", description="X", started_at=None,
    )
    statement = build_problem_statement(alert)
    assert statement.count("X") == 1


def test_problem_statement_is_nonempty_for_a_minimal_alert() -> None:
    alert = NormalizedAlert(
        source_type="pagerduty", alert_name="unknown_alert", severity="unknown",
        service=None, summary="", description="", started_at=None,
    )
    assert build_problem_statement(alert).strip()


def test_end_to_end_prometheus_to_problem_statement() -> None:
    alert = normalize_alert_payload(_alertmanager_envelope())
    statement = build_problem_statement(alert)
    assert "HighErrorRate" in statement and "checkout-api" in statement
