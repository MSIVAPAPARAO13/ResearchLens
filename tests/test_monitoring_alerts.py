"""Unit tests for ResearchAlert models and notification provider interfaces."""

from researchlens.analysis.monitoring_service import (
    ConsoleNotificationProvider,
    NotificationProvider,
    ResearchAlert,
)


def test_research_alert_serialization():
    alert = ResearchAlert(
        alert_id="alt_001",
        topic_id="top_001",
        type="NEW_METHOD",
        severity="IMPORTANT",
        title="New Architecture Discovered",
        message="Mamba state space model observed in recent preprint.",
        evidence="Observed in 'Mamba: Linear-Time Sequence Modeling'.",
    )

    d = alert.to_dict()
    assert d["alert_id"] == "alt_001"
    assert d["severity"] == "IMPORTANT"
    assert d["read"] is False

    restored = ResearchAlert.from_dict(d)
    assert restored.title == alert.title
    assert restored.evidence == alert.evidence


def test_console_notification_provider():
    provider = ConsoleNotificationProvider()
    assert isinstance(provider, NotificationProvider)

    alert = ResearchAlert(
        alert_id="alt_002",
        topic_id="top_001",
        type="NEW_PAPER",
        severity="INFO",
        title="New Paper Added",
        message="A paper was indexed.",
        evidence="Paper title.",
    )

    success = provider.send_alert(alert)
    assert success is True
