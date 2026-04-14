from click.testing import CliRunner

from federated_inference.cli.coordinator_cli import _coordinator_health_url, main


def test_coordinator_health_url_defaults_to_localhost():
    assert _coordinator_health_url(None) == "http://127.0.0.1:8080/health"


def test_status_reports_empty_registry(monkeypatch):
    class DummyResponse:
        def json(self):
            return {"coordinator_state": "IDLE", "workers": []}

    def fake_get(url, timeout):
        assert url == "http://127.0.0.1:8080/health"
        assert timeout == 5
        return DummyResponse()

    monkeypatch.setattr("httpx.get", fake_get)

    result = CliRunner().invoke(main, ["status"])

    assert result.exit_code == 0
    assert "Coordinator state: IDLE" in result.output
    assert "No workers registered yet." in result.output
