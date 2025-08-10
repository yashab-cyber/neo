from neo import create_app
from fastapi.testclient import TestClient


def token(client):
    r = client.post("/api/v1/auth/token", json={"api_key": "devkey", "role": "admin"})
    assert r.status_code == 200
    return r.json()["access_token"]


def test_monitoring_status():
    app = create_app()
    with TestClient(app) as client:
        t = token(client)
        r = client.get("/api/v1/cognitive/monitoring/status", headers={"Authorization": f"Bearer {t}"})
        assert r.status_code == 200
        data = r.json()
        assert data["total"] > 0
        assert data["implemented"] == data["total"]
        assert abs(data["completion"] - 1.0) < 1e-9
        domains = {d["domain"] for d in data["domains"]}
        expected = {"data_collection_layer","alerting_rules","notification_channels","dashboards_core","storage_backends","performance_optimization","resource_optimization","compliance_security","reporting"}
        assert expected.issubset(domains)
