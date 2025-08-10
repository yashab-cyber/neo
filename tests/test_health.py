from neo import create_app
from fastapi.testclient import TestClient


def test_root_and_health_endpoints():
    app = create_app()
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["app"].startswith("NEO")
        assert "version" in data
        assert "uptime_seconds" in data
        h = client.get("/healthz")
        assert h.status_code == 200 and h.json()["status"] == "ok"
        rd = client.get("/readyz")
        assert rd.status_code == 200 and rd.json()["ready"] is True
        fv = client.get("/favicon.ico")
        assert fv.status_code == 204
