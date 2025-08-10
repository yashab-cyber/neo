from neo import create_app
from fastapi.testclient import TestClient


def token(client):
    r = client.post("/api/v1/auth/token", json={"api_key": "devkey", "role": "admin"})
    assert r.status_code == 200
    return r.json()["access_token"]


def test_cognitive_endpoints():
    app = create_app()
    with TestClient(app) as client:
        t = token(client)
        cfg = client.get("/api/v1/cognitive/config", headers={"Authorization": f"Bearer {t}"})
        assert cfg.status_code == 200
        assert "cognitive_architecture" in cfg.json()
        roadmap = client.get("/api/v1/cognitive/roadmap", headers={"Authorization": f"Bearer {t}"})
        assert roadmap.status_code == 200
        phases = roadmap.json()["phases"]
        assert "1" in phases or 1 in phases
