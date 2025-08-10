from neo import create_app
from fastapi.testclient import TestClient


def token(client):
    r = client.post("/api/v1/auth/token", json={"api_key": "devkey", "role": "admin"})
    assert r.status_code == 200
    return r.json()["access_token"]


def test_learning_status_endpoint():
    app = create_app()
    with TestClient(app) as client:
        t = token(client)
        r = client.get("/api/v1/knowledge/learning/status", headers={"Authorization": f"Bearer {t}"})
        assert r.status_code == 200
        data = r.json()
        assert "layers" in data and isinstance(data["layers"], list)
        assert len(data["layers"]) >= 5
        for layer in data["layers"]:
            assert set(["name","implemented","description"]).issubset(layer.keys())
        assert 0.0 <= data["completion"] <= 1.0
        assert data["metrics"]["layers_total"] == len(data["layers"])
        assert data["metrics"]["layers_implemented"] <= data["metrics"]["layers_total"]
