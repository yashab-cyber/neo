from fastapi.testclient import TestClient
from neo import create_app


def test_system_architecture_status():
    app = create_app()
    with TestClient(app) as client:
        token_resp = client.post("/api/v1/auth/token", json={"api_key": "devkey", "role": "admin"})
        assert token_resp.status_code == 200
        access = token_resp.json()["access_token"]
        resp = client.get("/api/v1/cognitive/system/status", headers={"Authorization": f"Bearer {access}"})
        assert resp.status_code == 200
        data = resp.json()
    assert "domains" in data
    assert data["total"] >= data["implemented"]
