from neo import create_app
from fastapi.testclient import TestClient


def test_documentation_index():
    app = create_app()
    with TestClient(app) as client:
        r = client.post("/api/v1/auth/token", json={"api_key": "devkey", "role": "admin"})
        assert r.status_code == 200
        token = r.json()["access_token"]
        resp = client.get("/api/v1/documentation/index", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()
        for section in ["manual", "technical", "research", "user_guide"]:
            assert section in data["sections"]
            assert data["sections"][section]["pages"] >= 0
        assert data["totals"]["all_pages"] >= 0
        assert data["version"] == "1.0.0"
