from neo import create_app
from fastapi.testclient import TestClient


def token(client):
    r = client.post("/api/v1/auth/token", json={"api_key": "devkey", "role": "admin"})
    assert r.status_code == 200
    return r.json()["access_token"]


def test_security_framework_status():
    app = create_app()
    with TestClient(app) as client:
        t = token(client)
        r = client.get("/api/v1/cognitive/security/status", headers={"Authorization": f"Bearer {t}"})
        assert r.status_code == 200
        data = r.json()
        assert data["total"] > 0
        assert data["implemented"] == data["total"]
        assert abs(data["completion"] - 1.0) < 1e-9
        # Spot check a few domains exist
        domain_names = {d["domain"] for d in data["domains"]}
        for expect in [
            "perimeter_defense",
            "network_security",
            "application_security",
            "data_security",
            "ai_security",
            "cryptographic_architecture",
            "compliance_governance",
        ]:
            assert expect in domain_names
