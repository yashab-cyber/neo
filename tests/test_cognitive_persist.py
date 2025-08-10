from neo import create_app
from fastapi.testclient import TestClient


def token(client):
    r = client.post("/api/v1/auth/token", json={"api_key": "devkey", "role": "admin"})
    assert r.status_code == 200
    return r.json()["access_token"]


def test_cognitive_roadmap_patch():
    app = create_app()
    with TestClient(app) as client:
        t = token(client)
        item = "memory_systems.working_memory"
        patch = client.patch(
            f"/api/v1/cognitive/roadmap/1/{item}",
            json={"implemented": True, "description": "Implemented core WM"},
            headers={"Authorization": f"Bearer {t}"},
        )
        assert patch.status_code == 200
        rm = client.get("/api/v1/cognitive/roadmap", headers={"Authorization": f"Bearer {t}"})
        assert rm.status_code == 200
        phases = rm.json()["phases"]
        phase1 = phases.get("1") or phases.get(1)
        found = [x for x in phase1["items"] if x["item"] == item]
        assert found and found[0]["implemented"] is True
