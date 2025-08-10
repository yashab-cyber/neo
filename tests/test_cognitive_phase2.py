from neo import create_app
from fastapi.testclient import TestClient


def token(client):
    r = client.post("/api/v1/auth/token", json={"api_key": "devkey", "role": "admin"})
    assert r.status_code == 200
    return r.json()["access_token"]


def test_phase2_roadmap_items_present_and_detected():
    app = create_app()
    with TestClient(app) as client:
        t = token(client)
        r = client.get("/api/v1/cognitive/roadmap", headers={"Authorization": f"Bearer {t}"})
        assert r.status_code == 200
        phases = r.json()["phases"]
        p2 = phases.get("2") or phases.get(2)
        assert p2 is not None
        items = {i["item"]: i for i in p2["items"]}
        expected = [
            "reasoning_engine.hybrid_integration.neuro_symbolic_bridges",
            "memory_systems.long_term_memory.episodic_memory.storage_format",
            "learning_systems.meta_learning.algorithms",
            "executive_control.goal_management.priority_scheduling",
            "perception_layer.attention_mechanisms.visual_attention.saliency_maps",
            "perception_layer.attention_mechanisms.linguistic_attention.self_attention",
        ]
        for e in expected:
            assert e in items
            assert items[e]["implemented"] is True  # all exist in config example


def test_integration_status_endpoint():
    app = create_app()
    with TestClient(app) as client:
        t = token(client)
        r = client.get("/api/v1/cognitive/integration/status", headers={"Authorization": f"Bearer {t}"})
        assert r.status_code == 200
        data = r.json()
        assert set(data["domains"]) == {
            "neuro_symbolic",
            "advanced_memory",
            "meta_learning",
            "executive_control",
            "attention_integration",
        }
        assert data["complete"] is True
