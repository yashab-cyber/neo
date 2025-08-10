from neo import create_app
from fastapi.testclient import TestClient


def token(client):
    r = client.post("/api/v1/auth/token", json={"api_key": "devkey", "role": "admin"})
    assert r.status_code == 200
    return r.json()["access_token"]


def test_phase3_and_phase4_items_present():
    app = create_app()
    with TestClient(app) as client:
        t = token(client)
        r = client.get("/api/v1/cognitive/roadmap", headers={"Authorization": f"Bearer {t}"})
        assert r.status_code == 200
        phases = r.json()["phases"]
        p3 = phases.get("3") or phases.get(3)
        p4 = phases.get("4") or phases.get(4)
        assert p3 and p4
        p3_items = {i["item"]: i for i in p3["items"]}
        p4_items = {i["item"]: i for i in p4["items"]}
        expected_p3 = [
            "distributed_processing.multi_agent.communication",
            "learning_systems.continual_learning.enabled",
            "reasoning_engine.hybrid_integration.symbolic_grounding",
            "adaptation.mechanisms.sophisticated",
            "performance_monitoring.metrics.enabled",
        ]
        expected_p4 = [
            "performance_optimization.memory_optimization.garbage_collection",
            "distributed_processing.multi_agent.agents",
            "tuning.parameters.auto_fine_tuning",
            "performance_monitoring.benchmarking.longitudinal",
            "deployment.readiness.status",
        ]
        for e in expected_p3:
            assert e in p3_items and p3_items[e]["implemented"] is True
        for e in expected_p4:
            assert e in p4_items and p4_items[e]["implemented"] is True


def test_advanced_status_endpoint():
    app = create_app()
    with TestClient(app) as client:
        t = token(client)
        r = client.get("/api/v1/cognitive/advanced/status", headers={"Authorization": f"Bearer {t}"})
        assert r.status_code == 200
        data = r.json()
        assert data["complete"] is True
        required = {
            "distributed_processing",
            "continual_learning",
            "advanced_reasoning",
            "adaptation_mechanisms",
            "performance_monitoring",
            "optimization_scaling",
            "production_readiness",
        }
        assert set(data["domains"]) == required
        assert all(data["domains"].values())
