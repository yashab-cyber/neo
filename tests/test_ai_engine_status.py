from neo import create_app
from fastapi.testclient import TestClient


def token(client):
    r = client.post("/api/v1/auth/token", json={"api_key": "devkey", "role": "admin"})
    assert r.status_code == 200
    return r.json()["access_token"]


def test_ai_engine_status_all_implemented():
    app = create_app()
    with TestClient(app) as client:
        t = token(client)
        r = client.get("/api/v1/cognitive/ai-engine/status", headers={"Authorization": f"Bearer {t}"})
        assert r.status_code == 200
        data = r.json()
        assert data["total"] > 0
        assert data["implemented"] == data["total"]
        assert abs(data["completion"] - 1.0) < 1e-9
        domains = {d["domain"] for d in data["domains"]}
        expected = {
            "input_layer","preprocessing_pipeline","deep_learning_paradigm","neuro_learning_paradigm","recursive_learning_paradigm","integration_layer","output_generation","feedback_learning","decision_fusion_details","performance_monitoring","optimization_engine","adaptation_triggers"
        }
        assert expected.issubset(domains)
