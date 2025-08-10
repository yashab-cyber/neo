from neo import create_app
from fastapi.testclient import TestClient


def token(client):
    r = client.post("/api/v1/auth/token", json={"api_key": "devkey", "role": "admin"})
    assert r.status_code == 200
    return r.json()["access_token"]


def test_neural_overview_completion():
    app = create_app()
    with TestClient(app) as client:
        t = token(client)
        r = client.get("/api/v1/cognitive/neural/overview", headers={"Authorization": f"Bearer {t}"})
        assert r.status_code == 200
        data = r.json()
        assert data["total"] > 0
        assert data["implemented"] == data["total"]
        assert abs(data["completion"] - 1.0) < 1e-9
        names = {c["component"] for c in data["components"]}
        required = {
            "input_processing","transformer","convolutional","spiking","recursive","memory_augmented_ntm","memory_augmented_dnc","attention_multi_head","attention_cross_modal","generative_vae","generative_gan","gnn","capsule","training_progressive_growing","training_transfer_learning"
        }
        assert required.issubset(names)
