from fastapi.testclient import TestClient
from neo import create_app


def token(client):
    r = client.post("/api/v1/auth/token", json={"api_key": "devkey", "role": "admin"})
    assert r.status_code == 200
    return r.json()["access_token"]


def test_system_architecture_status():
    app = create_app()
    with TestClient(app) as client:
        t = token(client)
        r = client.get('/api/v1/cognitive/system/status', headers={"Authorization": f"Bearer {t}"})
        assert r.status_code == 200
        data = r.json()
        assert data['total'] > 0
        assert data['implemented'] == data['total']
        assert abs(data['completion'] - 1.0) < 1e-9
        # spot-check a couple of expected domains
        names = {d['domain'] for d in data['domains']}
        assert {'user_interface_layer','deployment_architecture','quality_attributes'}.issubset(names)
