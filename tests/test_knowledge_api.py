from neo import create_app
from fastapi.testclient import TestClient


def token(client):
    r = client.post("/api/v1/auth/token", json={"api_key": "devkey", "role": "admin"})
    assert r.status_code == 200
    return r.json()["access_token"]


def test_knowledge_api_flow():
    app = create_app()
    with TestClient(app) as client:
        t = token(client)
        # create nodes
        r1 = client.post("/api/v1/knowledge/nodes", json={"type": "User", "properties": {"name": "alice"}}, headers={"Authorization": f"Bearer {t}"})
        assert r1.status_code == 200
        alice_id = r1.json()["id"]
        r2 = client.post("/api/v1/knowledge/nodes", json={"type": "Action", "properties": {"verb": "open"}}, headers={"Authorization": f"Bearer {t}"})
        action_id = r2.json()["id"]
        # edge
        r3 = client.post("/api/v1/knowledge/edges", json={"source_id": alice_id, "target_id": action_id, "relationship_type": "performsAction"}, headers={"Authorization": f"Bearer {t}"})
        assert r3.status_code == 200
        # query
        q = client.get("/api/v1/knowledge/query?type=User", headers={"Authorization": f"Bearer {t}"})
        assert q.status_code == 200 and len(q.json()) == 1
        # neighbors
        nb = client.get(f"/api/v1/knowledge/nodes/{alice_id}/neighbors", headers={"Authorization": f"Bearer {t}"})
        assert nb.status_code == 200 and len(nb.json()) == 1
        # traverse
        tr = client.get(f"/api/v1/knowledge/traverse/{alice_id}?max_depth=2", headers={"Authorization": f"Bearer {t}"})
        assert tr.status_code == 200 and len(tr.json()) >= 1
        # delete
        dl = client.delete(f"/api/v1/knowledge/nodes/{alice_id}", headers={"Authorization": f"Bearer {t}"})
        assert dl.status_code == 200
        get_missing = client.get(f"/api/v1/knowledge/nodes/{alice_id}", headers={"Authorization": f"Bearer {t}"})
        assert get_missing.status_code == 404
