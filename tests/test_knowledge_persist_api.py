from neo import create_app
from fastapi.testclient import TestClient


def token(client):
    r = client.post("/api/v1/auth/token", json={"api_key": "devkey", "role": "admin"})
    assert r.status_code == 200
    return r.json()["access_token"]


def test_persistent_knowledge_api_flow():
    app = create_app()
    with TestClient(app) as client:
        t = token(client)
        headers = {"Authorization": f"Bearer {t}"}
        # create nodes persisted
        r1 = client.post("/api/v1/knowledge/nodes?backend=persist", json={"type": "User", "properties": {"name": "alice"}}, headers=headers)
        assert r1.status_code == 200
        alice_id = r1.json()["id"]
        r2 = client.post("/api/v1/knowledge/nodes?backend=persist", json={"type": "Action", "properties": {"verb": "open"}}, headers=headers)
        action_id = r2.json()["id"]
        # create edge
        e = client.post("/api/v1/knowledge/edges?backend=persist", json={"source_id": alice_id, "target_id": action_id, "relationship_type": "performsAction"}, headers=headers)
        assert e.status_code == 200
        # query
        q = client.get("/api/v1/knowledge/query?backend=persist&type=User", headers=headers)
        assert q.status_code == 200 and len(q.json()) == 1
        # neighbors
        nb = client.get(f"/api/v1/knowledge/nodes/{alice_id}/neighbors?backend=persist", headers=headers)
        assert nb.status_code == 200 and len(nb.json()) == 1
        # delete node
        dl = client.delete(f"/api/v1/knowledge/nodes/{alice_id}?backend=persist", headers=headers)
        assert dl.status_code == 200
        missing = client.get(f"/api/v1/knowledge/nodes/{alice_id}?backend=persist", headers=headers)
        assert missing.status_code == 404
