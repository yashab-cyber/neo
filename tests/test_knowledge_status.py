from neo import create_app
from fastapi.testclient import TestClient

def token(client):
    r = client.post("/api/v1/auth/token", json={"api_key": "devkey", "role": "admin"})
    return r.json()["access_token"]


def test_knowledge_status_layers():
    app = create_app()
    with TestClient(app) as client:
        t = token(client)
        # seed required entity types into memory backend
        for typ in ["Person","Organization","Concept","Event","Object","Location","Technology","Science","Business"]:
            r = client.post("/api/v1/knowledge/nodes", json={"type": typ, "properties": {"seed": True}}, headers={"Authorization": f"Bearer {t}"})
            assert r.status_code == 200
        # add one edge to satisfy relationship layer
        nodes = client.get("/api/v1/knowledge/query?type=Person", headers={"Authorization": f"Bearer {t}"}).json()
        n1 = nodes[0]["id"]
        nodes2 = client.get("/api/v1/knowledge/query?type=Concept", headers={"Authorization": f"Bearer {t}"}).json()
        n2 = nodes2[0]["id"]
        e = client.post("/api/v1/knowledge/edges", json={"source_id": n1, "target_id": n2, "relationship_type": "relatedTo"}, headers={"Authorization": f"Bearer {t}"})
        assert e.status_code == 200
        s = client.get("/api/v1/knowledge/status", headers={"Authorization": f"Bearer {t}"})
        assert s.status_code == 200
        data = s.json()
        assert data["total"] > 0
        # ensure at least 4 layers implemented (entity, relationship, attribute, domain)
        assert data["implemented"] >= 4
