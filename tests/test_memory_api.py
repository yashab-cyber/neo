from neo import create_app
from fastapi.testclient import TestClient

def token(client):
    r = client.post("/api/v1/auth/token", json={"api_key": "devkey", "role": "admin"})
    assert r.status_code == 200
    return r.json()["access_token"]


def test_memory_api_flow():
    app = create_app()
    with TestClient(app) as client:
        t = token(client)
        headers = {"Authorization": f"Bearer {t}"}
        # create memory
    r = client.post("/api/v1/memory/items", json={"kind": "episodic", "content": {"a":1}}, headers=headers)
    assert r.status_code == 200
    mem_id = r.json()["id"]
    # list
    lst = client.get("/api/v1/memory/items?stage=short_term", headers=headers)
    assert lst.status_code == 200 and any(i["id"] == mem_id for i in lst.json())
    # access
    acc = client.post(f"/api/v1/memory/items/{mem_id}/access", headers=headers)
    assert acc.status_code == 200 and acc.json()["access_count"] >= 1
    # promote
    prom = client.post(f"/api/v1/memory/items/{mem_id}/promote", headers=headers)
    assert prom.status_code == 200
    # working set
    wk = client.get("/api/v1/memory/working", headers=headers)
    assert wk.status_code == 200
    # consolidate
    cons = client.post("/api/v1/memory/consolidate", headers=headers)
    assert cons.status_code == 200
    # prune short term
    prune = client.post("/api/v1/memory/prune/short_term", headers=headers)
    assert prune.status_code == 200
    # forgetting curve
    fc = client.post("/api/v1/memory/prune/forgetting_curve", headers=headers)
    assert fc.status_code == 200
    # status
    status_resp = client.get("/api/v1/memory/status", headers=headers)
    assert status_resp.status_code == 200 and "counts" in status_resp.json()
    # delete
    dele = client.delete(f"/api/v1/memory/items/{mem_id}", headers=headers)
    assert dele.status_code == 200
