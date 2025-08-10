from neo import create_app
from fastapi.testclient import TestClient
from neo.db import AsyncSessionLocal, engine, Base
from neo.services.memory_service import MemoryService
import asyncio


def token(client):
    r = client.post("/api/v1/auth/token", json={"api_key": "devkey", "role": "admin"})
    assert r.status_code == 200
    return r.json()["access_token"]


def test_knowledge_memory_crud():
    app = create_app()
    with TestClient(app) as client:
        t = token(client)
        headers = {"Authorization": f"Bearer {t}"}
        # memory persistence store & retrieve
        async def mem_flow():
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            async with AsyncSessionLocal() as session:
                svc = MemoryService(session)
                m = await svc.store("episodic", {"val": 1}, stage="short_term")
                got = await svc.retrieve(kind="episodic")
                assert any(x.id == m.id for x in got)
        asyncio.get_event_loop().run_until_complete(mem_flow())

        # knowledge CRUD (memory backend)
        n1 = client.post("/api/v1/knowledge/nodes", json={"type": "Entity", "properties": {"k": "v"}}, headers=headers)
        assert n1.status_code == 200
        nid = n1.json()["id"]
        get1 = client.get(f"/api/v1/knowledge/nodes/{nid}", headers=headers)
        assert get1.status_code == 200
        q = client.get("/api/v1/knowledge/query?type=Entity", headers=headers)
        assert q.status_code == 200 and len(q.json()) >= 1
        d = client.delete(f"/api/v1/knowledge/nodes/{nid}", headers=headers)
        assert d.status_code == 200
        g2 = client.get(f"/api/v1/knowledge/nodes/{nid}", headers=headers)
        assert g2.status_code == 404
