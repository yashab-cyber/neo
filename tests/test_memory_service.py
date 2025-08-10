import asyncio
from neo import create_app
from fastapi.testclient import TestClient
from neo.db import AsyncSessionLocal, engine, Base
from neo.services.memory_service import MemoryService, SHORT_TERM_RETENTION
from datetime import timedelta


def test_memory_store_consolidate_forget():
    app = create_app()
    # direct DB session usage
    async def run_flow():
        # ensure tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with AsyncSessionLocal() as session:
            svc = MemoryService(session)
            # store items
            items = []
            for i in range(6):
                it = await svc.store("episodic", {"seq": i}, stage="short_term", importance=0.6)
                items.append(it)
            moved = await svc.consolidate()
            assert moved > 0
            long_terms = await svc.retrieve(stage="long_term")
            # There may be pre-existing long_term items from previous tests sharing the DB; ensure at least moved were added
            assert len(long_terms) >= moved
            # simulate old items for forgetting
            for lt in long_terms:
                lt.updated_at = lt.updated_at - timedelta(seconds=400)
            await session.commit()
            pruned = await svc.forgetting_curve_prune()
            # importance 0.6 should keep them (low threshold <0.3)
            assert pruned == 0
            # lower importance simulate
            for lt in long_terms:
                lt.importance = {"relevance_score": 0.1}
                lt.updated_at = lt.updated_at - timedelta(seconds=400)
            await session.commit()
            pruned2 = await svc.forgetting_curve_prune()
            assert pruned2 >= 1
    asyncio.get_event_loop().run_until_complete(run_flow())
