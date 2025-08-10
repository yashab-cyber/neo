from __future__ import annotations
from datetime import datetime, timedelta, UTC
from typing import Dict, Any, List, Optional
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from neo.db.models import MemoryItemModel

# Simple retention defaults
SHORT_TERM_RETENTION = timedelta(seconds=30)
WORKING_CAPACITY = 8  # items
CONSOLIDATION_BATCH = 5  # how many short_term to move each consolidation run
FORGETTING_DECAY_SECONDS = 300  # items older than this with low importance may be removed


def _now():
    return datetime.now(UTC)


class MemoryService:
    """Memory system with stages: short_term -> working -> long_term
    Consolidation moves items from short_term to long_term, and forgetting prunes low value items.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def store(self, kind: str, content: Dict[str, Any], stage: str = "short_term", context: Optional[Dict[str, Any]] = None, importance: float = 0.5) -> MemoryItemModel:
        item_id = uuid.uuid4().hex[:32]
        item = MemoryItemModel(
            id=item_id,
            kind=kind,
            stage=stage,
            content=content,
            context=context or {},
            encoding={"strength": importance, "timestamp": _now().isoformat()},
            importance={"relevance_score": importance},
        )
        self.session.add(item)
        await self.session.commit()
        return item

    async def retrieve(self, kind: Optional[str] = None, stage: Optional[str] = None, limit: int = 20) -> List[MemoryItemModel]:
        stmt = select(MemoryItemModel)
        if kind:
            stmt = stmt.where(MemoryItemModel.kind == kind)
        if stage:
            stmt = stmt.where(MemoryItemModel.stage == stage)
        stmt = stmt.order_by(MemoryItemModel.updated_at.desc()).limit(limit)
        res = await self.session.execute(stmt)
        return list(res.scalars())

    async def access(self, item_id: str) -> Optional[MemoryItemModel]:
        item = await self.session.get(MemoryItemModel, item_id)
        if not item:
            return None
        item.access_count += 1
        await self.session.commit()
        return item

    async def consolidate(self) -> int:
        # move some short_term items to long_term
        stmt = select(MemoryItemModel).where(MemoryItemModel.stage == "short_term").order_by(MemoryItemModel.created_at.asc()).limit(CONSOLIDATION_BATCH)
        res = await self.session.execute(stmt)
        items = list(res.scalars())
        moved = 0
        for it in items:
            it.stage = "long_term"
            moved += 1
        if moved:
            await self.session.commit()
        return moved

    async def prune_short_term(self) -> int:
        # forget short_term items older than retention
        cutoff = _now() - SHORT_TERM_RETENTION
        stmt = select(MemoryItemModel).where(MemoryItemModel.stage == "short_term", MemoryItemModel.created_at < cutoff)
        res = await self.session.execute(stmt)
        ids = [i.id for i in res.scalars()]
        if ids:
            await self.session.execute(delete(MemoryItemModel).where(MemoryItemModel.id.in_(ids)))
            await self.session.commit()
        return len(ids)

    async def forgetting_curve_prune(self) -> int:
        # remove long_term with low importance and old
        cutoff = _now() - timedelta(seconds=FORGETTING_DECAY_SECONDS)
        stmt = select(MemoryItemModel).where(MemoryItemModel.stage == "long_term", MemoryItemModel.updated_at < cutoff)
        res = await self.session.execute(stmt)
        candidates = list(res.scalars())
        to_delete = [c.id for c in candidates if (c.importance.get("relevance_score", 0) < 0.3 and c.access_count < 1)]
        if to_delete:
            await self.session.execute(delete(MemoryItemModel).where(MemoryItemModel.id.in_(to_delete)))
            await self.session.commit()
        return len(to_delete)

    async def working_set(self) -> List[MemoryItemModel]:
        stmt = select(MemoryItemModel).where(MemoryItemModel.stage == "working").order_by(MemoryItemModel.updated_at.desc()).limit(WORKING_CAPACITY)
        res = await self.session.execute(stmt)
        return list(res.scalars())

    async def promote_to_working(self, item_id: str) -> bool:
        item = await self.session.get(MemoryItemModel, item_id)
        if not item:
            return False
        item.stage = "working"
        await self.session.commit()
        # enforce capacity
        working = await self.working_set()
        if len(working) > WORKING_CAPACITY:
            # drop least recently updated beyond capacity back to short_term
            overflow = working[WORKING_CAPACITY:]
            for it in overflow:
                it.stage = "short_term"
            await self.session.commit()
        return True

    async def delete_item(self, item_id: str) -> bool:
        res = await self.session.execute(delete(MemoryItemModel).where(MemoryItemModel.id == item_id))
        await self.session.commit()
        return res.rowcount > 0

    async def stats(self) -> dict[str, int]:
        """Return simple counts of items per stage."""
        from sqlalchemy import func
        stmt = select(MemoryItemModel.stage, func.count()).group_by(MemoryItemModel.stage)
        res = await self.session.execute(stmt)
        counts = {row[0]: row[1] for row in res}
        for stage in ["short_term", "working", "long_term"]:
            counts.setdefault(stage, 0)
        return counts

__all__ = ["MemoryService"]
