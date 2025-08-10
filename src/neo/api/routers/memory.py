from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from neo.services.memory_service import MemoryService
from neo.db import get_session
from sqlalchemy.ext.asyncio import AsyncSession
from .auth import require_scope

router = APIRouter()


class MemoryCreate(BaseModel):
    kind: str
    content: Dict[str, Any]
    stage: str = "short_term"
    context: Optional[Dict[str, Any]] = None
    importance: float = 0.5


class MemoryItemOut(BaseModel):
    id: str
    kind: str
    stage: str
    access_count: int
    importance: Dict[str, Any]


@router.post("/memory/items", dependencies=[Depends(require_scope("write:memory"))])
async def create_memory(body: MemoryCreate, session: AsyncSession = Depends(get_session)):
    svc = MemoryService(session)
    item = await svc.store(body.kind, body.content, stage=body.stage, context=body.context, importance=body.importance)
    return {"id": item.id, "kind": item.kind, "stage": item.stage, "importance": item.importance, "access_count": item.access_count}


@router.get("/memory/items", dependencies=[Depends(require_scope("read:memory"))])
async def list_memory(kind: Optional[str] = None, stage: Optional[str] = None, limit: int = 20, session: AsyncSession = Depends(get_session)):
    svc = MemoryService(session)
    items = await svc.retrieve(kind=kind, stage=stage, limit=limit)
    return [{"id": i.id, "kind": i.kind, "stage": i.stage, "importance": i.importance, "access_count": i.access_count} for i in items]


@router.post("/memory/items/{item_id}/access", dependencies=[Depends(require_scope("read:memory"))])
async def access_memory(item_id: str, session: AsyncSession = Depends(get_session)):
    svc = MemoryService(session)
    item = await svc.access(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Not found")
    return {"id": item.id, "kind": item.kind, "stage": item.stage, "access_count": item.access_count, "importance": item.importance}


@router.post("/memory/consolidate", dependencies=[Depends(require_scope("write:memory"))])
async def consolidate(session: AsyncSession = Depends(get_session)):
    svc = MemoryService(session)
    moved = await svc.consolidate()
    return {"moved": moved}


@router.post("/memory/prune/short_term", dependencies=[Depends(require_scope("write:memory"))])
async def prune_short_term(session: AsyncSession = Depends(get_session)):
    svc = MemoryService(session)
    deleted = await svc.prune_short_term()
    return {"deleted": deleted}


@router.post("/memory/prune/forgetting_curve", dependencies=[Depends(require_scope("write:memory"))])
async def forgetting_curve(session: AsyncSession = Depends(get_session)):
    svc = MemoryService(session)
    deleted = await svc.forgetting_curve_prune()
    return {"deleted": deleted}


@router.post("/memory/items/{item_id}/promote", dependencies=[Depends(require_scope("write:memory"))])
async def promote(item_id: str, session: AsyncSession = Depends(get_session)):
    svc = MemoryService(session)
    ok = await svc.promote_to_working(item_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Not found")
    return {"promoted": item_id}


@router.get("/memory/working", dependencies=[Depends(require_scope("read:memory"))])
async def working_set(session: AsyncSession = Depends(get_session)):
    svc = MemoryService(session)
    items = await svc.working_set()
    return [{"id": i.id, "kind": i.kind, "stage": i.stage, "importance": i.importance, "access_count": i.access_count} for i in items]


@router.delete("/memory/items/{item_id}", dependencies=[Depends(require_scope("write:memory"))])
async def delete_item(item_id: str, session: AsyncSession = Depends(get_session)):
    svc = MemoryService(session)
    ok = await svc.delete_item(item_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Not found")
    return {"deleted": item_id}


@router.get("/memory/status", dependencies=[Depends(require_scope("read:memory"))])
async def memory_status(session: AsyncSession = Depends(get_session)):
    svc = MemoryService(session)
    all_items = await svc.retrieve(limit=200)
    stages = {}
    for it in all_items:
        stages.setdefault(it.stage, 0)
        stages[it.stage] += 1
    return {
        "counts": stages,
        "total": len(all_items),
        "working_capacity": 8,
    }

__all__ = ["router"]
