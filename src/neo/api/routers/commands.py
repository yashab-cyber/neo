from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import uuid, time, asyncio

from .auth import require_scope
from neo.services.command_queue import queue

router = APIRouter()


class CommandRequest(BaseModel):
    command: str
    parameters: Dict[str, Any] | None = None
    async_: bool | None = False
    timeout: int | None = 30


class BatchCommandItem(BaseModel):
    id: str
    command: str
    parameters: Dict[str, Any] | None = None


class BatchCommandRequest(BaseModel):
    commands: List[BatchCommandItem]
    parallel: bool | None = True
    fail_fast: bool | None = False


@router.post("/commands/execute", dependencies=[Depends(require_scope("write:commands"))])
async def execute_command(req: CommandRequest):
    if req.async_:
        await queue.start()
        exec_id = await queue.enqueue(req.command, req.parameters)
        return {"execution_id": exec_id, "status": "queued", "timestamp": datetime_utc()}
    exec_id = f"exec_{uuid.uuid4().hex[:8]}"
    start = time.time()
    result = {"echo": req.command, "parameters": req.parameters or {}}
    return {"execution_id": exec_id, "status": "completed", "result": result, "execution_time": round(time.time() - start, 3), "timestamp": datetime_utc()}


@router.post("/commands/batch", dependencies=[Depends(require_scope("write:commands"))])
async def batch_commands(req: BatchCommandRequest):
    responses = []
    for item in req.commands:
        responses.append({
            "id": item.id,
            "status": "completed",
            "result": {"echo": item.command}
        })
    return {"results": responses}


@router.get("/commands/{execution_id}")
async def command_status(execution_id: str):
    res = queue.get(execution_id)
    if res:
        return res
    return {"execution_id": execution_id, "status": "unknown"}


@router.get("/commands/_debug/queue")  # simple optional debug helper
async def queue_debug():  # pragma: no cover - diagnostic
    size = 0
    try:
        if queue._queue is not None:  # type: ignore[attr-defined]
            size = queue._queue.qsize()  # type: ignore[attr-defined]
    except Exception:
        pass
    return {"queue_size": size}


def datetime_utc():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
