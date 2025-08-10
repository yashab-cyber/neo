from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid, time

from .auth import require_scope

router = APIRouter()

class TaskCreate(BaseModel):
    name: str
    command: str
    parameters: Dict[str, Any] | None = None
    schedule: str | None = None
    enabled: bool | None = True
    notifications: Dict[str, Any] | None = None


tasks_store: Dict[str, Dict[str, Any]] = {}


@router.get("/tasks")
async def list_tasks():
    tasks = []
    for tid, t in tasks_store.items():
        tasks.append({
            "id": tid,
            "name": t["name"],
            "status": t.get("status", "scheduled"),
            "next_run": t.get("next_run"),
            "last_run": t.get("last_run"),
            "success_rate": 100.0,
        })
    return {"tasks": tasks, "total": len(tasks), "running": 0, "scheduled": len(tasks)}


@router.post("/tasks", dependencies=[Depends(require_scope("manage:tasks"))])
async def create_task(req: TaskCreate):
    tid = f"task_{uuid.uuid4().hex[:6]}"
    tasks_store[tid] = req.model_dump()
    return {"id": tid, **req.model_dump()}


@router.post("/tasks/{task_id}/run", dependencies=[Depends(require_scope("manage:tasks"))])
async def run_task(task_id: str):
    task = tasks_store.get(task_id)
    if not task:
        return {"error": "NOT_FOUND"}
    task["last_run"] = datetime_utc()
    task["status"] = "completed"
    return {"id": task_id, "status": "completed", "finished_at": datetime_utc()}


def datetime_utc():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
