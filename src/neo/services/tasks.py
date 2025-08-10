"""Simple in-memory task scheduler placeholder."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Any
import time


@dataclass
class Task:
    name: str
    command: str
    schedule: str | None = None
    last_run: float | None = None
    handler: Callable[[], Any] | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskScheduler:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}

    def add(self, task_id: str, task: Task):
        self.tasks[task_id] = task

    def run(self, task_id: str):
        t = self.tasks[task_id]
        t.last_run = time.time()
        if t.handler:
            return t.handler()
        return {"status": "ok"}


task_scheduler = TaskScheduler()
