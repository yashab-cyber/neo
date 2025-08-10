from __future__ import annotations
import asyncio, uuid, time
from typing import Dict, Optional


class CommandQueue:
    def __init__(self):
        # Lazily instantiate asyncio.Queue inside running loop to avoid cross-loop issues in tests
        self._queue: Optional[asyncio.Queue[tuple[str, str, dict]]] = None
        self._results: Dict[str, dict] = {}
        self._worker_task: asyncio.Task | None = None

    async def start(self):
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker())

    async def stop(self):
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:  # pragma: no cover - expected on shutdown
                pass

    async def _worker(self):
        try:
            while True:
                assert self._queue is not None  # for type checkers
                exec_id, command, params = await self._queue.get()
                start = time.time()
                try:
                    await asyncio.sleep(0.05)
                    result = {"echo": command, "parameters": params}
                    status = "completed"
                except Exception as e:  # noqa: BLE001
                    result = {"error": str(e)}
                    status = "failed"
                self._results[exec_id] = {
                    "execution_id": exec_id,
                    "status": status,
                    "result": result,
                    "execution_time": round(time.time() - start, 3),
                }
                self._queue.task_done()
        except asyncio.CancelledError:  # pragma: no cover - normal shutdown path
            pass

    async def enqueue(self, command: str, params: dict | None) -> str:
        exec_id = f"exec_{uuid.uuid4().hex[:8]}"
        if self._worker_task is None or self._worker_task.done():  # ensure worker alive
            await self.start()
        if self._queue is None:
            # Should not happen because start() initializes, but guard for safety
            self._queue = asyncio.Queue()
        await self._queue.put((exec_id, command, params or {}))
        return exec_id

    def get(self, exec_id: str) -> dict | None:
        return self._results.get(exec_id)


queue = CommandQueue()
