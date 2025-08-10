from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json, asyncio, random, time

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            # Send dummy metrics update
            msg = {"event": "system.metrics", "cpu": random.uniform(10,70), "ts": time.time()}
            await ws.send_text(json.dumps(msg))
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        return
