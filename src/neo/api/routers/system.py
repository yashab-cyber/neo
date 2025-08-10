from fastapi import APIRouter
import platform, socket, psutil, time
from datetime import datetime

router = APIRouter()
start_time = time.time()


@router.get("/status")
async def status():
    uptime_seconds = int(time.time() - start_time)
    uptime = f"{uptime_seconds // 3600}h {(uptime_seconds % 3600)//60}m {uptime_seconds % 60}s"
    from neo.config import settings
    return {
        "status": "healthy",
        "version": settings.version,
        "uptime": uptime,
        "system": {
            "cpu_usage": psutil.cpu_percent(interval=0.1),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage("/").percent,
        },
        "services": {
            "ai_engine": "initializing",
            "database": "unknown",
            "scheduler": "initializing",
        },
    }


@router.get("/system/info")
async def system_info():
    return {
        "hostname": socket.gethostname(),
        "platform": platform.system().lower(),
        "architecture": platform.machine(),
        "cpu_cores": psutil.cpu_count(logical=True),
        "total_memory": f"{round(psutil.virtual_memory().total / (1024**3))}GB",
        "neo_version": "0.1.0",
        "python_version": platform.python_version(),
        "ai_models": [],
    }
