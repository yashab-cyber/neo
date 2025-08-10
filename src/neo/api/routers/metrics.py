from fastapi import APIRouter
from datetime import datetime, timezone, timedelta
import random

router = APIRouter()

@router.get("/metrics")
async def get_metrics(metric: str | None = None, start_time: str | None = None, end_time: str | None = None, granularity: str | None = None):
    now = datetime.now(timezone.utc)
    data_points = [{"timestamp": (now - timedelta(minutes=i)).isoformat(), "value": round(random.uniform(10,70),2)} for i in range(5)]
    return {
        "metrics": {
            metric or "cpu_usage": {
                "current": data_points[0]["value"],
                "average": sum(dp["value"] for dp in data_points)/len(data_points),
                "peak": max(dp["value"] for dp in data_points),
                "data_points": list(reversed(data_points))
            }
        },
        "time_range": {"start": data_points[-1]["timestamp"], "end": data_points[0]["timestamp"]}
    }
