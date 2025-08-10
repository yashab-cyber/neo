from __future__ import annotations
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from neo.config import settings
from pathlib import Path

Base = declarative_base()


def _ensure_dir(url: str):
    if url.startswith("sqlite") and ":///" in url and "memory" not in url:
        path_part = url.split("///", 1)[1]
        if path_part and "/" in path_part:
            Path(path_part).parent.mkdir(parents=True, exist_ok=True)


_ensure_dir(settings.database_url)
engine = create_async_engine(settings.database_url, future=True, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session():
    async with AsyncSessionLocal() as session:
        yield session
