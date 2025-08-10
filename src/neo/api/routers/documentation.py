from __future__ import annotations
from fastapi import APIRouter
from pathlib import Path
from neo.config import settings

router = APIRouter()

# Resolve docs root relative to project root (src/neo/api/routers -> up 4 levels to repo root)
DOC_ROOT = Path(__file__).resolve().parents[4] / "docs"


def _count_md(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*.md") if p.name.lower() != "readme.md")


@router.get("/documentation/index")
async def documentation_index():
    manual_dir = DOC_ROOT / "manual"
    technical_dir = DOC_ROOT / "technical"
    research_dir = DOC_ROOT / "research"
    user_guide_dir = DOC_ROOT / "user-guide"

    data = {
        "version": settings.version,
        "sections": {
            "manual": {
                "pages": _count_md(manual_dir),
                "path": str(manual_dir),
            },
            "technical": {
                "pages": _count_md(technical_dir),
                "path": str(technical_dir),
            },
            "research": {
                "pages": _count_md(research_dir),
                "path": str(research_dir),
            },
            "user_guide": {
                "pages": _count_md(user_guide_dir),
                "path": str(user_guide_dir),
            },
        },
        "totals": {},
    }
    data["totals"]["all_pages"] = sum(s["pages"] for s in data["sections"].values())
    return data
