from fastapi import APIRouter, UploadFile, File, Form
from typing import List
import os, pathlib, shutil, stat, time

router = APIRouter()


@router.get("/files")
async def list_files(path: str = ".", recursive: bool = False, limit: int = 100, offset: int = 0):
    base = pathlib.Path(path)
    items = []
    if recursive:
        iterator = base.rglob("*")
    else:
        iterator = base.glob("*")
    for p in iterator:
        if len(items) >= limit + offset:
            break
        info = p.stat()
        if len(items) >= offset:
            items.append({
                "name": p.name,
                "type": "dir" if p.is_dir() else "file",
                "size": info.st_size,
                "modified": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(info.st_mtime)),
                "permissions": stat.filemode(info.st_mode),
            })
    return {"path": str(base.resolve()), "total_items": len(items), "items": items[offset:offset+limit], "pagination": {"limit": limit, "offset": offset, "has_more": False}}


@router.post("/files/upload")
async def upload_file(file: UploadFile = File(...), path: str = Form("."), overwrite: bool = Form(False)):
    dest_dir = pathlib.Path(path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / file.filename
    if dest_file.exists() and not overwrite:
        return {"error": "FILE_EXISTS"}
    with dest_file.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"status": "uploaded", "path": str(dest_file)}
