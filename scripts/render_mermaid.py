#!/usr/bin/env python3
"""Render Mermaid diagrams in docs to SVG using mermaid-cli if available.
Falls back silently if mmdc is not installed.
"""
from __future__ import annotations
import subprocess, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
OUT = ROOT / "docs" / "_diagrams"

MERMAID_EXTS = {".mmd", ".mermaid"}


def find_mermaid_code_blocks(md_path: Path):
    code = md_path.read_text(encoding="utf-8")
    blocks = []
    inside = False
    lines = []
    for line in code.splitlines():
        if line.strip().startswith("```mermaid") and not inside:
            inside = True
            lines = []
            continue
        if inside and line.strip().startswith("```"):
            inside = False
            if lines:
                blocks.append("\n".join(lines))
            continue
        if inside:
            lines.append(line)
    return blocks


def ensure_tool():
    return shutil.which("mmdc") is not None


def render_block(block: str, out_file: Path):
    tmp_in = out_file.with_suffix(".tmp.mmd")
    tmp_in.write_text(block, encoding="utf-8")
    try:
        subprocess.run(["mmdc", "-i", str(tmp_in), "-o", str(out_file)], check=True)
    finally:
        try:
            tmp_in.unlink()
        except Exception:
            pass


def main():
    if not ensure_tool():
        print("mermaid-cli not installed; skipping diagram rendering", file=sys.stderr)
        return 0
    OUT.mkdir(exist_ok=True)
    count = 0
    for md in DOCS.rglob("*.md"):
        blocks = find_mermaid_code_blocks(md)
        for idx, blk in enumerate(blocks, 1):
            name = md.stem + f"_{idx}.svg"
            out_file = OUT / name
            render_block(blk, out_file)
            count += 1
    print(f"Rendered {count} diagrams to {OUT}")
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
