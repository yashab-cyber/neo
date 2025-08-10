"""NEO CLI - Operational & Development Utilities.

Top-level commands (run `neo <cmd> -h` for details):
    serve               Run API server (uvicorn)
    config-validate     Validate a YAML config against internal schema
    token               Issue a JWT from local app factory
    status              System status snapshot
    learning-status     Learning data layer heuristic status
    cognitive           Architecture / AI engine status groups
    commands            Execute / manage queued commands
    knowledge           Knowledge graph operations (memory or persist backends)
    memory              Local memory service utilities (stages, pruning, etc.)
    version             Show version & environment

Notes:
 - HTTP-based commands spin up an in-process FastAPI app per invocation and require no external server.
 - Memory & knowledge operations can target either the in-memory or persistent backend (if implemented) using --backend.
 - JSON inputs are accepted with --props / --content / --params; wrap them in single quotes for shell safety.
"""
from __future__ import annotations
import argparse, sys, asyncio, json, contextlib
from pathlib import Path
from typing import Any

import uvicorn  # type: ignore
import yaml  # type: ignore

from neo import create_app
from neo.config import settings
from neo.config_schema import validate_config
from fastapi.testclient import TestClient
from neo.services.memory_service import MemoryService
from neo.db import AsyncSessionLocal, engine, Base


def _print(obj: Any):  # lightweight pretty output
    if isinstance(obj, (dict, list)):
        print(json.dumps(obj, indent=2, sort_keys=True))
    else:
        print(obj)


def _parse_json(maybe: str | None, default: Any = None) -> Any:
    if maybe is None:
        return default
    try:
        return json.loads(maybe)
    except Exception:  # noqa: BLE001
        print(f"Invalid JSON: {maybe}", file=sys.stderr)
        sys.exit(2)


def cmd_serve(args: argparse.Namespace) -> None:  # pragma: no cover - runtime path
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, log_level=settings.log_level.lower())


def cmd_config_validate(args: argparse.Namespace) -> None:
    path = Path(args.file)
    if not path.exists():
        print(f"Config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    try:
        model = validate_config(data)
    except Exception as e:  # noqa: BLE001
        print("INVALID:")
        print(str(e))
        sys.exit(2)
    print("VALID")
    if args.pretty:
        _print(model.model_dump())


@contextlib.contextmanager
def _client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def cmd_token(args: argparse.Namespace) -> None:
    role_payload = {"api_key": args.api_key}
    if args.role:
        role_payload["role"] = args.role
    with _client() as c:
        r = c.post("/api/v1/auth/token", json=role_payload)
        if r.status_code != 200:
            print(f"Error: {r.status_code} {r.text}", file=sys.stderr)
            sys.exit(1)
        data = r.json()
        if args.json:
            _print(data)
        else:
            print(data["access_token"])


def _get_token(c: TestClient, role: str | None = None) -> str:
    payload = {"api_key": settings.api_keys[0]}
    if role:
        payload["role"] = role
    return c.post("/api/v1/auth/token", json=payload).json()["access_token"]


def cmd_status(_args: argparse.Namespace) -> None:
    with _client() as c:
        token = _get_token(c)
        r = c.get("/api/v1/status", headers={"Authorization": f"Bearer {token}"})
        _print(r.json())


def cmd_learning_status(_args: argparse.Namespace) -> None:
    with _client() as c:
        token = _get_token(c, role="admin")
        r = c.get("/api/v1/knowledge/knowledge/learning/status", headers={"Authorization": f"Bearer {token}"})
        if r.status_code != 200:
            print(f"Error: {r.status_code} {r.text}", file=sys.stderr)
            sys.exit(1)
        _print(r.json())


def cmd_cognitive(args: argparse.Namespace) -> None:
    endpoint_map = {
        "ai-engine": "/api/v1/cognitive/ai-engine/status",
        "monitoring": "/api/v1/cognitive/monitoring/status",
        "security": "/api/v1/cognitive/security/status",
        "system": "/api/v1/cognitive/system/status",
    }
    path = endpoint_map[args.section]
    with _client() as c:
        token = _get_token(c, role="admin")
        r = c.get(path, headers={"Authorization": f"Bearer {token}"})
        if r.status_code != 200:
            print(f"Error: {r.status_code} {r.text}", file=sys.stderr)
            sys.exit(1)
        _print(r.json())


def cmd_chat(args: argparse.Namespace) -> None:
    with _client() as c:
        token = _get_token(c, role="user")
        headers = {"Authorization": f"Bearer {token}"}
        body = {"message": args.message, "model": args.model}
        if args.session:
            body["session_id"] = args.session
        if args.context:
            body["context"] = _parse_json(args.context, {})
        r = c.post("/api/v1/ai/chat", json=body, headers=headers)
        if r.status_code != 200:
            print(f"Error: {r.status_code} {r.text}", file=sys.stderr)
            sys.exit(1)
        _print(r.json())


def cmd_commands(args: argparse.Namespace) -> None:
    with _client() as c:
        token = _get_token(c, role="admin")
        headers = {"Authorization": f"Bearer {token}"}
        if args.action == "exec":
            payload = {"command": args.command, "parameters": _parse_json(args.params), "async_": args.async_exec}
            r = c.post("/api/v1/commands/execute", json=payload, headers=headers)
            _print(r.json())
        elif args.action == "status":
            r = c.get(f"/api/v1/commands/{args.execution_id}", headers=headers)
            _print(r.json())
        elif args.action == "batch":
            items = _parse_json(args.batch)
            r = c.post("/api/v1/commands/batch", json={"commands": items}, headers=headers)
            _print(r.json())
        else:
            print("Unknown commands action", file=sys.stderr)
            sys.exit(2)


def cmd_knowledge(args: argparse.Namespace) -> None:
    backend = args.backend
    with _client() as c:
        token = _get_token(c, role="admin")
        headers = {"Authorization": f"Bearer {token}"}
        if args.action == "node-create":
            r = c.post("/api/v1/knowledge/nodes", params={"backend": backend}, json={"type": args.type, "properties": _parse_json(args.props, {})}, headers=headers)
            _print(r.json())
        elif args.action == "node-get":
            r = c.get(f"/api/v1/knowledge/nodes/{args.id}", params={"backend": backend}, headers=headers)
            _print(r.json())
        elif args.action == "edge-create":
            body = {"source_id": args.source, "target_id": args.target, "relationship_type": args.rel, "properties": _parse_json(args.props, {})}
            r = c.post("/api/v1/knowledge/edges", params={"backend": backend}, json=body, headers=headers)
            _print(r.json())
        elif args.action == "query":
            params = {"backend": backend}
            if args.type:
                params["type"] = args.type
            if args.property_key:
                params["property_key"] = args.property_key
                params["property_value"] = args.property_value
            r = c.get("/api/v1/knowledge/query", params=params, headers=headers)
            _print(r.json())
        elif args.action == "neighbors":
            params = {"backend": backend}
            if args.rel:
                params["rel_type"] = args.rel
            r = c.get(f"/api/v1/knowledge/nodes/{args.id}/neighbors", params=params, headers=headers)
            _print(r.json())
        elif args.action == "traverse":
            params = {"max_depth": args.depth}
            if args.rel:
                params["rel_type"] = args.rel
            r = c.get(f"/api/v1/knowledge/traverse/{args.id}", params=params, headers=headers)
            _print(r.json())
        elif args.action == "status":
            r = c.get("/api/v1/knowledge/knowledge/status", headers=headers)
            _print(r.json())
        elif args.action == "learning-status":
            r = c.get("/api/v1/knowledge/knowledge/learning/status", headers=headers)
            _print(r.json())
        else:
            print("Unknown knowledge action", file=sys.stderr)
            sys.exit(2)


async def _memory_service() -> MemoryService:
    # ensure tables (idempotent)
    async with engine.begin() as conn:  # type: ignore
        await conn.run_sync(Base.metadata.create_all)
    async with AsyncSessionLocal() as session:  # type: ignore
        return MemoryService(session)


def cmd_memory(args: argparse.Namespace) -> None:
    async def run():
        if args.action == "store":
            service = await _memory_service()
            # Treat plain text as a 'note' kind with content payload
            item = await service.store(kind="note", content={"text": args.text})
            _print({"id": item.id, "content": item.content, "stage": item.stage})
        elif args.action == "list":
            service = await _memory_service()
            items = await service.list_stage("short_term")
            _print([{"id": i.id, "stage": i.stage, "content": i.content} for i in items])
        elif args.action == "working":
            service = await _memory_service()
            items = await service.working_set()
            _print([{"id": i.id, "stage": i.stage, "content": i.content} for i in items])
        elif args.action == "status":
            service = await _memory_service()
            stats = await service.stats()
            _print(stats)
        elif args.action == "consolidate":
            service = await _memory_service()
            moved = await service.consolidate()
            _print({"moved": moved})
        elif args.action == "prune-short":
            service = await _memory_service()
            deleted = await service.prune_short_term()
            _print({"deleted": deleted})
        elif args.action == "prune-forgetting":
            service = await _memory_service()
            deleted = await service.forgetting_curve_prune()
            _print({"deleted": deleted})
        elif args.action == "promote":
            service = await _memory_service()
            ok = await service.promote_to_working(args.id)
            _print({"promoted": args.id, "ok": ok})
        elif args.action == "access":
            service = await _memory_service()
            item = await service.access(args.id)
            if not item:
                print("Not found", file=sys.stderr)
                sys.exit(1)
            _print({"id": item.id, "stage": item.stage, "access_count": item.access_count})
        elif args.action == "delete":
            service = await _memory_service()
            ok = await service.delete_item(args.id)
            if not ok:
                print("Not found", file=sys.stderr)
                sys.exit(1)
            _print({"deleted": args.id})
        else:
            print("Unknown memory action", file=sys.stderr)
            sys.exit(2)
    asyncio.run(run())


def cmd_version(_args: argparse.Namespace) -> None:
    print(f"NEO {settings.version} (env={settings.environment})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="neo", description="NEO Assistant CLI")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("serve", help="Run API server")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--reload", action="store_true", help="Enable auto-reload")
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("config-validate", help="Validate a YAML config against schema")
    sp.add_argument("file")
    sp.add_argument("--pretty", action="store_true")
    sp.set_defaults(func=cmd_config_validate)

    sp = sub.add_parser("token", help="Issue a JWT using a local app instance")
    sp.add_argument("--api-key", default=settings.api_keys[0])
    sp.add_argument("--role", default=None)
    sp.add_argument("--json", action="store_true", help="Output full JSON response")
    sp.set_defaults(func=cmd_token)

    sp = sub.add_parser("status", help="Show system status")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("learning-status", help="Show learning layer completion metrics")
    sp.set_defaults(func=cmd_learning_status)

    # cognitive
    sp = sub.add_parser("cognitive", help="Cognitive architecture statuses")
    sp.add_argument("section", choices=["ai-engine", "monitoring", "security", "system"], help="Status section")
    sp.set_defaults(func=cmd_cognitive)

    sp = sub.add_parser("chat", help="Send a chat message to NEO")
    sp.add_argument("message", help="User message text")
    sp.add_argument("--context", help="Optional JSON context payload")
    sp.add_argument("--model", default="stub-model")
    sp.add_argument("--session", help="Existing session id (to maintain conversation)")
    sp.set_defaults(func=cmd_chat)

    # commands
    sp = sub.add_parser("commands", help="Execute and inspect commands")
    csub = sp.add_subparsers(dest="action", required=True)
    sp_exec = csub.add_parser("exec", help="Execute a command")
    sp_exec.add_argument("command")
    sp_exec.add_argument("--params", help="JSON parameters", default=None)
    sp_exec.add_argument("--async", dest="async_exec", action="store_true", help="Queue asynchronously")
    sp_exec.set_defaults(func=cmd_commands)
    sp_status = csub.add_parser("status", help="Check execution status")
    sp_status.add_argument("execution_id")
    sp_status.set_defaults(func=cmd_commands)
    sp_batch = csub.add_parser("batch", help="Batch execute commands (JSON list)")
    sp_batch.add_argument("batch", help='JSON list e.g. "[{\"id\":\"a\",\"command\":\"echo\"}]"')
    sp_batch.set_defaults(func=cmd_commands)

    # knowledge
    sp = sub.add_parser("knowledge", help="Knowledge graph operations")
    sp.add_argument("--backend", choices=["memory", "persist"], default="memory")
    ksub = sp.add_subparsers(dest="action", required=True)
    k_nc = ksub.add_parser("node-create", help="Create node")
    k_nc.add_argument("--type", required=True)
    k_nc.add_argument("--props", help="JSON properties")
    k_nc.set_defaults(func=cmd_knowledge)
    k_ng = ksub.add_parser("node-get", help="Get node")
    k_ng.add_argument("id")
    k_ng.set_defaults(func=cmd_knowledge)
    k_ec = ksub.add_parser("edge-create", help="Create edge")
    k_ec.add_argument("--source", required=True)
    k_ec.add_argument("--target", required=True)
    k_ec.add_argument("--rel", required=True)
    k_ec.add_argument("--props", help="JSON properties")
    k_ec.set_defaults(func=cmd_knowledge)
    k_q = ksub.add_parser("query", help="Query nodes")
    k_q.add_argument("--type")
    k_q.add_argument("--property-key")
    k_q.add_argument("--property-value")
    k_q.set_defaults(func=cmd_knowledge)
    k_nb = ksub.add_parser("neighbors", help="List neighbors")
    k_nb.add_argument("id")
    k_nb.add_argument("--rel")
    k_nb.set_defaults(func=cmd_knowledge)
    k_tr = ksub.add_parser("traverse", help="Traverse BFS")
    k_tr.add_argument("id")
    k_tr.add_argument("--depth", type=int, default=3)
    k_tr.add_argument("--rel")
    k_tr.set_defaults(func=cmd_knowledge)
    k_st = ksub.add_parser("status", help="Knowledge implementation status")
    k_st.set_defaults(func=cmd_knowledge)
    k_ls = ksub.add_parser("learning-status", help="Learning data status (knowledge)")
    k_ls.set_defaults(func=cmd_knowledge)

    sp = sub.add_parser("memory", help="Memory system utilities")
    msub = sp.add_subparsers(dest="action", required=True)
    sp_store = msub.add_parser("store", help="Store a memory item (short_term)")
    sp_store.add_argument("text")
    sp_store.set_defaults(func=cmd_memory)
    for name in ["list", "working", "status", "consolidate", "prune-short", "prune-forgetting"]:
        m = msub.add_parser(name, help=f"Memory {name} command")
        m.set_defaults(func=cmd_memory)
    m_promote = msub.add_parser("promote", help="Promote item to working set")
    m_promote.add_argument("id")
    m_promote.set_defaults(func=cmd_memory)
    m_access = msub.add_parser("access", help="Access (touch) memory item")
    m_access.add_argument("id")
    m_access.set_defaults(func=cmd_memory)
    m_delete = msub.add_parser("delete", help="Delete memory item")
    m_delete.add_argument("id")
    m_delete.set_defaults(func=cmd_memory)

    sp = sub.add_parser("version", help="Show version")
    sp.set_defaults(func=cmd_version)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except KeyboardInterrupt:  # pragma: no cover
        print("Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
