"""Convenience launcher for the NEO FastAPI application.

Features added beyond the minimal original script:
 - Argument parsing (host, port, reload, workers, log-level, root-path)
 - Environment variable overrides (NEO_HOST / HOST, NEO_PORT / PORT, NEO_RELOAD, NEO_WORKERS, NEO_LOG_LEVEL)
 - Lightweight .env loader (if a .env file exists in project root)
 - Graceful shutdown handling with informative banner
 - Mirrors `neo serve` defaults while remaining a single-file entry point

Recommended production usage: prefer a process manager (systemd, docker, supervisord)
or the CLI `neo serve`. This script is primarily for quick local runs.
"""
from __future__ import annotations
import argparse, os, signal, sys, textwrap
import uvicorn
from neo import create_app
from neo.config import settings

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no external dependency)."""
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return
    try:
        with open(full, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                os.environ.setdefault(k, v)
    except Exception as e:  # pragma: no cover - non critical
        print(f"[run.py] Warning: failed loading .env: {e}", file=sys.stderr)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the NEO API server (quick launcher).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default=os.getenv("NEO_HOST", os.getenv("HOST", "0.0.0.0")), help="Bind host address")
    p.add_argument("--port", type=int, default=int(os.getenv("NEO_PORT", os.getenv("PORT", "8000"))), help="Bind port")
    p.add_argument("--reload", action="store_true", default=os.getenv("NEO_RELOAD", "false").lower() == "true", help="Enable auto-reload (development)")
    p.add_argument("--workers", type=int, default=int(os.getenv("NEO_WORKERS", "1")), help="Worker processes (ignored with --reload)")
    p.add_argument("--log-level", default=os.getenv("NEO_LOG_LEVEL", settings.log_level.lower()), help="Log level")
    p.add_argument("--root-path", default=os.getenv("NEO_ROOT_PATH", ""), help="ASGI root_path (if behind a reverse proxy)")
    return p.parse_args(argv)


def banner(args: argparse.Namespace) -> str:
    return textwrap.dedent(
        f"""
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ NEO Assistant API                                  ┃
        ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
        ┃ Version : {settings.version:<42} ┃
        ┃ Env     : {settings.environment:<42} ┃
        ┃ Host    : {args.host:<42} ┃
        ┃ Port    : {args.port:<42} ┃
        ┃ Reload  : {str(args.reload):<42} ┃
        ┃ Workers : {args.workers:<42} ┃
        ┃ LogLvl  : {args.log_level:<42} ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
        """.rstrip()
    )


_should_exit = False


def _handle_signal(signum, frame):  # pragma: no cover - signal handling
    global _should_exit
    if not _should_exit:
        print(f"\n[run.py] Received signal {signum}; shutting down gracefully...", file=sys.stderr)
        _should_exit = True


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    # Register signals
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except Exception:  # pragma: no cover - some platforms
            pass

    print(banner(args))
    config = uvicorn.Config(
        app=create_app(),
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level,
        root_path=args.root_path or None,
    )
    server = uvicorn.Server(config)
    return 0 if server.run() else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
