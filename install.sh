#!/usr/bin/env bash
set -euo pipefail

# NEO Installation Script
# Usage:
#   ./install.sh                # default dev install (editable, dev extras)
#   ./install.sh --prod         # production install (no dev extras)
#   ./install.sh --upgrade      # upgrade dependencies in existing venv
#   ./install.sh --help         # show help
# Options can be combined (e.g. --prod --upgrade)

PROJECT_NAME="neo-assistant"
PY_MIN="3.11"
VENV_DIR=".venv"
MODE="dev"
UPGRADE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prod) MODE="prod"; shift ;;
    --dev) MODE="dev"; shift ;;
    --upgrade) UPGRADE="true"; shift ;;
    --venv-dir) VENV_DIR="$2"; shift 2 ;;
    --python) PY_BIN="$2"; shift 2 ;;
    --help|-h)
      grep '^# ' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

# Choose python
if [[ -z "${PY_BIN:-}" ]]; then
  for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1; then PY_BIN="$c"; break; fi
  done
fi
if [[ -z "${PY_BIN:-}" ]]; then echo "Python not found" >&2; exit 1; fi

# Version check
PY_VER=$($PY_BIN - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
verlte() { [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$1" ]; }
if ! verlte "$PY_MIN" "$PY_VER"; then
  echo "Python $PY_MIN+ required (found $PY_VER)" >&2; exit 1
fi

echo "==> Using python: $PY_BIN ($PY_VER)"
echo "==> Mode: $MODE"
echo "==> Venv: $VENV_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "==> Creating virtual environment"
  "$PY_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

if [[ "$MODE" = "prod" ]]; then
  if [[ "$UPGRADE" = "true" ]]; then
    echo "==> Installing (prod, upgrade)"
    pip install --upgrade .
  else
    echo "==> Installing (prod)"
    pip install .
  fi
else
  if [[ "$UPGRADE" = "true" ]]; then
    echo "==> Installing (dev, upgrade)"
    pip install --upgrade -e .[dev]
  else
    echo "==> Installing (dev)"
    pip install -e .[dev]
  fi
fi

# Create data directory for sqlite if needed
DB_PATH=$(python - <<'PY'
from neo.config import settings
import os
url=settings.database_url
if url.startswith('sqlite') and 'memory' not in url:
    path=url.split('///',1)[1]
    print(path)
PY
)
if [[ -n "$DB_PATH" ]]; then
  mkdir -p "$(dirname "$DB_PATH")"
fi

# Initialize database tables (idempotent)
python - <<'PY'
from neo.db import Base, engine
import asyncio
async def main():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
asyncio.run(main())
print("DB initialized")
PY

# Smoke test CLI
if command -v neo >/dev/null 2>&1; then
  echo "==> CLI version:" && neo version || true
else
  echo "(neo entry point not on PATH yet in this shell; run 'source $VENV_DIR/bin/activate')"
fi

echo "==> Installation complete"
echo "Next steps:"
echo "  source $VENV_DIR/bin/activate" 
echo "  neo serve --reload  # start API"
echo "  neo chat 'hello'    # test chat"
