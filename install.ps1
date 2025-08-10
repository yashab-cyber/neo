<#
NEO Installation Script (PowerShell)

Usage examples:
  powershell -ExecutionPolicy Bypass -File install.ps1                 # dev (editable) install
  powershell -ExecutionPolicy Bypass -File install.ps1 -Prod            # prod install
  powershell -ExecutionPolicy Bypass -File install.ps1 -Upgrade         # upgrade existing env
  powershell -ExecutionPolicy Bypass -File install.ps1 -VenvDir .env    # custom venv dir
  powershell -ExecutionPolicy Bypass -File install.ps1 -Python py      # custom python command

Parameters:
  -Prod       : Production mode (no dev extras, non-editable)
  -Upgrade    : Upgrade packages
  -VenvDir    : Virtual environment directory (default .venv)
  -Python     : Python executable (default first found python / py)
  -Help       : Show help

Notes:
  - Requires Python >= 3.11
  - Creates / reuses a virtual environment
  - Initializes database tables (idempotent)
  - Performs a CLI smoke test (neo version)
#>
[CmdletBinding()] param(
    [switch]$Prod,
    [switch]$Upgrade,
    [string]$VenvDir = ".venv",
    [string]$Python = "",
    [switch]$Help
)

if ($Help) { Get-Content -Path $MyInvocation.MyCommand.Path | Select-String -Pattern '^#|^<#$' | ForEach-Object { $_.Line }; exit 0 }

function Write-Info($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

# Discover python if not provided
if (-not $Python -or $Python -eq "") {
    $candidates = @('python', 'python3', 'py')
    foreach ($c in $candidates) { if (Get-Command $c -ErrorAction SilentlyContinue) { $Python = $c; break } }
}
if (-not $Python) { Fail "Python not found in PATH" }

# Version check
$pyVer = & $Python -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
if (-not $pyVer) { Fail "Failed to invoke Python ($Python)" }
$minMajor = 3; $minMinor = 11
$parts = $pyVer.Split('.') | ForEach-Object {[int]$_}
if ($parts[0] -lt $minMajor -or ($parts[0] -eq $minMajor -and $parts[1] -lt $minMinor)) {
    Fail "Python 3.11+ required (found $pyVer)"
}

$mode = if ($Prod) { 'prod' } else { 'dev' }
Write-Info "Python: $Python ($pyVer)"
Write-Info "Mode: $mode"
Write-Info "Venv: $VenvDir"

if (-not (Test-Path $VenvDir)) {
    Write-Info "Creating virtual environment"
    & $Python -m venv $VenvDir || Fail "venv creation failed"
}

# Activate venv
$activate = Join-Path $VenvDir 'Scripts' 'Activate.ps1'
if (-not (Test-Path $activate)) { Fail "Activation script not found: $activate" }
. $activate

pip install --upgrade pip setuptools wheel | Out-Null

if ($mode -eq 'prod') {
    if ($Upgrade) { Write-Info "Installing (prod, upgrade)"; pip install --upgrade . } else { Write-Info "Installing (prod)"; pip install . }
} else {
    if ($Upgrade) { Write-Info "Installing (dev, upgrade)"; pip install --upgrade -e .[dev] } else { Write-Info "Installing (dev)"; pip install -e .[dev] }
}
if ($LASTEXITCODE -ne 0) { Fail "pip install failed" }

# Determine sqlite path & init DB
$dbPath = & $Python - <<'PY'
from neo.config import settings
u = settings.database_url
import pathlib, sys
if u.startswith('sqlite') and 'memory' not in u:
    p = u.split('///',1)[1]
    print(p)
PY
if ($dbPath) {
  $dir = Split-Path $dbPath -Parent
  if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
}

Write-Info "Initializing database schema"
& $Python - <<'PY'
from neo.db import Base, engine
import asyncio
async def run():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
asyncio.run(run())
print('DB initialized')
PY

# Smoke test CLI
$neoCmd = Get-Command neo -ErrorAction SilentlyContinue
if ($neoCmd) {
    Write-Info "CLI version:"; neo version
} else {
    Write-Info "CLI entry not on PATH in current session; using module"
    & $Python -m neo version
}

Write-Info "Installation complete"
Write-Host "Next steps:" -ForegroundColor Green
Write-Host "  . $VenvDir\\Scripts\\Activate.ps1" -ForegroundColor Green
Write-Host "  neo serve --reload" -ForegroundColor Green
Write-Host "  neo chat 'hello'" -ForegroundColor Green
