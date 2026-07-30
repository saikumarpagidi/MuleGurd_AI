# ============================================================
# MuleGuard.AI - Developer Hooks Installer (Windows PowerShell)
#
# Run once after cloning the repository to install:
#   - pre-commit hooks (credential scanning, linting)
#
# Usage: .\scripts\install_hooks.ps1
# ============================================================

Write-Host ""
Write-Host "MuleGuard.AI - Developer Hooks Setup" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# ── Check prerequisites ──────────────────────────────────
$pythonOk = $false
try {
    $pyVersion = python --version 2>&1
    if ($pyVersion -match "Python 3") {
        Write-Host "  [OK] $pyVersion" -ForegroundColor Green
        $pythonOk = $true
    }
} catch {
    Write-Host "  [FAIL] Python not found. Install Python 3.11 first." -ForegroundColor Red
    exit 1
}

$gitOk = $false
try {
    git --version | Out-Null
    Write-Host "  [OK] Git found" -ForegroundColor Green
    $gitOk = $true
} catch {
    Write-Host "  [FAIL] Git not found." -ForegroundColor Red
    exit 1
}

if (-not ($pythonOk -and $gitOk)) { exit 1 }

# ── Install pre-commit ───────────────────────────────────
Write-Host ""
Write-Host ">> Installing pre-commit..." -ForegroundColor Magenta

pip install pre-commit detect-secrets ruff --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [FAIL] pip install failed." -ForegroundColor Red
    exit 1
}
Write-Host "  [OK] pre-commit installed" -ForegroundColor Green

# ── Install hooks ────────────────────────────────────────
Write-Host ""
Write-Host ">> Installing git hooks from .pre-commit-config.yaml..." -ForegroundColor Magenta

pre-commit install
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [FAIL] Hook installation failed." -ForegroundColor Red
    exit 1
}
Write-Host "  [OK] Git hooks installed" -ForegroundColor Green

# ── Create initial secrets baseline ─────────────────────
Write-Host ""
Write-Host ">> Creating initial secrets baseline..." -ForegroundColor Magenta

if (-not (Test-Path ".secrets.baseline")) {
    detect-secrets scan `
        --exclude-files '\.env\.example$' `
        --exclude-files '\.gitignore$' `
        > .secrets.baseline
    Write-Host "  [OK] .secrets.baseline created" -ForegroundColor Green
} else {
    Write-Host "  [INFO] .secrets.baseline already exists (skipped)" -ForegroundColor Yellow
}

# ── Run a quick validation test ──────────────────────────
Write-Host ""
Write-Host ">> Running hooks on all existing files..." -ForegroundColor Magenta
pre-commit run --all-files
$hookResult = $LASTEXITCODE

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
if ($hookResult -eq 0) {
    Write-Host "  Setup complete. Hooks are active." -ForegroundColor Green
    Write-Host "  Any future 'git commit' will run:" -ForegroundColor Cyan
    Write-Host "    - detect-secrets (block credentials)" -ForegroundColor White
    Write-Host "    - ruff (Python linting)" -ForegroundColor White
    Write-Host "    - standard file hygiene checks" -ForegroundColor White
} else {
    Write-Host "  Hooks installed but found issues in existing code." -ForegroundColor Yellow
    Write-Host "  Review output above and fix before committing." -ForegroundColor Yellow
}
Write-Host ""
