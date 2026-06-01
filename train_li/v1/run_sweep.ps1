# run_sweep.ps1
# PowerShell launcher for V1 GRU training sweep.
# Run from PowerShell:  .\run_sweep.ps1

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$Python  = "D:\Software\miniconda3\envs\dl_lab1\python.exe"
$Root    = Split-Path -Parent $ScriptDir          # LAB5/
$Shared  = Join-Path $Root "shared"
$PData   = Join-Path $Root "processed"

# ── Step 0: Check / build preprocessed data ──
if (-not (Test-Path (Join-Path $PData "all_data.parquet"))) {
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host " Step 0a: Preprocessing raw CSVs  ->  all_data.parquet" -ForegroundColor Yellow
    Write-Host "========================================"
    & $Python (Join-Path $Shared "preprocess.py")
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: preprocess.py failed" -ForegroundColor Red
        exit 1
    }
}

if (-not (Test-Path (Join-Path $PData "windows"))) {
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host " Step 0b: Building windows  ->  processed/windows/" -ForegroundColor Yellow
    Write-Host "========================================"
    & $Python (Join-Path $Shared "build_windows.py")
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: build_windows.py failed" -ForegroundColor Red
        exit 1
    }
}

# ── Phase 1: Coarse sweep (T x H) ──
Write-Host "`n========================================" -ForegroundColor Green
Write-Host " Phase 1: window_size x hidden_size" -ForegroundColor Green
Write-Host " (9 configs, ~1.5 hours)" -ForegroundColor Green
Write-Host "========================================"
& $Python train.py --sweep phase1
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: phase1 had errors, continuing ..." -ForegroundColor Yellow
}

# ── Phase 2: Fine sweep (dropout x lr) ──
Write-Host "`n========================================" -ForegroundColor Green
Write-Host " Phase 2: dropout x lr" -ForegroundColor Green
Write-Host " (9 configs, ~1.5 hours)" -ForegroundColor Green
Write-Host "========================================"
& $Python train.py --sweep phase2
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: phase2 had errors, continuing ..." -ForegroundColor Yellow
}

# ── Phase 3: Layer refinement ──
Write-Host "`n========================================" -ForegroundColor Green
Write-Host " Phase 3: num_layers" -ForegroundColor Green
Write-Host " (3 configs, ~30 min)" -ForegroundColor Green
Write-Host "========================================"
& $Python train.py --sweep phase3
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: phase3 had errors, continuing ..." -ForegroundColor Yellow
}

# ── Final: best model + backtest ──
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " Final: best-params training + backtest" -ForegroundColor Cyan
Write-Host "========================================"
& $Python train.py --best

Write-Host "`n========================================" -ForegroundColor Green
Write-Host " All done! Results in:" -ForegroundColor Green
Write-Host "   $ScriptDir\results.csv" -ForegroundColor White
Write-Host "   $ScriptDir\best_params.json" -ForegroundColor White
Write-Host "========================================"
