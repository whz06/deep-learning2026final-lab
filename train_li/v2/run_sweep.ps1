# run_sweep.ps1 — V2 multi-model sweep with resume support
# Run:  .\run_sweep.ps1
# Resume: .\run_sweep.ps1 -Resume  (skips already-completed configs)

param([switch]$Resume)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
$Python = "D:\Software\miniconda3\envs\dl_lab1\python.exe"
$Root   = Split-Path -Parent $ScriptDir
$V2Data = Join-Path $Root "processed\v2_windows"

# ── Step 0: build v2 windows ──
if (-not (Test-Path $V2Data)) {
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host " Step 0: Building v2 windows" -ForegroundColor Yellow
    Write-Host "========================================"
    & $Python build_windows.py
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED" -ForegroundColor Red; exit 1 }
}

# ── Results file path ──
$ResultsCSV = Join-Path $ScriptDir "results.csv"
if ($Resume -and (Test-Path $ResultsCSV)) {
    $done = (Get-Content $ResultsCSV | Select-Object -Skip 1).Count
    Write-Host "[resume] $done configs already done, skipping those" -ForegroundColor Cyan
} elseif (Test-Path $ResultsCSV) {
    Write-Host "[fresh] Removing old results.csv" -ForegroundColor Cyan
    Remove-Item $ResultsCSV -Force
}

function Run-Train {
    param($Model, $SweepArg = "")
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host " Training: $Model $SweepArg" -ForegroundColor Green
    Write-Host "========================================"
    $args = @("train.py", "--model", $Model)
    if ($SweepArg) { $args += "--sweep", $SweepArg }
    & $Python @args
    $LASTEXITCODE = 0
}

# ── Phase 1: MLP ──
Run-Train "mlp"

# ── Phase 2: GRU ──
Run-Train "gru"

# ── Phase 3: Transformer ──
Run-Train "tf" "phase1"
Run-Train "tf" "phase2"

# ── Phase 4: Ensemble ──
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " Ensemble: save scores + sweep weights" -ForegroundColor Cyan
Write-Host "========================================"
& $Python ensemble.py --stage save_scores
& $Python ensemble.py --stage sweep

# ── Phase 5: Backtest ──
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " Backtest" -ForegroundColor Cyan
Write-Host "========================================"
& $Python backtest.py --model gru --ckpt gru_best.pt

Write-Host "`n========================================" -ForegroundColor Green
Write-Host " V2 Done! Results: $ResultsCSV" -ForegroundColor Green
Write-Host "========================================"
