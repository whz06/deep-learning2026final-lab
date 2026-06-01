# v5/run_step1.ps1 — Step 1: Rebuild all_data.parquet, build windows, train GRU
# Usage: powershell -File run_step1.ps1

$Python = "D:\Software\miniconda3\envs\dl_lab1\python.exe"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " STEP 1: Fix Features + Retrain Baseline GRU" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

# ---- Step 0: Rebuild all_data.parquet with new columns ----
Write-Host "`n[Step 0] Rebuilding all_data.parquet (adding vwap, pe, pb, circ_mv) ..." -ForegroundColor Yellow
$PreprocessScript = Join-Path $Root "shared" "preprocess.py"
& $Python $PreprocessScript
if ($LASTEXITCODE -ne 0) {
    Write-Host "Step 0 FAILED" -ForegroundColor Red
    exit 1
}

# ---- Step 1: Build v5_windows with fixed normalization ----
Write-Host "`n[Step 1] Building v5_windows (fixed 2-stage normalization, 26 dims) ..." -ForegroundColor Yellow
$BuildScript = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "build_windows.py"
& $Python $BuildScript
if ($LASTEXITCODE -ne 0) {
    Write-Host "Step 1 FAILED" -ForegroundColor Red
    exit 1
}

# ---- Step 2: Train baseline GRU on v5 features ----
Write-Host "`n[Step 2] Training baseline GRU (26 dims, H=128 L=1 D=0.2) ..." -ForegroundColor Yellow
$TrainScript = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "train.py"
& $Python $TrainScript
if ($LASTEXITCODE -ne 0) {
    Write-Host "Step 2 FAILED" -ForegroundColor Red
    exit 1
}

Write-Host "`n================================================================" -ForegroundColor Green
Write-Host " STEP 1 COMPLETE" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green

$ResultsCsv = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "results" "results.csv"
if (Test-Path $ResultsCsv) {
    Write-Host "`nTraining results:" -ForegroundColor Green
    Import-Csv $ResultsCsv | Format-Table name, best_val_rankic, best_epoch, time_min -AutoSize
}
