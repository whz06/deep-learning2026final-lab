# run_all.ps1 — V3: Strategy C, A+C, and multi-task D training + comparison

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
$Python = "D:\Software\miniconda3\envs\dl_lab1\python.exe"

Write-Host "========================================" -ForegroundColor Green
Write-Host " V3: Multi-strategy comparison" -ForegroundColor Green
Write-Host "========================================"

# Step 1: C & A+C strategy sweeps
Write-Host "`n[1/3] Strategy C & A+C sweeps ..." -ForegroundColor Cyan
& $Python run_strategies.py
$LASTEXITCODE = 0

# Step 2: Train multi-task D model
Write-Host "`n[2/3] Training multi-task D (sweep + best) ..." -ForegroundColor Cyan
& $Python train_D.py --sweep
$LASTEXITCODE = 0
& $Python train_D.py --best
$LASTEXITCODE = 0

# Step 3: Evaluate D model
Write-Host "`n[3/3] Evaluating D model ..." -ForegroundColor Cyan
& $Python strategy_D.py
$LASTEXITCODE = 0

Write-Host "`n========================================" -ForegroundColor Green
Write-Host " All done. Results:"
Write-Host "   C & A+C: run_strategies.py output above"
Write-Host "   D sweep: d_sweep.csv"
Write-Host "   D model: checkpoints/multitask_best.pt"
Write-Host "========================================"
