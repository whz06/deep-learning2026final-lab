# run_test.ps1
# Walk-forward test: 10 segments of 10 days on 2026 data

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$Python = "D:\Software\miniconda3\envs\dl_lab1\python.exe"

Write-Host "========================================" -ForegroundColor Green
Write-Host " Walk-Forward Test (2026 Feb-May)" -ForegroundColor Green
Write-Host "========================================"

& $Python walkforward.py

Write-Host "`n========================================" -ForegroundColor Green
Write-Host " Done." -ForegroundColor Green
Write-Host "========================================"
