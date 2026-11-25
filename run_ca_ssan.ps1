# Run from any directory; uses the repository-local virtual environment.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$stamp = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + $PID
$run = "runs/ca_smoke_$stamp"
$hex = "runs/ca_hex_$stamp"
$log = "logs/ca_ssan_$stamp.log"
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
New-Item -ItemType Directory -Force -Path logs | Out-Null
Start-Transcript -Path $log
$code = 0
function Invoke-Step {
    param([string]$Name, [string]$Exe, [string[]]$Arguments)
    Write-Host "`n===== $Name =====" -ForegroundColor Cyan
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Name failed (exit $LASTEXITCODE)" }
    Write-Host "$Name PASSED" -ForegroundColor Green
}
try {
    Invoke-Step 'Git branch' 'git' @('switch','ca-ssan-improved')
    Invoke-Step 'Git update' 'git' @('pull','--ff-only','origin','ca-ssan-improved')
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Missing .venv/Scripts/python.exe' }
    Invoke-Step 'Dependencies' $python @('-c','import torch,numpy,scipy,sklearn,wfdb; print(torch.__version__)')
    Invoke-Step 'Dataset check' $python @('-c',"import wfdb; print(wfdb.rdrecord('SSANproject/Modeltrain/mitdb/100').p_signal.shape)")
    $base = 'SSANproject/Modeltrain'
    Invoke-Step 'Syntax check' $python @('-m','py_compile',"$base/ca_ssan_model.py","$base/train_ca_ssan.py","$base/export_weights.py","$base/fixed_point_reference.py","$base/verify_hex_weights.py")
    Invoke-Step 'Two-epoch smoke training' $python @("$base/train_ca_ssan.py",'--data',"$base/mitdb",'--epochs','2','--limit','512','--variant','full','--out',$run)
    Invoke-Step 'HEX export' $python @("$base/export_weights.py",'--checkpoint',"$run/model.pt",'--out',$hex,'--data',"$base/mitdb")
    Invoke-Step 'HEX verification' $python @("$base/verify_hex_weights.py",'--checkpoint',"$run/model.pt",'--dir',$hex)
    Get-Content -LiteralPath "$run/metrics.json"
    Write-Host 'ALL SMOKE TESTS PASSED' -ForegroundColor Green
} catch {
    $code = 1
    Write-Host "AUTOMATION FAILED: $($_.Exception.Message)" -ForegroundColor Red
} finally {
    Write-Host "Training folder: $run"
    Write-Host "HEX folder: $hex"
    Write-Host "Log: $log"
    Stop-Transcript
}
exit $code
