param([int]$Epochs=10,[int]$TeacherEpochs=10,[int]$Pool=10,[int]$Width=96,[switch]$Sweep,[switch]$Smote)
$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
$stamp=(Get-Date -Format 'yyyyMMdd_HHmmss')+'_'+$PID
$py=Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$b='SSANproject/Modeltrain'
$out="runs/ca_v2_$stamp"
New-Item -ItemType Directory -Force -Path logs | Out-Null
Start-Transcript -Path "logs/ca_v2_$stamp.log"
$code=0
try {
    & $py -m py_compile "$b/ca_ssan_v2.py" "$b/train_ca_ssan_v2.py"
    if ($LASTEXITCODE -ne 0) { throw 'Syntax check failed' }
    $argv=@("$b/train_ca_ssan_v2.py",'--data',"$b/mitdb",'--out',$out,'--epochs',"$Epochs",'--teacher-epochs',"$TeacherEpochs",'--pool',"$Pool",'--width',"$Width")
    if ($Sweep) { $argv+='--sweep' }
    if ($Smote) { $argv+='--smote' }
    & $py @argv
    if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
    Import-Csv "$out/comparison.csv" | Format-Table -AutoSize
} catch { $code=1; Write-Host "FAILED: $_" -ForegroundColor Red }
finally { Write-Host "Results: $out"; Stop-Transcript }
exit $code
