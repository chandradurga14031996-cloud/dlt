param([int]$Epochs=10,[int]$Patience=15,[switch]$Sweep)
$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
$stamp=(Get-Date -Format 'yyyyMMdd_HHmmss')+'_'+$PID
$root="runs/ca_compare_$stamp"
$py=Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$b='SSANproject/Modeltrain'
New-Item -ItemType Directory -Force -Path logs,$root | Out-Null
Start-Transcript -Path "logs/ca_compare_$stamp.log"
function Run-Step($exe,[string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "Failed: $exe ($LASTEXITCODE)" }
}
$code=0
try {
    Run-Step git @('switch','ca-ssan-improved')
    Run-Step git @('pull','--ff-only','origin','ca-ssan-improved')
    Run-Step $py @('-m','py_compile',"$b/train_ca_ssan.py")
    $common=@("$b/train_ca_ssan.py",'--data',"$b/mitdb",'--epochs',"$Epochs",'--patience',"$Patience",'--schedule','--variant','full','--seed','42')
    $cases=@(@{name='baseline';extra=@()},@{name='smote';extra=@('--smote','--ratios','0.25','0.35','0.15')})
    if ($Sweep) {
        $cases+=@(@{name='smote10';extra=@('--smote','--ratios','0.1','0.1','0.1')},@{name='smote25';extra=@('--smote','--ratios','0.25','0.25','0.25')},@{name='smote50';extra=@('--smote','--ratios','0.5','0.5','0.5')})
    }
    $rows=@()
    foreach ($c in $cases) {
        $name=$c.name
        Run-Step $py ($common+@('--out',"$root/$name")+$c.extra)
        $m=Get-Content "$root/$name/metrics.json" -Raw | ConvertFrom-Json
        $r=$m.validation.report
        $rows += [pscustomobject]@{Model=$name;Accuracy=$m.validation.accuracy;MacroF1=$m.validation.macro_f1;N_F1=$r.N.'f1-score';S_F1=$r.S.'f1-score';V_F1=$r.V.'f1-score';F_F1=$r.F.'f1-score';N_Recall=$r.N.recall;S_Recall=$r.S.recall;V_Recall=$r.V.recall;F_Recall=$r.F.recall;AvgSteps=$m.validation.avg_steps;BestEpoch=$m.best_epoch;EpochsRun=$m.history.Count}
        $rows | Export-Csv "$root/comparison.csv" -NoTypeInformation
    }
    $rows | Sort-Object MacroF1 -Descending | Format-Table -AutoSize
    $winner=($rows | Sort-Object MacroF1 -Descending | Select-Object -First 1).Model
    Write-Host "Validation winner: $winner"
    foreach ($name in @('baseline',$winner) | Select-Object -Unique) {
        Run-Step $py @("$b/export_weights.py",'--checkpoint',"$root/$name/model.pt",'--out',"$root/${name}_hex",'--data',"$b/mitdb")
        Run-Step $py @("$b/verify_hex_weights.py",'--checkpoint',"$root/$name/model.pt",'--dir',"$root/${name}_hex")
    }
    Write-Host 'COMPARISON AND HEX VERIFICATION PASSED'
} catch { $code=1; Write-Host "FAILED: $_" -ForegroundColor Red }
finally { Write-Host "Results: $root"; Stop-Transcript }
exit $code
