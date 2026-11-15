$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$Host.UI.RawUI.WindowTitle = 'Prabhat - T01 to T08'
Start-Transcript -Path 'results/o1_t01_t08_20261028/terminal.log' -Append
try {
 & .\Prabhat\Scripts\python.exe -u audit_message_operator.py
 if ($LASTEXITCODE -ne 0) { throw 'Operator audit failed' }
 Write-Host 'Audit complete. Waiting for reviewed training specification.'
 while (-not (Test-Path 'results/o1_t01_t08_20261028/release_training.txt')) { Start-Sleep -Seconds 3 }
 & .\start_o1_selection.ps1
 if (-not $?) { throw 'Selection session failed' }
} catch { $_ | Out-String | Set-Content 'results/o1_t01_t08_20261028/terminal_error.txt'; Write-Host $_ -ForegroundColor Red }
finally { Stop-Transcript }
