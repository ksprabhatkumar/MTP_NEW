$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$Host.UI.RawUI.WindowTitle = 'Prabhat - O1 graph repair validation pilot'
$base = 'results/o1_graph_repair_20261029'
New-Item -ItemType Directory -Force $base | Out-Null
Start-Transcript -Path "$base/terminal.log" -Append
$python = Join-Path $PSScriptRoot 'Prabhat\Scripts\python.exe'
try {
 & $python -u check_graph_repair.py --output "$base/smoke"
 if ($LASTEXITCODE -ne 0) { throw 'Repair preflight failed' }
 & $python -u run_graph_repair.py --output "$base/smoke" --epochs 4 --subjects 2 --seeds 13 --batch-size 64
 if ($LASTEXITCODE -ne 0) { throw 'Smoke failed; full training stopped' }
 & $python -u check_graph_repair.py --output "$base/full"
 if ($LASTEXITCODE -ne 0) { throw 'Full preflight failed' }
 & $python -m pip freeze | Out-File "$base/full/requirements.lock.txt" -Encoding utf8
 & $python -u run_graph_repair.py --output "$base/full" --epochs 16 --subjects 2,5 --seeds 13 --batch-size 64
 if ($LASTEXITCODE -ne 0) { throw 'Full pilot failed; preserve outputs' }
 Write-Host 'Graph repair pilot completed; validation predictions and checkpoints audited.' -ForegroundColor Green
} catch { $_ | Out-String | Set-Content "$base/terminal_error.txt"; Write-Host $_ -ForegroundColor Red }
finally { Stop-Transcript }
