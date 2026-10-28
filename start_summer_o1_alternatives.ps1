$workspace = 'C:\Users\General\Desktop\prabhat\MTP'
$planPath = Join-Path $workspace 'summer_o1_alternatives_session.json'
$plan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
if (-not $plan.ready -or $plan.unresolved.Count -gt 0) {
    throw 'O1 alternative source pilot has not cleared the full-run gate.'
}
$pilot = Join-Path $workspace 'results\summer_o1_alternatives_pilot_20261021'
if (-not (Test-Path -LiteralPath (Join-Path $pilot 'completed.json')) -or
    -not (Test-Path -LiteralPath (Join-Path $pilot 'ANALYSIS.md'))) {
    throw 'O1 alternative pilot results or review are missing.'
}
$gatePilot = Join-Path $workspace 'results\summer_o1_gate_pilot_20261021'
if (-not (Test-Path -LiteralPath (Join-Path $gatePilot 'completed.json')) -or
    -not (Test-Path -LiteralPath (Join-Path $gatePilot 'ANALYSIS.md'))) {
    throw 'O1 gated pilot results or review are missing.'
}
foreach ($pointerName in @('summer_o1_alternatives_active_run.json',
                            'summer_o2_active_run.json', 'summer_o3_active_run.json')) {
    $pointerPath = Join-Path $workspace (Join-Path 'results' $pointerName)
    if (-not (Test-Path -LiteralPath $pointerPath)) { continue }
    $pointer = Get-Content -LiteralPath $pointerPath -Raw | ConvertFrom-Json
    $statusPath = Join-Path $pointer.run_directory 'status.json'
    if (-not (Test-Path -LiteralPath $statusPath)) { continue }
    $status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    if ($status.state -eq 'running' -and (Get-Process -Id $status.pid -ErrorAction SilentlyContinue)) {
        throw "An existing GPU session is active: $($pointer.run_directory)"
    }
}
$runDirectory = Join-Path $workspace ('results\summer_o1_alternatives_session_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $runDirectory -ErrorAction Stop | Out-Null
$env:PYTHONPATH = Join-Path $workspace 'prabhat\Lib\site-packages'
$launcherPython = Join-Path $workspace 'prabhat\Scripts\python.exe'
$trainingPython = Join-Path $workspace '.python\cpython-3.11.15-windows-x86_64-none\python.exe'
$log = Join-Path $runDirectory 'independent_stdout.log'
Set-Location -LiteralPath $workspace
$launch = & $launcherPython (Join-Path $workspace 'launch_independent.py') $log $trainingPython -u (Join-Path $workspace 'run_long_training.py') --plan $planPath --output $runDirectory
if ($LASTEXITCODE -ne 0) { throw 'Independent O1 alternative launch failed.' }
$pointer = @{ run_directory = $runDirectory; launched = (Get-Date).ToString('o'); launch = $launch; plan = 'summer_o1_alternatives_session.json' }
$pointer | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $workspace 'results\summer_o1_alternatives_active_run.json') -Encoding UTF8
$launch
