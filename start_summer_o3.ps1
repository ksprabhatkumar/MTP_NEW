$workspace = 'C:\Users\General\Desktop\prabhat\MTP'
$plan = Get-Content (Join-Path $workspace 'summer_o3_session.json') -Raw | ConvertFrom-Json
if (-not $plan.ready -or $plan.unresolved.Count -gt 0) { throw 'O3 selection/config is still blocked. Check summer_o3_session.json.' }
$study = Get-Content (Join-Path $workspace 'summer_o3_study.json') -Raw | ConvertFrom-Json
$o2Status = Get-Content (Join-Path $study.summer_o2_run 'status.json') -Raw | ConvertFrom-Json
if ($o2Status.state -ne 'complete') { throw 'Summer O2 must complete before O3 starts.' }
$runDirectory = Join-Path $workspace ('results\summer_o3_session_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $runDirectory -ErrorAction Stop | Out-Null
$env:PYTHONPATH = Join-Path $workspace 'prabhat\Lib\site-packages'
$launcherPython = Join-Path $workspace 'prabhat\Scripts\python.exe'
$trainingPython = Join-Path $workspace '.python\cpython-3.11.15-windows-x86_64-none\python.exe'
$log = Join-Path $runDirectory 'independent_stdout.log'
Set-Location -LiteralPath $workspace
$launch = & $launcherPython (Join-Path $workspace 'launch_independent.py') $log $trainingPython -u (Join-Path $workspace 'run_long_training.py') --plan (Join-Path $workspace 'summer_o3_session.json') --output $runDirectory
if ($LASTEXITCODE -ne 0) { throw 'Summer O3 independent launch failed.' }
$launch
$pointer = @{ run_directory = $runDirectory; launched = (Get-Date).ToString('o'); launch = $launch; plan = 'summer_o3_session.json' }
$pointer | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $workspace 'results\summer_o3_active_run.json') -Encoding UTF8
