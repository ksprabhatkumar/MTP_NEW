$workspace = 'C:\Users\General\Desktop\prabhat\MTP'
$o1 = Join-Path $workspace 'results\summer_o1_session_20261017_232945'
$status = Get-Content (Join-Path $o1 'status.json') -Raw | ConvertFrom-Json
if ($status.state -ne 'complete') { throw 'Summer O1 must complete before O2 starts.' }
$runDirectory = Join-Path $workspace ('results\summer_o2_session_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $runDirectory -ErrorAction Stop | Out-Null
$env:PYTHONPATH = Join-Path $workspace 'prabhat\Lib\site-packages'
$launcherPython = Join-Path $workspace 'prabhat\Scripts\python.exe'
$trainingPython = Join-Path $workspace '.python\cpython-3.11.15-windows-x86_64-none\python.exe'
$log = Join-Path $runDirectory 'independent_stdout.log'
Set-Location -LiteralPath $workspace
$launch = & $launcherPython (Join-Path $workspace 'launch_independent.py') $log $trainingPython -u (Join-Path $workspace 'run_long_training.py') --plan (Join-Path $workspace 'summer_o2_session.json') --output $runDirectory
if ($LASTEXITCODE -ne 0) { throw 'Summer O2 independent launch failed.' }
$launch
$pointer = @{ run_directory = $runDirectory; launched = (Get-Date).ToString('o'); launch = $launch; plan = 'summer_o2_session.json' }
$pointer | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $workspace 'results\summer_o2_active_run.json') -Encoding UTF8
