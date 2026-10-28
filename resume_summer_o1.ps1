$workspace = 'C:\Users\General\Desktop\prabhat\MTP'
$runDirectory = Join-Path $workspace 'results\summer_o1_session_20261017_232945'
$launcherPython = Join-Path $workspace 'prabhat\Scripts\python.exe'
$trainingPython = Join-Path $workspace '.python\cpython-3.11.15-windows-x86_64-none\python.exe'
$runner = Join-Path $workspace 'run_long_training.py'
$plan = Join-Path $workspace 'summer_o1_session.json'
$log = Join-Path $runDirectory 'base_independent_stdout.log'

Set-Location -LiteralPath $workspace
$env:PYTHONPATH = Join-Path $workspace 'prabhat\Lib\site-packages'
& $launcherPython (Join-Path $workspace 'launch_independent.py') $log $trainingPython -u $runner --plan $plan --output $runDirectory
exit $LASTEXITCODE
