param([string]$OutputDirectory = 'results/local_message_pilot_20261027')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$Host.UI.RawUI.WindowTitle = 'Prabhat - GPU setup and O1 message-passing pilot'
$outputPath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $OutputDirectory))
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
Start-Transcript -Path (Join-Path $outputPath 'terminal_transcript.log') -Append
try {
    $basePython = Join-Path (Split-Path $PSScriptRoot) 'GAE_Mamba_WHAR\.python\cpython-3.11.15-windows-x86_64-none\python.exe'
    $projectPython = Join-Path $PSScriptRoot 'Prabhat\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $projectPython)) {
        Write-Host 'Creating the isolated Prabhat environment...' -ForegroundColor Cyan
        & $basePython -m venv (Join-Path $PSScriptRoot 'Prabhat')
        if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
    }
    Write-Host 'Installing CUDA PyTorch 2.5.1 and project dependencies...' -ForegroundColor Cyan
    & $projectPython -m pip install --disable-pip-version-check torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
    & $projectPython -m pip install --disable-pip-version-check -r requirements_local_gpu.txt
    if ($LASTEXITCODE -ne 0) { throw 'Project dependency installation failed.' }
    & $projectPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed.' }
    & $projectPython -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; x=torch.randn(128,128,device='cuda'); y=x@x; torch.cuda.synchronize(); print('GPU:',torch.cuda.get_device_name()); print('Torch:',torch.__version__,'CUDA:',torch.version.cuda,'BF16:',torch.cuda.is_bf16_supported()); assert torch.isfinite(y).all()"
    if ($LASTEXITCODE -ne 0) { throw 'GPU computation check failed.' }
    & $projectPython -m pip freeze | Out-File -LiteralPath (Join-Path $outputPath 'requirements.lock.txt') -Encoding utf8
    'ready' | Set-Content -LiteralPath (Join-Path $outputPath 'environment_ready.txt')
    Write-Host 'Environment ready. Waiting for the reviewed pilot implementation...' -ForegroundColor Cyan
    $releasePath = Join-Path $outputPath 'implementation_ready.txt'
    $deadline = (Get-Date).AddHours(2)
    while (-not (Test-Path -LiteralPath $releasePath)) {
        if ((Get-Date) -gt $deadline) { throw 'Implementation readiness timeout. No training started.' }
        Start-Sleep -Seconds 3
    }
    Write-Host 'Checking message passing, GPU execution and checkpoint recovery...' -ForegroundColor Cyan
    & $projectPython -u check_local_message_training.py --output $outputPath
    if ($LASTEXITCODE -ne 0) { throw 'Preflight failed. Training was not launched.' }
    Write-Host 'Starting six sequential source-validation pilot jobs in THIS terminal.' -ForegroundColor Green
    & $projectPython -u run_local_message_pilot.py --output $outputPath --epochs 8 --batch-size 64
    if ($LASTEXITCODE -ne 0) { throw 'Training stopped. Inspect status.json; checkpoints are retained.' }
    Write-Host 'Pilot completed. See summary.json and PILOT_REVIEW.md.' -ForegroundColor Green
} catch {
    $_ | Out-String | Set-Content -LiteralPath (Join-Path $outputPath 'terminal_error.txt')
    Write-Host $_ -ForegroundColor Red
} finally {
    Stop-Transcript
}
