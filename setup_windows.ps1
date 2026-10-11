$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (Test-Path -LiteralPath '.venv\Scripts\python.exe') {
    Write-Host 'Existing environment preserved. Run doctor and pytest with the existing Python.'
    exit 0
}
py -3.11 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.11 from python.org first.' }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements-core.txt
if ($LASTEXITCODE -ne 0) { throw 'Core dependency installation failed.' }
Write-Host 'Core environment ready. Install matching torch/torchvision using https://pytorch.org/get-started/locally/ then run:'
Write-Host '.\.venv\Scripts\python.exe -m pip install -r requirements.txt'
Write-Host '.\.venv\Scripts\python.exe -m selight doctor'
Write-Host '.\.venv\Scripts\python.exe -m pytest -q'
