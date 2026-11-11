param(
    [string]$PythonVersion = '3.13'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$taskTemp = Join-Path $projectRoot '.cache\tmp'
New-Item -ItemType Directory -Path $taskTemp -Force | Out-Null
$env:TEMP = $taskTemp
$env:TMP = $taskTemp
$env:PIP_CACHE_DIR = Join-Path $projectRoot '.cache\pip'
$env:PYTHONIOENCODING = 'utf-8'

if (-not (Test-Path -LiteralPath $venvPython)) {
    & py "-$PythonVersion" -m venv --without-pip (Join-Path $projectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create Windows venv' }
}

& $venvPython -m ensurepip --upgrade --default-pip
if ($LASTEXITCODE -ne 0) { throw 'Failed to initialize pip in Windows venv' }
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Failed to update pip' }
& $venvPython -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'Failed to install PyTorch CUDA 12.8' }
& $venvPython -m pip install -r (Join-Path $projectRoot 'requirements-live.txt')
if ($LASTEXITCODE -ne 0) { throw 'Failed to install live demo dependencies' }

& $venvPython -c "import torch,cv2,ultralytics,serial; print('torch',torch.__version__,'CUDA',torch.version.cuda,'available',torch.cuda.is_available()); print('opencv',cv2.__version__,'ultralytics',ultralytics.__version__,'pyserial',serial.VERSION); assert torch.cuda.is_available(), 'GPU unavailable'"
if ($LASTEXITCODE -ne 0) { throw 'Dependencies installed, but GPU verification failed' }
