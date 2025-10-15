# PowerShell script to install GPU dependencies for murmur-inference-server
# Run this script in PowerShell with: .\install-gpu-dependencies.ps1

# Python - 3.12.7
# CUDA - 11.8
# CUDnn - 9.8.0
# Visual Studio build tools - 2019


Write-Host "=== Installing GPU Dependencies for Murmur Inference Server ===" -ForegroundColor Green
Write-Host ""

# Check if Python is available
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found. Please install Python 3.10 or higher." -ForegroundColor Red
    exit 1
}

# Check if CUDA is available
Write-Host "`nChecking CUDA installation..." -ForegroundColor Yellow
try {
    $nvccVersion = nvcc --version 2>&1
    if ($nvccVersion -match "release (\d+\.\d+)") {
        $cudaVersion = $matches[1]
        Write-Host "Found CUDA $cudaVersion" -ForegroundColor Green
    } else {
        Write-Host "WARNING: CUDA toolkit not found. GPU acceleration may not work." -ForegroundColor Yellow
        Write-Host "Please install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
    }
} catch {
    Write-Host "WARNING: nvcc not found. CUDA may not be installed." -ForegroundColor Yellow
}

# Check if virtual environment exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "`nVirtual environment found. Activating..." -ForegroundColor Yellow
    & ".\venv\Scripts\Activate.ps1"
} else {
    Write-Host "`nNo virtual environment found. Creating new one..." -ForegroundColor Yellow
    python -m venv venv
    & ".\venv\Scripts\Activate.ps1"
}

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install grpcio-tools
Write-Host "`nInstalling grpcio-tools..." -ForegroundColor Yellow
pip install grpcio-tools

# Install llama-cpp-python with CUDA support
Write-Host "`nInstalling llama-cpp-python with CUDA support..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Yellow
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
$env:CMAKE_ARGS = "-DGGML_CUDA=on -DGGML_NATIVE=off -DCMAKE_CUDA_ARCHITECTURES=61"
$env:FORCE_CMAKE = "1"
pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
# pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install llama-cpp-python with CUDA support." -ForegroundColor Red
    Write-Host "Make sure you have:" -ForegroundColor Yellow
    Write-Host "  1. Visual Studio with C++ build tools installed" -ForegroundColor Yellow
    Write-Host "  2. CUDA Toolkit installed" -ForegroundColor Yellow
    Write-Host "  3. CMake installed (or will be installed via pip)" -ForegroundColor Yellow
    exit 1
}

# Install faster-whisper and ctranslate2 (GPU-enabled)
Write-Host "`nInstalling faster-whisper with GPU support..." -ForegroundColor Yellow
pip install ctranslate2 --upgrade
pip install faster-whisper --upgrade

# Install remaining requirements
Write-Host "`nInstalling remaining dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Verify GPU support
Write-Host "`n=== Verifying GPU Support ===" -ForegroundColor Green

Write-Host "`nTesting llama-cpp-python CUDA support..." -ForegroundColor Yellow
python -c @"
try:
    from llama_cpp import Llama
    print('✓ llama-cpp-python imported successfully')
    # Try to check if CUDA was compiled
    import ctypes.util
    try:
        # This is a basic check - actual CUDA support is verified when loading a model
        print('✓ llama-cpp-python CUDA support: Check when loading model')
    except Exception as e:
        print(f'⚠ Could not verify CUDA: {e}')
except ImportError as e:
    print(f'✗ Failed to import llama-cpp-python: {e}')
"@

Write-Host "`nTesting faster-whisper GPU support..." -ForegroundColor Yellow
python -c @"
try:
    import ctranslate2
    print('✓ ctranslate2 imported successfully')
    if ctranslate2.get_cuda_device_count() > 0:
        print(f'✓ CUDA devices detected: {ctranslate2.get_cuda_device_count()}')
    else:
        print('⚠ No CUDA devices detected - will use CPU')
except ImportError as e:
    print(f'✗ Failed to import ctranslate2: {e}')
"@

Write-Host "`n=== Installation Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Copy env-template.txt to .env and configure GPU settings" -ForegroundColor White
Write-Host "2. Set WHISPER_DEVICE=cuda and WHISPER_COMPUTE_TYPE=float16" -ForegroundColor White
Write-Host "3. Set LLM_N_GPU_LAYERS=-1 to offload all layers to GPU" -ForegroundColor White
Write-Host "4. Run: python main.py" -ForegroundColor White
Write-Host ""
Write-Host "Monitor GPU usage with: nvidia-smi -l 1" -ForegroundColor Cyan








