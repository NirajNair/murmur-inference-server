#!/bin/bash
# Bash script to install GPU dependencies for murmur-inference-server
# Run this script with: bash install-gpu-dependencies.sh

# Python - 3.12.7
# CUDA - 11.8
# CUDnn - 9.8.0
# Visual Studio build tools - 2019

set -e

echo "=== Installing GPU Dependencies for Murmur Inference Server ==="
echo ""

# Check if Python is available
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "Found: $PYTHON_VERSION"
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version)
    echo "Found: $PYTHON_VERSION"
    PYTHON=python
else
    echo "ERROR: Python not found. Please install Python 3.10 or higher."
    exit 1
fi

# Check if CUDA is available
echo ""
echo "Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -c2-)
    echo "Found CUDA $CUDA_VERSION"
else
    echo "WARNING: nvcc not found. CUDA may not be installed."
    echo "Please install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads"
fi

# Check if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. NVIDIA drivers may not be installed."
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo ""
    echo "Virtual environment found. Activating..."
    source venv/bin/activate
else
    echo ""
    echo "No virtual environment found. Creating new one..."
    $PYTHON -m venv venv
    source venv/bin/activate
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install grpcio-tools
echo ""
echo "Installing grpcio-tools..."
pip install grpcio-tools

# Install llama-cpp-python with CUDA support
echo ""
echo "Installing llama-cpp-python with CUDA support..."
echo "This may take several minutes..."
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install llama-cpp-python with CUDA support."
    echo "Make sure you have:"
    echo "  1. GCC/G++ compiler installed"
    echo "  2. CUDA Toolkit installed"
    echo "  3. CMake installed"
    exit 1
fi

# Install faster-whisper and ctranslate2 (GPU-enabled)
echo ""
echo "Installing faster-whisper with GPU support..."
pip install ctranslate2 --upgrade
pip install faster-whisper --upgrade

# Install remaining requirements
echo ""
echo "Installing remaining dependencies..."
pip install -r requirements.txt

# Verify GPU support
echo ""
echo "=== Verifying GPU Support ==="

echo ""
echo "Testing llama-cpp-python CUDA support..."
$PYTHON -c "
try:
    from llama_cpp import Llama
    print('✓ llama-cpp-python imported successfully')
    print('✓ llama-cpp-python CUDA support: Check when loading model')
except ImportError as e:
    print(f'✗ Failed to import llama-cpp-python: {e}')
"

echo ""
echo "Testing faster-whisper GPU support..."
$PYTHON -c "
try:
    import ctranslate2
    print('✓ ctranslate2 imported successfully')
    if ctranslate2.get_cuda_device_count() > 0:
        print(f'✓ CUDA devices detected: {ctranslate2.get_cuda_device_count()}')
    else:
        print('⚠ No CUDA devices detected - will use CPU')
except ImportError as e:
    print(f'✗ Failed to import ctranslate2: {e}')
"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Next steps:"
echo "1. Copy env-template.txt to .env and configure GPU settings"
echo "2. Set WHISPER_DEVICE=cuda and WHISPER_COMPUTE_TYPE=float16"
echo "3. Set LLM_N_GPU_LAYERS=-1 to offload all layers to GPU"
echo "4. Run: python main.py"
echo ""
echo "Monitor GPU usage with: nvidia-smi -l 1"








