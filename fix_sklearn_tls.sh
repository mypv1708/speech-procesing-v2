#!/bin/bash
# Script để fix lỗi "cannot allocate memory in static TLS block" với scikit-learn
# Đặc biệt cho Jetson Xavier (ARM64)

echo "============================================================"
echo "Fixing scikit-learn TLS issue..."
echo "Detected architecture: $(uname -m)"
echo "============================================================"
echo ""

# Kích hoạt virtual environment nếu có
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠ Warning: venv directory not found"
fi

# Kiểm tra dependencies cần thiết cho build từ source
echo ""
echo "Checking build dependencies..."
if ! command -v gcc &> /dev/null; then
    echo "⚠ Warning: gcc not found. Installing build-essential..."
    sudo apt-get update && sudo apt-get install -y build-essential python3-dev
fi

# Set environment variables cho build
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo ""
echo "Step 1: Uninstalling scikit-learn..."
pip uninstall -y scikit-learn

echo ""
echo "Step 2: Reinstalling scikit-learn from source (no binary)..."
echo "⚠ This may take 10-30 minutes on Jetson Xavier..."
echo "⚠ Building from source ensures compatibility with your system's libgomp"
echo ""

# Build từ source với các flags phù hợp cho ARM64
pip install --no-binary=scikit-learn scikit-learn

echo ""
echo "============================================================"
echo "Done! Please try running the application again with:"
echo "  ./start.sh"
echo "============================================================"

