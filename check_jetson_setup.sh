#!/bin/bash
# Script kiểm tra cấu hình cho Jetson Xavier

echo "============================================================"
echo "Jetson Xavier Setup Check"
echo "============================================================"
echo ""

# Kiểm tra architecture
ARCH=$(uname -m)
echo "Architecture: $ARCH"
if [[ "$ARCH" == "aarch64" ]]; then
    echo "✓ ARM64 detected (Jetson)"
else
    echo "⚠ Not ARM64 - this script is optimized for Jetson"
fi
echo ""

# Kiểm tra libgomp
echo "Checking for libgomp..."
if [ -f "/usr/lib/aarch64-linux-gnu/libgomp.so.1" ]; then
    echo "✓ Found: /usr/lib/aarch64-linux-gnu/libgomp.so.1"
    ls -lh /usr/lib/aarch64-linux-gnu/libgomp.so.1
elif [ -f "/usr/lib/x86_64-linux-gnu/libgomp.so.1" ]; then
    echo "✓ Found: /usr/lib/x86_64-linux-gnu/libgomp.so.1"
    ls -lh /usr/lib/x86_64-linux-gnu/libgomp.so.1
elif [ -f "/usr/lib/libgomp.so.1" ]; then
    echo "✓ Found: /usr/lib/libgomp.so.1"
    ls -lh /usr/lib/libgomp.so.1
else
    echo "✗ libgomp.so.1 not found in standard locations"
    echo "  Searching system..."
    find /usr -name "libgomp.so.1" 2>/dev/null | head -5
fi
echo ""

# Kiểm tra Python và venv
echo "Checking Python environment..."
if [ -d "venv" ]; then
    echo "✓ venv directory found"
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        echo "✓ venv activated"
        echo "Python: $(which python)"
        echo "Python version: $(python --version)"
        
        # Kiểm tra scikit-learn
        if python -c "import sklearn" 2>/dev/null; then
            echo "✓ scikit-learn installed"
            python -c "import sklearn; print(f'  Version: {sklearn.__version__}')"
        else
            echo "✗ scikit-learn not installed or import failed"
        fi
    fi
else
    echo "⚠ venv directory not found"
fi
echo ""

# Kiểm tra build tools
echo "Checking build tools..."
if command -v gcc &> /dev/null; then
    echo "✓ gcc found: $(gcc --version | head -1)"
else
    echo "✗ gcc not found (needed for building scikit-learn from source)"
fi

if command -v python3-config &> /dev/null; then
    echo "✓ python3-config found"
else
    echo "⚠ python3-config not found (may need python3-dev)"
fi
echo ""

echo "============================================================"
echo "Recommendations:"
echo "============================================================"
if [[ "$ARCH" == "aarch64" ]]; then
    echo "1. If scikit-learn import fails, run: ./fix_sklearn_tls.sh"
    echo "2. Then run: ./start.sh"
    echo ""
    echo "The start.sh script will automatically:"
    echo "  - Set LD_PRELOAD to use system libgomp"
    echo "  - Set OMP_NUM_THREADS=1 to avoid conflicts"
    echo "  - Configure environment for Jetson"
else
    echo "This system is not ARM64. Standard fixes should work."
fi
echo "============================================================"

