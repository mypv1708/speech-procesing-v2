#!/bin/bash
# Script để fix lỗi "cannot allocate memory in static TLS block" với scikit-learn

echo "============================================================"
echo "Fixing scikit-learn TLS issue..."
echo "============================================================"
echo ""

# Kích hoạt virtual environment nếu có
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠ Warning: venv directory not found"
fi

echo ""
echo "Step 1: Uninstalling scikit-learn..."
pip uninstall -y scikit-learn

echo ""
echo "Step 2: Reinstalling scikit-learn from source (no binary)..."
echo "This may take several minutes..."
pip install --no-binary=scikit-learn scikit-learn

echo ""
echo "============================================================"
echo "Done! Please try running the application again."
echo "============================================================"

