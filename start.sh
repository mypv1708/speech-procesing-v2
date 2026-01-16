#!/bin/bash
# Script để khởi động hệ thống với cấu hình robot server

# Cấu hình Robot Server
export ROBOT_SERVER_HOST=192.168.0.111
export ROBOT_SERVER_PORT=12345

# Fix OpenMP/libgomp TLS conflict (cannot allocate memory in static TLS block)
# Đặc biệt quan trọng trên Jetson Xavier (ARM64)
# Ưu tiên tìm libgomp cho ARM64 trước (Jetson)
if [ -f "/usr/lib/aarch64-linux-gnu/libgomp.so.1" ]; then
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
elif [ -f "/usr/lib/x86_64-linux-gnu/libgomp.so.1" ]; then
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1
elif [ -f "/usr/lib/libgomp.so.1" ]; then
    export LD_PRELOAD=/usr/lib/libgomp.so.1
fi

# Giảm số thread OpenMP để tránh xung đột (quan trọng trên Jetson)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Jetson-specific optimizations
if [ -n "$(uname -m | grep aarch64)" ]; then
    echo "Detected ARM64 architecture (Jetson)"
    # Tăng kích thước TLS nếu có thể
    export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
fi

echo "============================================================"
echo "Robot Server Configuration:"
echo "  Host: $ROBOT_SERVER_HOST"
echo "  Port: $ROBOT_SERVER_PORT"
echo "============================================================"
if [ -n "$LD_PRELOAD" ]; then
    echo "OpenMP Fix: Using $LD_PRELOAD"
fi
echo "============================================================"
echo ""

# Kích hoạt virtual environment nếu có
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Chạy main application
python main.py

