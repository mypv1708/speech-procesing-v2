#!/bin/bash
# Script để khởi động hệ thống với cấu hình robot server

# Cấu hình Robot Server
export ROBOT_SERVER_HOST=192.168.0.111
export ROBOT_SERVER_PORT=12345

# Fix OpenMP/libgomp TLS conflict (cannot allocate memory in static TLS block)
# Thử sử dụng libgomp từ hệ thống thay vì từ scikit-learn
if [ -f "/usr/lib/x86_64-linux-gnu/libgomp.so.1" ]; then
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1
elif [ -f "/usr/lib/aarch64-linux-gnu/libgomp.so.1" ]; then
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
elif [ -f "/usr/lib/libgomp.so.1" ]; then
    export LD_PRELOAD=/usr/lib/libgomp.so.1
fi

# Giảm số thread OpenMP để tránh xung đột
export OMP_NUM_THREADS=1

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

