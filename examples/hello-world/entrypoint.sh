#!/bin/sh
# Hello World application entrypoint for NeoRuntime Platform
# Supports both running app.py and debugging mode

if [ "$1" = "debug" ] || [ "$1" = "/bin/bash" ] || [ "$1" = "/bin/sh" ]; then
    echo "========================================"
    echo "  Debug Mode"
    echo "  Platform: $(uname -m)"
    echo "  Time: $(date)"
    echo "========================================"
    echo ""
    echo "Available commands:"
    echo "  python3 /app/app.py    - Run the application"
    echo "  ps aux                 - List processes"
    echo "  netstat -tlnp          - List network ports"
    echo ""
    exec /bin/bash
else
    # Run the Python application
    exec python3 /app/app.py
fi
