#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo " LiteLLM Proxy - Install"
echo "============================================"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed."
    echo
    echo "Install Python from: https://www.python.org/downloads/"
    exit 1
fi

echo "Found: $(python3 --version)"
echo

# Install litellm
echo "Installing safe version of litellm (v1.82.3)..."
pip3 install litellm[proxy]==1.82.3
echo

# Check .env in src directory
if [ ! -f "$SCRIPT_DIR/src/.env" ]; then
    echo "[WARNING] .env file not found. Creating template..."
    cat > "$SCRIPT_DIR/src/.env" << 'EOF'
NVIDIA_API_BASE=https://your-api-base-url/
NVIDIA_API_KEY=your-api-key-here
EOF
    echo
    echo "Created src/.env file. Edit it with your API credentials before starting."
fi

echo
echo "============================================"
echo " Done! Usage:"
echo "   ./start.sh"
echo "   ./stop.sh"
echo "   ./status.sh"
echo "============================================"