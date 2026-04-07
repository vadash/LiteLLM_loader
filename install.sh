#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo " LiteLLM Proxy - Install (uv)"
echo "============================================"
echo

# Check uv
if ! command -v uv &> /dev/null; then
    echo "[ERROR] uv is not installed."
    echo
    echo "Install uv with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo
    echo "Or visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "Found: $(uv --version)"
echo

# Sync dependencies
echo "Installing dependencies..."
uv sync
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
