#!/usr/bin/env bash
set -euo pipefail

apt update
apt install -y tmux python3-dev vim curl

echo 'set number' > ~/.vimrc

curl -LsSf https://astral.sh/uv/install.sh | sh

export UV_CACHE_DIR="/workspace/.cache/uv"
if ! grep -q 'UV_CACHE_DIR="/workspace/.cache/uv"' ~/.bashrc 2>/dev/null; then
    echo 'export UV_CACHE_DIR="/workspace/.cache/uv"' >> ~/.bashrc
fi
mkdir -p "$UV_CACHE_DIR"

curl -fsSL https://claude.ai/install.sh | bash
