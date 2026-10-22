#!/usr/bin/env bash
# Set up the bxjudge environment inside WSL2 Ubuntu.
# Run from anywhere: bash scripts/setup_wsl.sh
set -euo pipefail

cd "$(dirname "$0")/.."

# 1. System packages (needs sudo)
sudo apt-get update
sudo apt-get install -y ffmpeg rclone

# 2. uv (installed into ~/.local/bin, no sudo). User-installed tools (uv, the
#    standalone rclone) live in ~/.local/bin, which a normal login PATH may lack.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# 3. Python 3.12 managed by uv
uv python install 3.12

# 4. Project environment. torch/torchvision resolve from the PyTorch CUDA 13.2
#    wheel index (cu132) via [tool.uv.sources] in pyproject.toml.
uv sync

echo "Setup done. Verify with: uv run pytest tests/test_env.py -v"
