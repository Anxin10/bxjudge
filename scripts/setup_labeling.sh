#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked --project scripts/sam-runtime
uv tool install label-studio==1.23.2
# Authentication is a human step: hf auth login (from the SAM runtime) if needed.