#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked --project scripts/locateanything-runtime
PYTHONPATH="$PWD/src" scripts/locateanything-runtime/.venv/bin/python scripts/install_locateanything_weights.py "$@"
uv tool install label-studio==1.23.2
