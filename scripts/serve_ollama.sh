#!/usr/bin/env bash
# Run the project-owned Ollama in the foreground (Ctrl+C stops it).
# Loopback only, no cloud, models under weights/ollama. Never manages or stops any other
# Ollama service (e.g. one on 11434).
# Usage: scripts/serve_ollama.sh [runtime_dir] [models_dir]
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
runtime="${1:-$root/weights/ollama-runtime}"
models="${2:-$root/weights/ollama}"
if [[ ! -x "$runtime/bin/ollama" ]]; then
    echo "error: no runtime at $runtime; run scripts/setup_ollama.py first" >&2
    exit 2
fi
mkdir -p "$models"
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_NO_CLOUD=1
export OLLAMA_MODELS="$models"
exec "$runtime/bin/ollama" serve
