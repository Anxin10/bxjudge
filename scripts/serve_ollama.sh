#!/usr/bin/env bash
# Foreground only; never manage or stop an existing/shared Ollama service.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
runtime="${1:-$root/weights/ollama-runtime}"
models="${2:-$root/weights/ollama}"
if [[ ! -x "$runtime/bin/ollama" ]]; then
    echo "error: no runtime at $runtime; run scripts/setup_ollama.py first" >&2
    exit 2
fi
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_NO_CLOUD=1
export OLLAMA_MODELS="$models"
exec "$runtime/bin/ollama" serve
