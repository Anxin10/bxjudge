#!/usr/bin/env bash
# Sync one batch of match videos from Google Drive into the WSL filesystem (read-only on Drive).
# Usage: scripts/sync_videos.sh <drive_folder_id> <batch_name>
# Example: scripts/sync_videos.sh 1ExampleDriveFolderId0123456789ab 20261001
set -euo pipefail

# The standalone rclone lives in ~/.local/bin, which a normal login PATH may lack.
export PATH="$HOME/.local/bin:$PATH"

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <drive_folder_id> <batch_name>" >&2
    exit 2
fi

folder_id="$1"
batch="$2"
# Drive folder IDs are URL-safe base64-like tokens; never let one become an rclone flag.
if [[ ! "$folder_id" =~ ^[A-Za-z0-9_][A-Za-z0-9_-]*$ ]]; then
    echo "error: invalid drive folder id: '$folder_id'" >&2
    exit 2
fi
# The batch must be ONE plain path segment so videos stay inside ~/bx/videos/<batch>.
if [[ ! "$batch" =~ ^[A-Za-z0-9_][A-Za-z0-9._-]*$ ]]; then
    echo "error: invalid batch name '$batch': use one folder name such as 20261001" >&2
    echo "(letters, digits, '.', '_', '-'; no '/', '..' or leading '.'/'-')" >&2
    exit 2
fi

if ! command -v rclone >/dev/null 2>&1; then
    echo "error: rclone is not installed (expected on PATH or in ~/.local/bin)." >&2
    echo "Install it, then run 'rclone config' once to create the remote named 'gdrive' (see README.md)." >&2
    exit 1
fi

# copy (never sync/move/delete): Drive stays untouched, existing local files are kept.
rclone copy gdrive: "$HOME/bx/videos/$batch" --drive-root-folder-id "$folder_id" \
    --include "*.mp4" --progress
