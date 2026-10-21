#!/usr/bin/env bash
# Sync one batch of match videos from Google Drive into the WSL filesystem (read-only on Drive).
# Usage: scripts/sync_videos.sh <drive_folder_id> <batch_name>
# Example: scripts/sync_videos.sh 1ExampleDriveFolderId0123456789ab 20261001
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <drive_folder_id> <batch_name>" >&2
    exit 2
fi

if ! command -v rclone >/dev/null 2>&1; then
    echo "error: rclone is not installed. Install it (e.g. 'sudo apt-get install rclone')," >&2
    echo "then run 'rclone config' once to create the remote named 'gdrive' (see README.md)." >&2
    exit 1
fi

rclone copy gdrive: ~/bx/videos/"$2" --drive-root-folder-id "$1" --include "*.mp4" --progress
