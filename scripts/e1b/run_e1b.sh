#!/usr/bin/env bash
# E1b controlled experiment: clean code with E1b & E1 enabled
set -euo pipefail
B=/home/<user>/bxjudge
E0=$B/runs/e0-baseline-20261124
E1B=$B/runs/e1b-eval-20261125
mkdir -p "$E1B"

SEL="$B/weights/selected.json"
TRT="$B/weights/rfdetr-trt/manifest.json"
OUT="$E1B/production"
rm -rf "$OUT"
echo "$(date +%T) start E1b production"
/usr/bin/time -f "%e s wall" env PYTHONPATH="$B/src" "$B/.venv/bin/python" -m bxjudge.cli run "$E0/videos" --out "$OUT" \
  --selection "$SEL" --trt-manifest "$TRT" \
  --dynamic-calibration "$E1B/dynamic_calibration.yaml" --two-bey-tracking "$B/config/two_bey_tracking.yaml" \
  --identity "$E1B/identity.yaml" \
  --stadium-keypoints --no-vlm > "$E1B/production.log" 2>&1
tail -2 "$E1B/production.log"
"$B/.venv/bin/python" "$B/scripts/e0/e0_eval.py" "$E0/truth/e0-truth.json" "$OUT" --json "$E1B/metrics-production.json"
echo "$(date +%T) E1b production done"
