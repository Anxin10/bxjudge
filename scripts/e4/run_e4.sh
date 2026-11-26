#!/usr/bin/env bash
# E4 controlled experiment: multi-feature finish fusion enabled with E1 & E1b
set -euo pipefail
B="$(cd "$(dirname "$0")/../.." && pwd)"
E0=$B/runs/e0-baseline-20261124
E1B=$B/runs/e1b-eval-20261125
E4=$B/runs/e4-eval-20261125
mkdir -p "$E4"

# Copy base configs and set fusion
cp "$E1B/dynamic_calibration.yaml" "$E4/dynamic_calibration.yaml"
cp "$E1B/identity.yaml" "$E4/identity.yaml"
cp "$B/config/finish_fusion_enabled.yaml" "$E4/finish_fusion.yaml"

SEL="$B/weights/selected.json"
TRT="$B/weights/rfdetr-trt/manifest.json"
OUT="$E4/production"
rm -rf "$OUT"
echo "$(date +%T) start E4 production"
/usr/bin/time -f "%e s wall" env PYTHONPATH="$B/src" "$B/.venv/bin/python" -m bxjudge.cli run "$E0/videos" --out "$OUT" \
  --selection "$SEL" --trt-manifest "$TRT" \
  --dynamic-calibration "$E4/dynamic_calibration.yaml" --two-bey-tracking "$B/config/two_bey_tracking.yaml" \
  --identity "$E4/identity.yaml" \
  --finish-fusion "$E4/finish_fusion.yaml" \
  --stadium-keypoints --no-vlm > "$E4/production.log" 2>&1
tail -2 "$E4/production.log"
"$B/.venv/bin/python" "$B/scripts/e0/e0_eval.py" "$E0/truth/e0-truth.json" "$OUT" --json "$E4/metrics-production.json"
echo "$(date +%T) E4 production done"
