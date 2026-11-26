#!/usr/bin/env bash
# E0 baseline: clean HEAD (95f4d9a) export, production vs v7 candidate, same flags as the v7 phase4/5 runs.
set -euo pipefail
B="$(cd "$(dirname "$0")/../.." && pwd)"
E=$B/runs/e0-baseline-20261124
mkdir -p $E
X=$E/code-95f4d9a
if [ -L $X/weights ]; then rm $X/weights $X/calib; rm -f $X/runs/pretrained; cp -al $B/weights $X/weights; cp -al $B/calib $X/calib; mkdir -p $X/runs; cp -al $B/runs/pretrained $X/runs/pretrained; fi
if [ ! -d $X ]; then
  mkdir -p $X
  git -C $B archive 95f4d9a | tar -x -C $X
  cp $B/src/bxjudge/_native.abi3.so $X/src/bxjudge/
  cp -al $B/weights $X/weights; cp -al $B/calib $X/calib; mkdir -p $X/runs; cp -al $B/runs/pretrained $X/runs/pretrained
fi
mkdir -p $E/videos; for v in 20261024_201331 20261024_201410 20261024_201444; do ln -sf "$HOME/bx/videos/20261024/"$v.mp4 $E/videos/$v.mp4; done
cat > $E/v7_selected.json <<J
$(python3 -c "import json;d=json.load(open('$B/weights/selected.json'));d['weights']='$B/runs/train-v7-120fps-20261124/fit/checkpoint_best_ema.pth';print(json.dumps(d,indent=1))")
J
cd $X
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
for cfg in production v7; do
  if [ $cfg = production ]; then SEL=$B/weights/selected.json; TRT=$B/weights/rfdetr-trt/manifest.json; else SEL=$E/v7_selected.json; TRT=$B/weights/rfdetr-trt-v7-candidate/manifest.json; fi
  OUT=$E/$cfg
  [ -f $OUT/rounds.json ] && { echo "$cfg already done"; continue; }
  echo "$(date +%T) start $cfg"
  /usr/bin/time -f "%e s wall" env PYTHONPATH=$X/src $B/.venv/bin/python -m bxjudge.cli run $E/videos --out $OUT \
    --selection $SEL --trt-manifest $TRT \
    --dynamic-calibration $X/config/dynamic_calibration.yaml --two-bey-tracking $X/config/two_bey_tracking.yaml \
    --stadium-keypoints --no-vlm > $E/$cfg.log 2>&1 || { echo "$cfg FAILED"; tail -30 $E/$cfg.log; exit 1; }
  tail -2 $E/$cfg.log
done
echo "$(date +%T) baselines done"
