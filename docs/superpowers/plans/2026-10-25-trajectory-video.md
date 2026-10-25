# Trajectory playback implementation plan

> Execute inline with superpowers:executing-plans, on feat/local-judge. No new branch/worktree.

**Goal:** Replace the primary round trajectory still with a seekable progressive H.264 MP4 generated from recorded observations.
**Architecture:** Share pure plotting helpers with the existing static writer. A separate video module reads the saved CSV and recorded calibration, incrementally paints observed segments, and encodes via the already installed ffmpeg. A content-addressed, atomically published cache prevents stale playback. The review UI creates the selected round once and uses native video controls, with static overview in an expander.
**Tech Stack:** Existing OpenCV/NumPy/Streamlit, stdlib subprocess/tempfile, ffmpeg/libx264.
**Spec:** User request: 該回合軌跡圖做成可撥放進程的影片.

## Global constraints
- Preserve raw results, reviews, CSV, calibration, timing, judge/tracker/identity behavior; no models or original video analysis needed.
- X orange/B blue, unknown gray. First/last actual observation per fragment, not assumed physical launch/stop.
- No future trajectory in earlier frames, no connections across missing source frames, no interpolated detections.
- One-timescale based on saved source timestamps; retain silent intervals. Background is the recorded calibration reference, not the final trajectory PNG.
- Browser-compatible H.264/yuv420p/faststart; policy in config/trajectory_video.yaml; codec error must be explicit, no fallback.
- Commit after tests, independent read-only review, personal verification/fixes/re-review. Keep unrelated files untouched.

## Review focus
- Cache invalidation on changed CSV/calibration/reference image/timeline/policy/renderer.
- Last observation must appear, including sub-frame tails and stationary/one-point tracks.
- Invalid/missing CSV or reference does not produce fabricated/stale playback.
- Encoder failure or interruption never publishes a partial MP4; paths with spaces are argv, not shell commands.
- Source coordinates use the recorded decoded-frame plane, not snapshot dimensions; the writer emits a CSV-hash-bound coordinate metadata JSON. Missing metadata is an explicit error; no inference/fallback from current settings. Bottom legend must not cover evidence.

### Task 1: Renderer and video encoder
Files: src/bxjudge/trajectory_plot.py, trajectory_video.py, output.py; config/trajectory_video.yaml; tests/test_trajectory_video.py.
Interfaces: plotting helpers draw base, segments, endpoints, footer; generate_trajectory_video(out, row, calibration, *, config) -> Path.
- [ ] RED: future visibility/gaps/unknown identity/time range/endpoints; real H.264 codec/timestamps/seeking input integrity; cache invalidation and failed encode.
- [ ] Implement incremental drawing/atomic MP4 publication; share plot helpers without changing static CSV outputs.
- [ ] Focused tests and static rendering regressions GREEN.

### Task 2: Review page and verification
Files: review_app.py; tests/test_review_ui.py; README.md.
- [ ] UI shows playback, timing/palette explanation and static overview; failed media leaves human review controls usable.
- [ ] Generate current 3 real round animations from saved CSV/calibration only. Verify browser, codec, duration, first/mid/end frames, and protected data hashes.
- [ ] Full pytest, lint baseline, independent frozen-source review; fix/retest/re-review if needed; commit and report.
