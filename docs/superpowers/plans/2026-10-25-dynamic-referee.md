# Dynamic Calibration and Referee Evidence Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. User explicitly requested implementation in the existing branch; do not create branches/worktrees or pause for repeated scope confirmation.

**Goal:** Recover stadium geometry, consume it in judging, and conservatively repair launch-phase/identity evidence with measurable small-batch validation.
**Architecture:** Reference-anchored streaming registration with confirmed relocation segments; raw detection/footage stays in source pixels, adjudication observations are normalized to immutable reference pixels. Held-bey filtering and unique fragment linking act on actual observations only.
**Tech Stack:** Existing Python3.12, OpenCV, NumPy, ByteTrack, RF-DETR TensorRT, pytest; no new dependency.
**Spec:** docs/superpowers/specs/2026-10-25-dynamic-referee-design.md

## Global Constraints
- feat/local-judge only; preserve existing private/untracked files and reviewed pilot.
- Explicit required YAML keys, no compatibility defaults. Existing static mode is the controlled baseline, dynamic mode explicit.
- Missing geometry stays unknown; no stale/identity substitution; no synthetic detection points.
- Original raw displacement and first-second motion reference remain untouched.
- R30 release deadline3sec remains causal; do not infer launch from landing or finish.
- Source/producing-run timing R66 and review population/audit rules remain.

## Review Focus
- Relocation drift or accepting a partly occluded lookalike: final candidate reprojection/rim gates and stable confirmations; Task1.
- Frame-index or coordinate-plane mismatch: inverse centre and box transform tests, Task2.
- Failed geometry looking like a genuine disappearance/exit: invalid interval must force review and block auto result; Task2.
- Held launcher disappears without observed separation: never silently claim a release; Task3.
- Fragment crossing or two equally plausible candidates: refuse to join; Task3.

### Task 1: Confirmed relocation and recovery preview
Files: new src/bxjudge/dynamic_calibration.py, config/dynamic_calibration.yaml, tests/test_dynamic_calibration.py; modify trajectory_video.py and review caption/cache.
Interfaces: DynamicCalibration(c, reference, fps, follow_policy, recovery_policy, autocalib).update(frame_index,image) -> GeometrySample(frame,alignment|None,segment,status). to_record() serializable immutable reference evidence.
- [ ] RED: persistent translation outside small-follow bounds must recover; single occluded candidate cannot switch; bad frame never emits last transform; sequential confirmation, no drift.
- [ ] Implement high-score periodic re-localization to original snapshot and confirmed anchor switch; all thresholds strict YAML.
- [ ] GREEN targeted tests; replay three weak scene windows and compare acceptance without changing original small-motion gates.

### Task 2: Recorded per-frame geometry in core judging
Files: dynamic_calibration.py, pipeline.py, cli.py, types.py/output.py if needed; tests/test_dynamic_pipeline.py.
Interfaces: normalize_tracks(tracks,samples) -> reference-pixel Track list preserving ids/frames/scores; raw export derives same grouped identities from raw observations. Explicit --dynamic-calibration PATH.
- [ ] RED: moving floor/pocket translates while normalized judgment stays identical; unavailable alignment makes review mandatory; raw exports and motion evidence remain unchanged.
- [ ] Integrate continuous measurement, traceable per-video geometry record and hash-backed row provenance, exact crop keys fromraw.
- [ ] GREEN pipeline and CLI tests; no results overwrite; dynamic duration covered by existing wall clock.

### Task 3: Observed launcher separation and conservative fragment continuity
Files: new src/bxjudge/launch_phase.py, config/launch_phase.yaml, tests/test_launch_phase.py; pipeline.py integration in explicit dynamic mode.
Interfaces: prepare_launch_tracks(raw_tracks,fps,policy) -> prepared tracks plus raw-support/provenance/uncertainty. Link only mutual unique short-gap motion-compatible candidates, trim only observed held-to-separated sequence; preserves actual points.
- [ ] RED: long mounted wait does not cause false missed landing; missing launcher separation stays unknown; no join across overlap/crossing; unique short gap links with original frame/score counts.
- [ ] Implement causal held/release tracking and supported identity joins; ambiguity remains review.
- [ ] GREEN launch/finish/track pipeline regression suite.

### Task 4: Bounded end-to-end validation and review
Files: runs/dynamic-referee-20261025/*, README, this plan ledger.
- [ ] Fixed synthetic end-to-end moved stadium + launcher release + exit event.
- [ ] Same two fully decoded real clips in new out with formal TensorRT engine; no VLM model, reviewed pilot unchanged.
- [ ] Report full wall time, review reasons, unknowns, matching truth limitations. No overallaccuracy claim without labels.
- [ ] Full suite, lint baseline comparison, frozen whole-change read-only review; personally reproduce valid findings, fix/retest; commit explicit files only, no push.
