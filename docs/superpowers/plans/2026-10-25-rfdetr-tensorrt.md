# RF-DETR TensorRT Runtime Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline in the existing branch.

**Goal:** Run the verified RF-DETR FP16 TensorRT backend through the real batch pipeline with explicit provenance and no silent fallback.
**Architecture:** Keep the existing RF-DETR decoder/preprocessing/postprocessing. A narrow TensorRT module validates a hashed export manifest and attaches the upstream synchronous runtime. `run --trt-manifest` explicitly selects that backend; training/comparison remain native and existing selections are not rewritten.
**Tech Stack:** RF-DETR 1.11.2, TensorRT-cu13 10.16.1.11, PyTorch CUDA, uv, pytest.
**Spec:** User's next-steps request and approved detector benchmark at `runs/next-steps-20261025/README.md`; binding handoff `.superpowers/sdd/codex-handoff-2026-10-24.md`.

## Global Constraints
- Work only on feat/local-judge; no new branches/worktrees, no push, preserve unrelated files.
- Do not change judging, tracking, calibration thresholds, population or existing results.
- Missing/mismatched engine evidence is a startup error, never a native-backend fallback.
- Policy export values live in config/tensorrt.yaml; engine hash, versions and GPU identity are measured metadata.
- Validate with tests, independent read-only review and personal checks before committing.
- A detector benchmark is not full pipeline speed or referee accuracy acceptance.

## Review Focus
- A different checkpoint, modified engine or wrong resolution must fail before batch writes.
- A missing runtime, different RF-DETR/TRT version or GPU must fail with actionable errors.
- Color order, source coordinates, class/background slots and original confidence stay unchanged.
- TensorRT must wait for upstream CUDA preprocessing and preserve output ordering.
- Export must refuse to overwrite an existing artifact directory; failure never publishes a complete manifest.

### Task 1: Hashed artifact and adapter (tests/test_tensorrt.py, src/bxjudge/tensorrt.py, src/bxjudge/detect.py)
**Interfaces:** `load_trt_manifest(path: Path, weights: Path, long_edge: int) -> dict`; `attach_tensorrt(model, artifact: dict, device: str) -> None`; `RFDETRDetector(..., trt_manifest: Path | None = None)` exposes runtime evidence.
- [ ] Write failing tests for manifest tampering/resolution/schema/relative paths; upstream I/O/dtype/stream/tuple behavior; constructor activation.
- [ ] Run focused pytest, observe missing functionality, implement minimal module and adapter, run green.

### Task 2: Production entrypoint and reproducible export (pipeline.py, cli.py, config/tensorrt.yaml, scripts/export_rfdetr_trt.py, pyproject.toml, uv.lock, README.md)
**Interfaces:** `run_batch(..., trt_manifest: Path | None = None)` / `bxjudge run --trt-manifest PATH`; per-round runtime provenance contains the exact validated manifest. `export_rfdetr_trt(weights, output, config, thresholds, detection_config) -> Path` builds using upstream export and publishes manifest last.
- [ ] Write failing CLI/pipeline fail-before-write and provenance tests; export no-overwrite/settings tests.
- [ ] Implement explicit flag and export settings validation; declare a pinned optional tensorrt extra, preserving installed Torch.
- [ ] Run focused tests, full suite, lint baseline comparison and diff checks.

### Task 3: Independent review and real pilot
- [ ] Freeze source/diff/requirements, request read-only independent review, verify each finding personally and fix/retest/re-review if blocking.
- [ ] Build a new formal artifact, verify real-frame detections against benchmark within documented floating-point differences, run a bounded whole-clip pipeline pilot into a fresh output folder.
- [ ] Report actual continuous pipeline wall time and human-review needs; preserve the 57 unmatched calibration cases.
- [ ] Commit only the reviewed code, tests and documentation once blocking findings are resolved. No push.
