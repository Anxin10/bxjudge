# Auto Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `bxjudge run` derive the 8 calibration zones for a slightly moved camera from existing manual calibrations (templates), record an immutable calibration snapshot per video, reanalyse when the chosen calibration changes, and let the reviewer veto a bad auto calibration without hiding it from evaluation.

**Architecture:** A new `autocalib.py` estimates a template→frame homography on polarity-free edge maps and warps the template polygons. A new `calibration_store.py` owns content-addressed snapshots in `<results>/calibrations/`, the manual-first → reuse-auto → new-auto selection, previews and vetoes. `pipeline.py`, `output.py`, `evaluate.py`, `vlm.py` and the review UI consume snapshots by id; nothing re-selects a calibration after the run.

**Tech Stack:** Python 3.12, OpenCV (`cv2.findTransformECC`, `cv2.perspectiveTransform`), NumPy, PyYAML, Streamlit (existing review UI), pytest, uv. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-23-auto-calibration-design.md`

## Global Constraints

- Everything runs locally in WSL; no network, no new AI model (spec §4, §6.2).
- Every numeric policy lives in `config/*.yaml`; new ones go in a new `config/autocalib.yaml` (R40).
- Pixel thresholds are in inference pixels (long edge `infer_long_edge` = 1280) (R15).
- All 8 zones (`calibration.ZONES`) are transformed, stored and drawn (R49).
- Edge comparison ignores brightness polarity and colour: blurred Sobel gradient magnitude, mask = stadium minus battle_area, as `calibration_similarity` does (R13, spec §6.2).
- Selection order: manual with `calibration_similarity ≥ calibration_min_similarity` (0.8) wins and is used unrefined; else best auto ≥ `min_score`; else `recalibrate` (spec §6.1, §6.5c).
- An auto snapshot's `ref_frame` is the video's own first-second frame, never the template image (spec §6.2b).
- Auto results are never written into the user's `calib/` (spec §6.3).
- Manual and auto snapshots are immutable copies in `<results>/calibrations/<id>.json` + `<id>.png`; `id` = first 16 hex of SHA-256 over the snapshot's canonical JSON + reference-image bytes (spec §6.4).
- No compatibility path for results without a calibration record: never guess zones (spec §6.5, §6.5b).
- Vetoed rounds stay in the evaluation population (spec §6.6b); vetoes never rewrite judgments (spec §6.5).
- Boundary-error acceptance threshold initial value: 8 inference px (spec §6.6).
- Reviewed rows are preserved on reanalysis (R36, spec §6.3b).

## Review Focus

- First-second frame with a hand, launcher or bey over the rim: alignment must either stay accurate or be rejected, never accepted misaligned (test in Task 1).
- Template whose reference PNG is missing or unreadable: skipped with a warning, other templates still tried (test in Task 2).
- Results folder moved or copied elsewhere: snapshots still load (reference image path stored relative to the snapshot JSON) (test in Task 2).
- One batch with two camera positions: two distinct auto snapshots; vetoing one leaves the other's matches intact (test in Task 4).
- Template calibrated at 1920×1080 while frames are decoded at 1280×720, and 120 fps clips: polygons scaled correctly, timing untouched (test in Task 3).

---

### Task 1: Edge-map alignment core

**Files:**
- Create: `src/bxjudge/autocalib.py`, `config/autocalib.yaml`, `tests/test_autocalib.py`
- Modify: `tests/helpers.py` (add `stadium_scene`)

**Interfaces:**
- Consumes: `calibration.ZONES`, `calibration.rim_mask`, `calibration._edge_map`, `calibration.scale_calibration`, `types.Calibration`.
- Produces:
  - `@dataclass(frozen=True) class AutoCalibConfig: work_width: int; min_score: float; max_shift_frac: float; max_rotation_deg: float; min_scale: float; max_scale: float; max_perspective: float; ecc_iterations: int; ecc_epsilon: float; max_boundary_error_px: float`
  - `load_autocalib_config(path: Path = Path("config/autocalib.yaml")) -> AutoCalibConfig` (rejects missing/extra keys, non-finite, `min_scale >= max_scale`, scores outside [0,1])
  - `@dataclass(frozen=True) class Alignment: homography: tuple[tuple[float, float, float], ...]  # 3x3 template-px → frame-px; score: float; shift_frac: float; rotation_deg: float; scale: float; perspective: float`
  - `estimate_alignment(frame: np.ndarray, template: Calibration, template_image: np.ndarray, cfg: AutoCalibConfig) -> Alignment | None`
  - `warp_calibration(template: Calibration, alignment: Alignment, frame_size: tuple[int, int], name: str) -> Calibration` (all 8 polygons through the homography; `ref_frame=""`)
  - `mean_boundary_error(a: Calibration, b: Calibration, long_edge: int) -> float` (symmetric mean distance between corresponding zone boundaries, both scaled so the long edge = `long_edge`)
- `tests/helpers.py`: `stadium_scene(size=(1280, 720), floor="white", shift=(0, 0), rotation_deg=0.0, scale=1.0) -> tuple[np.ndarray, Calibration]` draws a synthetic stadium (outer rim, lid ellipse, battle floor, three pockets as filled polygons with textured edges) and returns the image and its exact 8-zone calibration; `floor="black"` inverts the floor brightness.

`config/autocalib.yaml` initial values: `work_width: 640`, `min_score: 0.6`, `max_shift_frac: 0.15`, `max_rotation_deg: 8.0`, `min_scale: 0.85`, `max_scale: 1.15`, `max_perspective: 0.0005`, `ecc_iterations: 200`, `ecc_epsilon: 1.0e-6`, `max_boundary_error_px: 8.0`.

- [ ] **Step 1: Write the failing tests**

```python
def test_recovers_shift_rotation_scale_within_tolerance():
    tpl_img, tpl = stadium_scene()
    frame, truth = stadium_scene(shift=(40, -25), rotation_deg=3.0, scale=1.06)
    cfg = load_autocalib_config()
    a = estimate_alignment(frame, tpl, tpl_img, cfg)
    assert a is not None and a.score >= cfg.min_score
    got = warp_calibration(tpl, a, (1280, 720), "auto")
    assert mean_boundary_error(got, truth, 1280) < 2.0

def test_black_floor_aligns_to_white_template():
    tpl_img, tpl = stadium_scene(floor="white")
    frame, truth = stadium_scene(floor="black", shift=(20, 10))
    a = estimate_alignment(frame, tpl, tpl_img, load_autocalib_config())
    assert a is not None
    assert mean_boundary_error(warp_calibration(tpl, a, (1280, 720), "auto"), truth, 1280) < 3.0

@pytest.mark.parametrize("frame_kind", ["rotated_30deg", "unrelated_noise", "blank"])
def test_out_of_range_or_unrelated_frames_are_rejected(frame_kind):
    tpl_img, tpl = stadium_scene()
    frame = {"rotated_30deg": stadium_scene(rotation_deg=30.0)[0],
             "unrelated_noise": np.random.default_rng(0).integers(0, 255, (720, 1280, 3), np.uint8),
             "blank": np.zeros((720, 1280, 3), np.uint8)}[frame_kind]
    assert estimate_alignment(frame, tpl, tpl_img, load_autocalib_config()) is None

def test_hand_over_rim_is_accurate_or_rejected():
    tpl_img, tpl = stadium_scene()
    frame, truth = stadium_scene(shift=(30, 0))
    frame[0:260, 300:700] = (60, 90, 140)  # skin-coloured block over the far rim
    a = estimate_alignment(frame, tpl, tpl_img, load_autocalib_config())
    if a is not None:
        assert mean_boundary_error(warp_calibration(tpl, a, (1280, 720), "auto"), truth, 1280) < 8.0

def test_mean_boundary_error_identity_and_offset():
    _, c = stadium_scene()
    assert mean_boundary_error(c, c, 1280) == pytest.approx(0.0, abs=1e-6)
    _, moved = stadium_scene(shift=(10, 0))
    assert 8.0 < mean_boundary_error(c, moved, 1280) < 12.0

def test_config_rejects_bad_values(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text("work_width: 640\nmin_score: 1.5\n")
    with pytest.raises(ValueError):
        load_autocalib_config(p)
```

- [ ] **Step 2: Run** `uv run pytest tests/test_autocalib.py -v` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement `autocalib.py`**

Approach (the one non-obvious algorithm): resize both images to `work_width`; edge maps via `calibration._edge_map` (polarity-free); normalise each map to zero mean / unit std inside the template's `rim_mask` (dilated slightly) so contrast differences between floor colours do not dominate; `cv2.findTransformECC(template_map, frame_map, H, cv2.MOTION_HOMOGRAPHY, criteria, inputMask=mask)` from identity, run coarse-to-fine on a 2-level pyramid (half size, then full `work_width`); catch `cv2.error` → `None`. Rescale H back to template-px → frame-px. Decompose: `shift_frac` = translation of the template stadium centroid ÷ frame width; `rotation_deg`, `scale` from the affine 2×2 part (polar decomposition); `perspective = max(|H[2][0]|, |H[2][1]|)` after normalising `H[2][2]=1` at full resolution. Reject outside any configured bound. `score` = zero-mean NCC (clipped to [0,1]) between the warped template edge map and the frame edge map over the warped rim mask; reject below `min_score`.

- [ ] **Step 4: Run** `uv run pytest tests/test_autocalib.py -v` — Expected: PASS.

- [ ] **Step 5: Commit** — `git add src/bxjudge/autocalib.py config/autocalib.yaml tests/test_autocalib.py tests/helpers.py && git commit -m "feat: edge-map template alignment for auto calibration"`

---

### Task 2: Calibration snapshots and selection

**Files:**
- Create: `src/bxjudge/calibration_store.py`, `tests/test_calibration_store.py`

**Interfaces:**
- Consumes: Task 1 (`AutoCalibConfig`, `estimate_alignment`, `warp_calibration`); `calibration.load_calibration`, `calibration_similarity`, `scale_calibration`, `ZONES`; `config.Thresholds.calibration_min_similarity`.
- Produces:
  - `@dataclass(frozen=True) class Snapshot: id: str; kind: str  # "manual" | "auto"; calibration: Calibration  # ref_frame = absolute path of <id>.png; score: float; template_id: str | None; source_path: str | None; source_sha256: str | None; homography: tuple | None`
  - `save_snapshot(out_dir: Path, kind: str, calibration: Calibration, ref_image: np.ndarray, *, score: float, template_id: str | None = None, source_path: Path | None = None, homography: tuple | None = None) -> Snapshot` — writes `<out_dir>/calibrations/<id>.json` (ref image stored as relative `<id>.png`) and `<id>.png`; idempotent for identical content.
  - `load_snapshot(out_dir: Path, snapshot_id: str) -> Snapshot` (raises `FileNotFoundError` for an unknown id).
  - `select_snapshot(frame: np.ndarray, calib_dir: Path, out_dir: Path, th: Thresholds, cfg: AutoCalibConfig) -> Snapshot | None` — order: (1) best manual in `calib_dir` with similarity ≥ `th.calibration_min_similarity` → manual snapshot (copy of that file's polygons + PNG, `source_path`, `source_sha256` of the JSON); (2) best existing **auto** snapshot in `<out_dir>/calibrations/` with similarity ≥ threshold → reuse it (this is the grouping: same position = same snapshot); (3) each manual template → `estimate_alignment`; best score ≥ `cfg.min_score` → new auto snapshot whose ref image is `frame` and whose polygons are the warped template; (4) else `None`.
  - `write_preview(out_dir: Path, snap: Snapshot) -> Path` — `<out_dir>/calibrations/<id>.preview.jpg`: reference image with all 8 zones drawn and a text line `"{kind} {calibration.name} template={template_id} score={score:.3f}"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_manual_wins_when_similar(tmp_path):  # calib_dir holds the scene's manual calibration
    calib, out = make_calib_dir(tmp_path, stadium_scene())  # helper in this test file
    frame, _ = stadium_scene()
    s = select_snapshot(frame, calib, out, th(), load_autocalib_config())
    assert s.kind == "manual" and s.source_sha256 and s.template_id is None

def test_shifted_frame_gets_auto_snapshot_with_its_own_reference(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    frame, truth = stadium_scene(shift=(45, 20))
    s = select_snapshot(frame, calib, out, th(), load_autocalib_config())
    assert s.kind == "auto" and s.template_id
    assert np.array_equal(cv2.imread(s.calibration.ref_frame), frame)
    assert mean_boundary_error(s.calibration, truth, 1280) < 3.0
    assert not list(calib.glob("auto*")) and len(list(calib.glob("*.json"))) == 1

def test_second_video_at_same_position_reuses_the_auto_snapshot(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    a = select_snapshot(stadium_scene(shift=(45, 20))[0], calib, out, th(), load_autocalib_config())
    b = select_snapshot(stadium_scene(shift=(45, 20))[0], calib, out, th(), load_autocalib_config())
    assert a.id == b.id

def test_snapshot_survives_manual_file_edit_and_folder_move(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    s = select_snapshot(stadium_scene()[0], calib, out, th(), load_autocalib_config())
    (calib / "scene.json").write_text("{}")
    moved = tmp_path / "moved"; shutil.copytree(out, moved)
    again = load_snapshot(moved, s.id)
    assert again.calibration.stadium == s.calibration.stadium and Path(again.calibration.ref_frame).is_file()

def test_unreadable_template_is_skipped(tmp_path, caplog):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    (calib / "broken.json").write_text(json.dumps({**json.loads((calib / "scene.json").read_text()), "name": "broken", "ref_frame": "missing.png"}))
    assert select_snapshot(stadium_scene(shift=(45, 20))[0], calib, out, th(), load_autocalib_config()).kind == "auto"
    assert "broken" in caplog.text

def test_no_match_returns_none(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    assert select_snapshot(np.zeros((720, 1280, 3), np.uint8), calib, out, th(), load_autocalib_config()) is None

def test_snapshot_id_is_content_addressed(tmp_path):
    img, c = stadium_scene()
    a = save_snapshot(tmp_path, "manual", c, img, score=1.0)
    b = save_snapshot(tmp_path, "manual", c, img, score=1.0)
    assert a.id == b.id and len(a.id) == 16
```

- [ ] **Step 2: Run** `uv run pytest tests/test_calibration_store.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement `calibration_store.py`** per the Interfaces block. Canonical JSON = `json.dumps(..., sort_keys=True, separators=(",", ":"))` of kind, name, frame_size, 8 polygons rounded to 0.01 px, score rounded to 1e-4, template_id, source_sha256, homography rounded to 1e-9; hash also covers the PNG bytes (`cv2.imencode(".png", ref_image)`).
- [ ] **Step 4: Run** `uv run pytest tests/test_calibration_store.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** — `feat: content-addressed calibration snapshots and manual-first selection`

---

### Task 3: Pipeline uses snapshots; records them; reanalyses on change

**Files:**
- Modify: `src/bxjudge/pipeline.py` (`judge_video`, `run_batch`), `src/bxjudge/output.py` (`_result_row`/`ROUND_FIELDS` untouched; JSON row gains `calibration_id`; new `record_video_calibration`), `src/bxjudge/manifest.py` (processed record gains `calibration_id`), `src/bxjudge/types.py` (`RoundResult.calibration_id: str | None = None`), `src/bxjudge/cli.py` (`run --autocalib config/autocalib.yaml`)
- Test: `tests/test_pipeline.py`, `tests/test_manifest.py`

**Interfaces:**
- Consumes: Task 2 (`select_snapshot`, `write_preview`, `Snapshot`); Task 1 (`load_autocalib_config`).
- Produces:
  - `judge_video(info, calib_dir, detector, rules, th, *, out_dir, provenance=None, detection_config=..., autocalib: AutoCalibConfig) -> list[RoundResult]` — selects with `select_snapshot(first_second_frame, calib_dir, out_dir, th, autocalib)`; `None` → the existing `recalibrate` placeholder; otherwise scales `snap.calibration` to the decoded size, uses `snap.calibration.ref_frame` as the displacement reference, sets `calibration_id=snap.id` on every result, calls `write_preview`.
  - `record_video_calibration(out_dir: Path, video: str, snap: Snapshot | None) -> None` — `videos.json[video]["calibration"] = {"id", "kind", "name", "score", "template_id"}` or `None`.
  - `mark_processed(info: VideoInfo, db_path: Path, *, calibration_id: str | None) -> None`; `processed_calibration(db_path: Path, path: Path) -> str | None` (`calibration_id` absent in an old record → treated as "changed", so the video is reanalysed once).
  - `run_batch(..., autocalib_config: Path = Path("config/autocalib.yaml"), autocalib: AutoCalibConfig | None = None)` — besides `scan`'s new/changed videos, for every previously processed video it decodes the first second, runs `select_snapshot`, and for a non-vetoed recorded id reanalyses when the id differs from `processed_calibration`, including when selection now returns `None` after a previous success. For a recorded vetoed id (Task 4 `vetoed_ids`; T4-3), reanalyse only after selection yields a usable non-vetoed alternative; after unveto, use the normal id comparison (the same restored snapshot does not need reanalysis); while none is usable keep original rows and predictions untouched, never replace them with a veto-generated `recalibrate` placeholder. Preserve reviewed rows and the spec §6.6b evaluation population.

- [ ] **Step 1: Write the failing tests** (existing `FakeDetector` / synthetic-video helpers in `tests/helpers.py`)

```python
def test_moved_camera_is_judged_with_auto_snapshot_and_recorded(tmp_path):
    # video rendered from stadium_scene(shift=(45, 20)); calib dir has only the unshifted manual
    run_batch(videos, calib, out, detector=FakeDetector(...), th=th(), rules=rules(), autocalib=cfg())
    v = json.loads((out / "videos.json").read_text())["clip.mp4"]
    assert v["calibration"]["kind"] == "auto"
    rows = json.loads((out / "rounds.json").read_text())
    assert {r["calibration_id"] for r in rows} == {v["calibration"]["id"]}
    assert (out / "calibrations" / f"{v['calibration']['id']}.preview.jpg").is_file()

def test_adding_a_manual_calibration_reanalyses_an_auto_video(tmp_path):
    run_batch(...)  # first run: auto
    add_manual_calibration(calib, stadium_scene(shift=(45, 20)))
    calls = counting_detector()
    run_batch(..., detector=calls)
    assert calls.videos == {"clip.mp4"}
    assert json.loads((out / "videos.json").read_text())["clip.mp4"]["calibration"]["kind"] == "manual"

def test_unchanged_calibration_is_not_reanalysed(tmp_path):
    run_batch(...); calls = counting_detector(); run_batch(..., detector=calls)
    assert calls.videos == set()

def test_reanalysis_keeps_reviewed_rows(tmp_path):
    run_batch(...); confirm_reviewed(out, "clip.mp4", 1)
    add_manual_calibration(calib, stadium_scene(shift=(45, 20))); run_batch(...)
    row = next(r for r in json.loads((out / "rounds.json").read_text()) if r["round"] == 1)
    assert row["reviewed"] is True

def test_template_at_1080p_scales_to_720p_decode_and_keeps_120fps_timing(tmp_path):
    # manual calibration saved at 1920x1080, clip decoded at 1280x720 and probed at 120 fps
    run_batch(...)
    snap = load_snapshot(out, vid["calibration"]["id"])
    assert scale_calibration(snap.calibration, (1280, 720)).frame_size == (1280, 720)
    assert rows[0]["t_launch"] == pytest.approx(expected_launch_frame / 120.0, abs=1e-6)
```

- [ ] **Step 2: Run** `uv run pytest tests/test_pipeline.py tests/test_manifest.py -v` — Expected: new tests FAIL.
- [ ] **Step 3: Implement** per the Interfaces block. The pre-run selection pass reuses the first-second decode code of `judge_video` (extract a small `first_second_frame(info, th) -> np.ndarray` helper; do not duplicate it).
- [ ] **Step 4: Run** focused tests, then `uv run pytest -q` — Expected: all PASS.
- [ ] **Step 5: Commit** — `feat: pipeline judges with recorded calibration snapshots and reanalyses on change`

---

### Task 4: Vetoes, matches and evaluation population

**Files:**
- Modify: `src/bxjudge/calibration_store.py` (vetoes), `src/bxjudge/output.py` (`save_rounds` match exclusion), `src/bxjudge/evaluate.py` (calibration section, review ratio), `src/bxjudge/pipeline.py` (pass `vetoed_ids` into the reanalysis check)
- Test: `tests/test_calibration_store.py`, `tests/test_output.py`, `tests/test_evaluate.py`

**Interfaces:**
- Consumes: Task 2 snapshots; Task 3 `calibration_id` on rows and `videos.json[...]["calibration"]`.
- Produces:
  - `veto_calibration(out_dir: Path, snapshot_id: str, *, note: str = "") -> None`, `unveto_calibration(out_dir: Path, snapshot_id: str) -> None`, `vetoed_ids(out_dir: Path) -> set[str]` — file `<out_dir>/calibration_vetoes.json` `{id: {"at": iso, "note": str}}`; unknown id → `ValueError`; both rebuild CSVs through `save_rounds`.
  - `save_rounds` excludes from `matches.csv` every **unreviewed** row whose `calibration_id` is vetoed; `rounds.json`/`rounds.csv` keep them unchanged.
  - `evaluate(...)` Metrics gain `n_vetoed_rounds: int`, `n_recalibrate_videos: int`, `n_vetoed_groups: int`, `n_vetoed_videos: int`, `coverage: float  # videos judged with a non-vetoed snapshot / all videos in videos.json`, `n_pending_videos: int  # recalibrate + vetoed`; vetoed unreviewed rounds count as needing human work in `review_ratio`; vetoed rows stay in every accuracy population with their original predictions; audit-sample population is unchanged. The eval report shows these in a "校準 Calibration" section.

- [ ] **Step 1: Write the failing tests**

```python
def test_veto_excludes_unreviewed_rounds_from_matches_only(tmp_path):
    out = results_with_two_positions(tmp_path)  # videos a.mp4 (snapshot S1) and b.mp4 (S2), 2 rounds each
    veto_calibration(out, S1)
    matches = read_matches(out)
    assert all(r["video"] == "b.mp4" for m in matches for r in m["rounds"])
    assert len(json.loads((out / "rounds.json").read_text())) == 4
    unveto_calibration(out, S1)
    assert {r["video"] for m in read_matches(out) for r in m["rounds"]} == {"a.mp4", "b.mp4"}

def test_reviewed_rows_under_veto_still_score(tmp_path):
    out = results_with_two_positions(tmp_path); confirm_reviewed(out, "a.mp4", 1); veto_calibration(out, S1)
    assert ("a.mp4", 1) in {(r["video"], r["round"]) for m in read_matches(out) for r in m["rounds"]}

def test_veto_unknown_id_rejected(tmp_path):
    with pytest.raises(ValueError):
        veto_calibration(results_with_two_positions(tmp_path), "0" * 16)

def test_vetoed_rounds_stay_in_population_and_raise_review_ratio():
    rows = [rev(calibration_id="S1", needs_review=False, fix={}), rev(calibration_id="S2", needs_review=False, fix={}),
            row(calibration_id="S1", needs_review=False), row(calibration_id="S2", needs_review=False)]
    m = evaluate(rows, timings=videos_with_calibration({"a": "S1", "b": "S2"}), vetoed={"S1"})
    assert m.n_rounds == 4 and m.n_vetoed_rounds == 2
    assert m.review_ratio == pytest.approx(0.25)  # the one unreviewed vetoed round
    assert m.coverage == pytest.approx(0.5) and m.n_vetoed_groups == 1 and m.n_pending_videos == 1

def test_recalibrate_videos_counted_at_video_level():
    m = evaluate([placeholder("c.mp4", "recalibrate")], timings=videos_with_calibration({"c": None}), vetoed=set())
    assert m.n_recalibrate_videos == 1 and m.coverage == 0.0
```

- [ ] **Step 2: Run** the three test files — Expected: new tests FAIL.
- [ ] **Step 3: Implement** per the Interfaces block (`evaluate` gains keyword `vetoed: set[str] | None = None`; the CLI `eval` passes `vetoed_ids(results)`).
- [ ] **Step 4: Run** focused tests, then `uv run pytest -q` — Expected: PASS.
- [ ] **Step 5: Commit** — `feat: calibration vetoes that keep rounds in the evaluation population`

---

### Task 5: Review UI and VLM read the recorded snapshot

**Files:**
- Modify: `src/bxjudge/review_ui.py`, `src/bxjudge/review_app.py`, `src/bxjudge/vlm.py`, `src/bxjudge/cli.py` (`review` and `vlm` no longer need `--calib`; remove the option)
- Test: `tests/test_review_ui.py`, `tests/test_vlm.py`

**Interfaces:**
- Consumes: Task 2 (`load_snapshot`), Task 3 (`calibration_id`, `videos.json` calibration record), Task 4 (`veto_calibration`, `unveto_calibration`, `vetoed_ids`).
- Produces:
  - `review_ui.row_calibration(out_dir: Path, row: dict) -> Snapshot | None` — snapshot by the row's `calibration_id`; `None` when the row has none (results made before this feature).
  - `review_ui.calibration_status(out_dir: Path, row: dict) -> str` — one of `"ok"`, `"vetoed"`, `"changed_since_review"` (reviewed row whose `calibration_id` differs from `videos.json`'s current one or is vetoed), `"no_record"`.
  - Remove `review_ui.video_calibration` and `vlm.select_calibration`; `vlm.event_frames(source, t, snapshot: Snapshot, config, th)` takes the snapshot instead of `calib_dir`. A row without a record gets `vlm_unavailable: no calibration record` and the batch continues.
  - UI copy (exact): banner for `no_record`: `此結果沒有校準紀錄，請用新版重新執行 bxjudge run`; for `changed_since_review`: `覆核當時使用的校準已被否決／更換，請確認是否需要重新覆核`; veto button: `不採用此組校準`; undo button: `恢復採用此組校準`; info line: `校準：{kind}（{name}）分數 {score:.3f}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_overlay_uses_recorded_snapshot_not_calib_dir(tmp_path):
    out = results_with_auto_snapshot(tmp_path); shutil.rmtree(tmp_path / "calib")
    assert row_calibration(out, read_rounds(out)[0]).kind == "auto"

def test_no_record_banner_and_no_overlay(tmp_path):
    out = results_without_calibration_ids(tmp_path)
    at = app_test(out); at.run()
    assert any("此結果沒有校準紀錄" in m.value for m in at.warning)

def test_veto_button_round_trip(tmp_path):
    out = results_with_auto_snapshot(tmp_path); at = app_test(out); at.run()
    at.button(key=f"veto-{SNAP_ID}").click().run()
    assert SNAP_ID in vetoed_ids(out)
    at.button(key=f"unveto-{SNAP_ID}").click().run()
    assert SNAP_ID not in vetoed_ids(out)

def test_reviewed_row_flagged_when_calibration_vetoed(tmp_path):
    out = results_with_auto_snapshot(tmp_path); confirm_reviewed(out, "clip.mp4", 1); veto_calibration(out, SNAP_ID)
    assert calibration_status(out, find_row(out, "clip.mp4", 1)) == "changed_since_review"

def test_vlm_row_without_record_is_unavailable_and_batch_continues(tmp_path, fake_ollama):
    out = mixed_results(tmp_path)  # one row with calibration_id, one without
    review_results_dir(out, th=th(), rules=rules(), config=vlm_cfg())
    notes = {r["video"]: r.get("vlm_note", "") for r in read_rounds(out)}
    assert "vlm_unavailable: no calibration record" in notes["old.mp4"] and notes["new.mp4"]
```

- [ ] **Step 2: Run** `uv run pytest tests/test_review_ui.py tests/test_vlm.py -v` — Expected: new tests FAIL.
- [ ] **Step 3: Implement** per the Interfaces block; update every caller of the removed functions (no shim).
- [ ] **Step 4: Run** focused tests, then `uv run pytest -q` — Expected: PASS.
- [ ] **Step 5: Commit** — `feat: review UI and VLM use recorded calibration snapshots; calibration veto in UI`

---

### Task 6: `bxjudge autocalib` acceptance tool, real-footage check, README

**Files:**
- Modify: `src/bxjudge/cli.py` (new `autocalib` subcommand), `README.md` (auto-calibration section in the 完整流程 list, before 判定)
- Test: `tests/test_cli_autocalib.py`

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces: `bxjudge autocalib VIDEO [VIDEO ...] --calib calib --out DIR [--truth CALIB_JSON] [--autocalib config/autocalib.yaml]` — for each video: first-second frame → `select_snapshot` (snapshots and previews under `DIR`), prints one line `"{video}: {kind|recalibrate} name={name} score={score:.3f}"`; with `--truth`, also prints `boundary_error_px={e:.2f} PASS|FAIL` against `max_boundary_error_px` and exits 1 on any FAIL. Never writes to `--calib`.

- [ ] **Step 1: Write the failing tests**

```python
def test_autocalib_cli_reports_auto_and_passes_truth(tmp_path, capsys):
    calib = make_calib_dir_with(stadium_scene()); video = render_video(tmp_path, stadium_scene(shift=(45, 20)))
    truth = save_truth(tmp_path, stadium_scene(shift=(45, 20))[1])
    assert main(["autocalib", str(video), "--calib", str(calib), "--out", str(tmp_path / "o"), "--truth", str(truth)]) == 0
    out = capsys.readouterr().out
    assert "auto" in out and "PASS" in out
    assert sorted(p.name for p in calib.iterdir()) == ["scene.json", "scene.png"]

def test_autocalib_cli_fails_on_bad_truth(tmp_path):
    ...  # truth shifted by 60 px from the video → exit code 1 and "FAIL"
```

- [ ] **Step 2: Run** `uv run pytest tests/test_cli_autocalib.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** the subcommand and README section (commands: `bxjudge autocalib ~/bx/videos/20261014/*.mp4 --calib calib --out runs/autocalib-check`; how to read the previews; how to veto in the review UI; that black / transparent floors are unverified until footage exists).
- [ ] **Step 4: Run** focused tests, then `uv run pytest -q` — Expected: PASS.
- [ ] **Step 5: Real-footage check (record results in the commit message body):** `uv run bxjudge autocalib ~/bx/videos/20261001/20261001_201622.mp4 ~/bx/videos/20261014/*.mp4 --calib calib --out runs/autocalib-check-20261023`. Expected: the 10/01 clip → `manual 20261001`; each 10/14 clip → `auto` with a preview, or `recalibrate`. Open the previews and state whether the zones look aligned; do not tune thresholds without the user's manual truth for 10/14 (spec §8).
- [ ] **Step 6: Commit** — `feat: bxjudge autocalib acceptance tool and docs`
