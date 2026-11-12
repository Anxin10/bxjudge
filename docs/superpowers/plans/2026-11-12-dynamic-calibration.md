# 動態校準（固定／手持兩種模式）實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **本計畫的執行方式（使用者已定）：** Claude 依序實作，Codex 以唯讀模式審查（`codex exec`／`codex review`）。每個 Task 完成後交 Codex 審查；Critical／High／Medium 修正後重測再複審，沒有阻斷問題才 commit。

**Goal：** 讓判定流程用關鍵點找到場地，並依鏡頭是固定還是手持，選擇「小幅跟隨」或「定期重找＋跟隨」，使區域在整回合都對得上；對不上的區間明確標為未知。

**Architecture：**
- 新模組 `stadium_locator.py` 把關鍵點擬合包成「把校準座標對到任一格畫面」的定位器，並負責建立 `keypoints` 種類的校準快照。
- 新模組 `camera_motion.py` 量開頭的鏡頭移動，決定模式（可依批次或影片覆寫）。
- 固定模式沿用 `DynamicCalibration`，跟丟重定位時多試關鍵點。
- 手持模式由新類別 `HandheldCalibration` 負責，輸出與 `DynamicCalibration` 相同的 `GeometrySample` 流與 `record()` 格式，所以下游不需要知道是哪種模式。

**Tech Stack：** Python 3.12、OpenCV、NumPy、Ultralytics YOLO-pose（現有 4 模型集成）、pytest。

**Spec：** `docs/superpowers/specs/2026-11-12-dynamic-calibration-design.md`

## Global Constraints

- repo 在 WSL `/home/<user>/bxjudge`，分支 `feat/local-judge`。指令都在 WSL 內執行（例：`.venv/bin/python -m pytest -q -p no:cacheprovider ...`；從 Git Bash 呼叫時前置 `MSYS_NO_PATHCONV=1 wsl.exe -d ubuntu -- bash -lc "cd ~/bxjudge && ..."`）。
- 不動他人未提交的修改：`.gitignore`、`README.md`、`pyproject.toml`、`scripts/build_reid_review.py`、`scripts/setup_wsl.sh`、`src/bxjudge/finish.py`、`src/bxjudge/geometry.py`、`src/bxjudge/web/reid_review.*`、`src/bxjudge/workbench_server.py`、`tests/test_geometry.py`、`tests/test_workbench_server.py`，以及未追蹤的 `rust/`、`judge_view.py`、`bx_fx.js`、`bx_theme.css`、`judge_view.js`、`tests/test_judge_evidence.py`、`workbench_restyle.py`、`workbench_theme.py`。受保護的未追蹤檔（`.agents/ AGENTS.md CLAUDE.md .claude.code-workspace antigravity.db .graftignore .graphifyignore graphify-out/ src/graphify-out/`）不 stage、不刪。
- 只提交自己的檔案：`git commit -o <paths>`（新檔先 `git add -- <新檔>`）；commit 訊息結尾 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`。不 push、不 merge。
- `pipeline.py`、`launch_check.py`、`rules.py` 是 Codex 也在改的檔案（落地回歸 D265／G555／202324）。動這些檔案前先 `git status --short` 與 `git log -3`，確認 Codex 沒有未提交的修改；有的話等它提交。
- 原始影片、人工真值、`calib/`、811 驗證／測試集、Label Studio 匯出唯讀。測試一律用 `tmp_path` 複本。
- 用 GPU 前先跑 `nvidia-smi`；Codex 可能同時在用。
- 時間門檻一律用**播放秒**（ruling T4），換成樣本數時乘以影片播放 fps：`max(1, round(秒數 × fps))`。20261001 是 240 FPS 慢動作、以 30 FPS 播放；其他批次 120／120（`config/video_sources.yaml`）。
- 新門檻放 YAML，不寫死在程式裡；所有長度單位是推論座標（長邊 1280，`thresholds.yaml` 的 `infer_long_edge`）。
- 不加 `--stadium-keypoints` 時，判定流程的行為與輸出必須和現在**完全一樣**。
- 寧可輸出未知或 `recalibrate`，也不輸出偏掉的區域（spec 反目標）。
- 改完程式碼後跑 `graphify update .`。

## Review Focus

1. **甩鏡頭或長時間遮擋：** 手持影片中場地好幾秒看不到時，那段必須是未知，不能沿用舊的區域；之後的區間要等重找確認後才恢復，不回填。→ Task 5 `test_whip_pan_stays_unknown_until_a_refind_confirms`。
2. **固定機位被誤判成手持**（例如手在畫面中大幅晃動）：影片照常判定，只有觸碰變成 `unknown`，並在 provenance 記錄量到的移動量；覆寫成 `fixed` 後恢復原本行為。→ Task 3 `test_override_beats_measurement`、Task 6 `test_handheld_round_marks_touch_unknown_and_notes_camera`。
3. **手持影片裡找不到任何場地**（沒有場地、開頭全被遮住）：整支輸出 `recalibrate` 並寫明原因，不當掉。→ Task 6 `test_handheld_without_locatable_reference_is_recalibrate`。
4. **影片中途關鍵點推論出錯**（GPU 顯存不足、驅動錯誤）：那一次重找當作「沒找到」，記下原因，不中斷整支影片。→ Task 2 `test_locate_turns_inference_errors_into_a_reason`、Task 5 `test_refind_error_is_recorded_and_tracking_continues`。
5. **240 FPS 慢動作**：重找間隔以播放秒換成格數，30 FPS 播放與 120 FPS 播放的重找格距要不同。→ Task 5 `test_refind_interval_uses_playback_fps`。

---

## 檔案結構

| 檔案 | 動作 | 責任 |
|---|---|---|
| `src/bxjudge/stadium_locator.py` | 新增 | `StadiumTools`、`KeypointLocator`、`LocatedStadium`、`keypoint_snapshot`、`zone_disagreement_px` |
| `src/bxjudge/camera_motion.py` | 新增 | 量鏡頭移動、決定模式、讀 `config/camera_mode.yaml` |
| `src/bxjudge/handheld_calibration.py` | 新增 | `HandheldCalibration`（手持模式逐格幾何） |
| `config/camera_mode.yaml` | 新增 | 模式判斷、手持重找、覆寫設定 |
| `src/bxjudge/calibration_store.py` | 修改 | 快照種類加 `keypoints` |
| `src/bxjudge/dynamic_calibration.py` | 修改 | 固定模式跟丟時用關鍵點重定位；`finish()` |
| `src/bxjudge/launch_check.py` | 修改 | `analyze_launch(..., motion_unknown_reason=)` |
| `src/bxjudge/pipeline.py` | 修改 | 快照第三順位、模式判斷、選擇幾何類別、手持時不量盤面位移 |
| `src/bxjudge/cli.py` | 修改 | `--stadium-keypoints`、`--camera-mode` |
| `src/bxjudge/review_ui.py` | 修改 | 新原因的處理步驟 |
| `scripts/eval_dynamic_calibration.py` | 新增 | 驗證：模式判斷、重找準度、幾何覆蓋率與漂移 |
| `tests/test_stadium_locator.py`、`tests/test_camera_motion.py`、`tests/test_handheld_calibration.py`、`tests/test_eval_dynamic_calibration.py` | 新增 | 對應測試 |
| `tests/test_calibration_store.py`、`tests/test_dynamic_calibration.py`、`tests/test_launch_check.py`、`tests/test_dynamic_pipeline.py`、`tests/test_review_ui.py` | 修改 | 加測試 |

---

### Task 1：`keypoints` 校準快照

**Files：**
- Modify: `src/bxjudge/calibration_store.py`（`_KINDS`、`_canonical`、`save_snapshot`、`_checked_meta`）
- Test: `tests/test_calibration_store.py`

**Interfaces：**
- Consumes：現有 `save_snapshot(out_dir, kind, calibration, ref_image, *, score, template_id, source_path, homography, template_set, autocalib) -> Snapshot`、`load_snapshot(out_dir, snapshot_id, *, autocalib) -> Snapshot`。
- Produces：`_KINDS = ("manual", "auto", "keypoints")`。`keypoints` 快照的 `homography` 是**場地模型 px → 該快照參考畫面 px**（`StadiumFit.homography`）；`score` 參與 id，和 `auto` 一樣存原值；`template_id`、`template_set`、`source_path` 都是 `None`。

- [ ] **Step 1：寫失敗測試**

```python
def test_keypoint_snapshot_round_trips_and_score_is_part_of_identity(tmp_path, autocalib_cfg):
    c = make_calibration()
    img = np.zeros((720, 1280, 3), np.uint8)
    h = ((1.0, 0.0, 5.0), (0.0, 1.0, 7.0), (0.0, 0.0, 1.0))
    a = save_snapshot(tmp_path, "keypoints", c, img, score=0.91, homography=h, autocalib=autocalib_cfg)
    b = save_snapshot(tmp_path, "keypoints", c, img, score=0.92, homography=h, autocalib=autocalib_cfg)
    assert a.kind == "keypoints" and a.homography == h and a.score == pytest.approx(0.91)
    assert a.template_id is None and a.template_set is None and a.source_path is None
    assert a.id != b.id
    assert load_snapshot(tmp_path, a.id, autocalib=autocalib_cfg) == a

def test_keypoint_snapshot_requires_a_homography(tmp_path, autocalib_cfg):
    with pytest.raises(ValueError, match="homography"):
        save_snapshot(tmp_path, "keypoints", make_calibration(), np.zeros((720, 1280, 3), np.uint8),
                      score=0.9, autocalib=autocalib_cfg)
```

（`autocalib_cfg` 用該測試檔現有的 AutoCalibConfig fixture 或載入方式，名稱照檔內慣例。）

- [ ] **Step 2：跑測試確認失敗**　`.venv/bin/python -m pytest -q tests/test_calibration_store.py -k keypoint` → FAIL（kind 不在 `_KINDS`）。
- [ ] **Step 3：實作**　`_canonical` 與 `stored_score` 對 `auto`、`keypoints` 都保留 score；`keypoints` 沒有 homography 時 `ValueError("a keypoints snapshot needs its homography")`；`_checked_meta` 接受新種類，並套用與 `auto` 相同的 homography 形狀檢查。
- [ ] **Step 4：跑測試確認通過**　同上指令，加跑整個 `tests/test_calibration_store.py` → 全過。
- [ ] **Step 5：交 Codex 唯讀審查，修正後 commit**　`git commit -o src/bxjudge/calibration_store.py tests/test_calibration_store.py -m "Store keypoint-found stadium snapshots"`（訊息加 Co-Authored-By）。

---

### Task 2：關鍵點定位器與快照建立

**Files：**
- Create: `src/bxjudge/stadium_locator.py`
- Test: `tests/test_stadium_locator.py`

**Interfaces：**
- Consumes：Task 1 的 `keypoints` 快照；`stadium_finder.find_stadium(video, model, cfg, *, hwaccel, keypoints, keypoint_errors) -> FoundStadium | None`（`FoundStadium(calibration, fit, frame_index, frame, zone_visibility, agreeing_frames, method)`）；`fit_from_keypoints(points, reference, mask, model, cfg) -> StadiumFit | None`；`rail_mask(frame, cfg)`；`_AGREEMENT_ZONES`；`load_stadium_finder_config()`、`load_stadium_model()`；`KeypointFinder.from_config(cfg, reference=None, device=None)`、`.predict(frame) -> (10,3) | None`、`.reference`；`autocalib.Alignment`、`autocalib._decompose`、`autocalib._stadium_centroid`、`warp_calibration`。
- Produces：
  - `@dataclass(frozen=True) class StadiumTools: model: StadiumModel; cfg: StadiumFinderConfig; finder: KeypointFinder`
  - `load_stadium_tools(device: str | int | None = None) -> StadiumTools`（只載入一次，供整批影片共用）
  - `@dataclass(frozen=True) class LocatedStadium: homography: np.ndarray  # 校準 px → 畫面 px；alignment: Alignment；fit: StadiumFit`
  - `class KeypointLocator`：
    - `__init__(self, tools: StadiumTools, calibration: Calibration, model_to_calibration: np.ndarray)`
    - `@classmethod for_snapshot(cls, tools, snap: Snapshot, calibration: Calibration, reference_image: np.ndarray) -> "KeypointLocator | None"`：`keypoints` 快照直接用 `snap.homography`；其他種類在 `reference_image` 上跑一次擬合，`model_to_calibration = fit.homography`；擬合失敗時以 `_log.warning` 記下原因（`no_keypoints`／`fit_rejected`／`keypoint_error: …`）並回傳 `None`。
    - `locate(self, image: np.ndarray) -> LocatedStadium | None`；`last_reason: str | None`（`"no_keypoints"`、`"fit_rejected"`、`"keypoint_error: <型別>: <訊息>"`）。
  - `zone_disagreement_px(c: Calibration, h1: np.ndarray, h2: np.ndarray) -> float`：`_AGREEMENT_ZONES` 四區，各區平均頂點距離取最大值（與 `_worst_zone_shift` 同定義，但用校準多邊形、畫面 px）。
  - `alignment_from_homography(c: Calibration, h: np.ndarray, score: float) -> Alignment`：shift／rotation／scale／perspective 的算法與 `DynamicCalibration._compose` 相同，不做邊界檢查。
  - `keypoint_snapshot(video: Path, out_dir: Path, tools: StadiumTools, autocalib, vetoed, *, hwaccel: str = "cpu") -> tuple[Snapshot | None, FoundStadium | None]`：跑 `find_stadium`；**只接受 `method == "keypoints"`**（軌道法結果回傳 `(None, found)`，因為量測顯示軌道法較不準）；存成 `keypoints` 快照，參考畫面是 `found.frame`，`score = found.fit.coverage`，`homography = found.fit.homography`；id 在 `vetoed` 內時回傳 `(None, found)`。

定位的算法：`pts = finder.predict(image)` → `fit = fit_from_keypoints(pts, finder.reference, rail_mask(image, cfg), model, cfg)` → `h = fit.homography @ inv(model_to_calibration)`，正規化 `h[2,2] = 1`。捕捉 `(RuntimeError, ValueError, cv2.error)`。

- [ ] **Step 1：寫失敗測試**（用假的 finder：`predict` 回傳以已知單應投影的 10 點，rail mask 用 `tests/helpers.py` 的合成場地；參考 `tests/test_stadium_keypoints.py` 的合成方式）

```python
def test_locate_maps_calibration_to_a_moved_view(tools_fake):
    loc = KeypointLocator(tools_fake, c, H_BASE)          # 模型 → 校準 px
    tools_fake.finder.view = MOVE @ H_BASE                # 畫面相對校準平移 40 px
    got = loc.locate(frame_for(MOVE))
    assert got is not None and zone_disagreement_px(c, got.homography, MOVE) < 2.0

def test_locate_turns_inference_errors_into_a_reason(tools_fake):
    tools_fake.finder.raise_with = RuntimeError("CUDA out of memory")
    loc = KeypointLocator(tools_fake, c, H_BASE)
    assert loc.locate(frame_for(np.eye(3))) is None
    assert loc.last_reason.startswith("keypoint_error: RuntimeError")

def test_for_snapshot_fits_the_reference_of_a_manual_snapshot_and_gives_none_when_it_cannot(tools_fake, tmp_path): ...
def test_keypoint_snapshot_rejects_rail_results_and_vetoed_ids(tmp_path, monkeypatch, tools_fake): ...
def test_zone_disagreement_is_the_worst_zone_not_the_mean(): ...   # 只偏一區 30 px，其餘 0 → 30
```

- [ ] **Step 2：確認失敗**　`.venv/bin/python -m pytest -q tests/test_stadium_locator.py` → FAIL（模組不存在）。
- [ ] **Step 3：實作** `src/bxjudge/stadium_locator.py`（介面如上）。
- [ ] **Step 4：確認通過**　同上 → 全過；加跑 `-m gpu tests/test_stadium_keypoints.py`（先 `nvidia-smi`）確認 `load_stadium_tools` 能載入真模型。
- [ ] **Step 5：審查後 commit**　`git add -- src/bxjudge/stadium_locator.py tests/test_stadium_locator.py`，`git commit -o` 這兩個檔，訊息 "Locate the stadium in any frame from keypoints"。

---

### Task 3：鏡頭模式判斷

**Files：**
- Create: `src/bxjudge/camera_motion.py`、`config/camera_mode.yaml`
- Test: `tests/test_camera_motion.py`

**Interfaces：**
- Produces：
  - `load_camera_policy(path: Path = DEFAULT_CAMERA_MODE_CONFIG) -> dict`：鍵集合必須完全等於 YAML 範本，數值須為正且有限，覆寫值只能是 `fixed`、`handheld`、`auto`；違反時 `ValueError`（與 `load_dynamic_policy` 同風格）。
  - `@dataclass(frozen=True) class CameraMotion: px_per_s: float | None; pairs: int`
  - `measure_camera_motion(frames: list[np.ndarray], step_s: float) -> CameraMotion`：相鄰兩格做 ORB（3000 特徵）＋交叉比對＋RANSAC 單應（3 px），四角平均位移；匹配少於 30 或單應失敗的配對略過；`px_per_s = 中位位移 / step_s`，沒有有效配對時為 `None`。（算法照 scratchpad 的 `cam_motion.py`。）
  - `@dataclass(frozen=True) class CameraMode: mode: Literal["fixed", "handheld"]; source: Literal["measured", "override", "unmeasured"]; motion: CameraMotion | None`
  - `decide_camera_mode(video: Path, batch: str, motion: CameraMotion | None, policy: dict) -> CameraMode`：覆寫優先（`overrides.videos[檔名]` 先於 `overrides.batches[批次]`，值 `auto` 等於沒覆寫）；否則 `px_per_s > handheld_px_per_s` 為手持；有效配對少於 `min_pairs` 時判 `handheld`、`source="unmeasured"`（量不到就用較保守的手持模式）。
  - `sample_for_motion(video: Path, fps: float, policy: dict, *, long_edge: int, hwaccel: str) -> list[np.ndarray]`：用 `iter_frames(video, 0.0, window_s, long_edge=..., hwaccel=...)`，每 `max(1, round(step_s × fps))` 格取一格。

`config/camera_mode.yaml` 初值（之後由 Task 8 實測校正）：

```yaml
# Camera mode (spec 2026-11-12-dynamic-calibration-design.md). Lengths in inference px (long
# edge 1280), times in playback seconds. Measured 2026-11-12: fixed ~5.7 px/s, handheld 56-74.
detection:
  window_s: 10.0
  step_s: 1.0
  min_pairs: 3
  handheld_px_per_s: 20.0
handheld:
  refind_s: 1.0
  confirm_px: 12.0
  max_unchecked_s: 3.0
overrides:
  batches: {}
  videos: {}
```

- [ ] **Step 1：寫失敗測試**

```python
def test_still_frames_measure_near_zero_and_moving_frames_measure_the_shift():
    base = textured_frame()                                  # 有足夠角點的合成畫面
    still = measure_camera_motion([base] * 5, 1.0)
    moving = measure_camera_motion([shift(base, 40 * i) for i in range(5)], 1.0)
    assert still.px_per_s < 1.0 and moving.px_per_s == pytest.approx(40, abs=2) and moving.pairs == 4

def test_mode_threshold_and_unmeasured_fall_back_to_handheld(policy): ...
def test_override_beats_measurement(policy):
    p = {**policy, "overrides": {"batches": {"20261101": "fixed"}, "videos": {"a.mp4": "handheld"}}}
    assert decide_camera_mode(Path("x/20261101/b.mp4"), "20261101", CameraMotion(70.0, 9), p).mode == "fixed"
    assert decide_camera_mode(Path("x/20261101/a.mp4"), "20261101", CameraMotion(1.0, 9), p).source == "override"

def test_policy_rejects_missing_keys_and_unknown_override_values(tmp_path): ...
def test_sampling_steps_in_playback_seconds(tmp_path): ...   # 用 tests/helpers 的 make_video，fps 30 與 120 取樣格號不同
```

- [ ] **Step 2：確認失敗**　`.venv/bin/python -m pytest -q tests/test_camera_motion.py` → FAIL。
- [ ] **Step 3：實作**。
- [ ] **Step 4：確認通過**。
- [ ] **Step 5：審查後 commit**（新檔先 `git add --`）訊息 "Decide fixed or handheld camera from early motion"。

---

### Task 4：固定模式跟丟時用關鍵點重定位

**Files：**
- Modify: `src/bxjudge/dynamic_calibration.py`（`DynamicCalibration.__init__`、`update` 的重定位分支、新增 `finish`）
- Test: `tests/test_dynamic_calibration.py`

**Interfaces：**
- Consumes：Task 2 的 `KeypointLocator.locate`、`LocatedStadium.alignment`。
- Produces：`DynamicCalibration(c, reference, fps, follow_policy, policy, *, temporal_policy=None, keypoint_locator: KeypointLocator | None = None)`；`finish(self) -> None`（固定模式為空操作，讓 pipeline 對兩種模式一視同仁地呼叫）。

行為：重定位時先跑現有的 `relocator.align(image)`；它回傳 `None` 且有 `keypoint_locator` 時，改用 `locate(image)`，候選必須通過與 relocator 相同的寬鬆全域限制（`max_shift_frac`、`max_rotation_deg`、`min_scale`、`max_scale`，相對原始校準），再走原本的 `confirmations` 次一致確認。由關鍵點確認的新段落 `kind = "relocated_keypoints"`；`diagnostic["relocation"]["keypoints"]` 記 `last_reason` 或 `"located"`。沒有 `keypoint_locator` 時行為完全不變。

- [ ] **Step 1：寫失敗測試**

```python
def test_keypoints_relocate_when_the_reference_relocator_cannot(c, ref):
    fake = FakeLocator(per_frame={f: moved_alignment(c, dx=60) for f in range(40, 80)})
    g = tracker(c, ref, keypoint_locator=fake)                # 沿用檔內 tracker() 工廠，加參數
    for f, img in moved_sequence_where_rim_matching_fails(dx=60, frames=range(80)):
        g.update(f, img)
    assert any(s["kind"] == "relocated_keypoints" for s in g.segments)
    assert all(g.samples[f].alignment is None for f in range(40, g.segments[-1]["start_frame"]))   # 不回填

def test_keypoint_relocation_obeys_the_global_bounds(c, ref): ...   # 偏移超過 max_shift_frac 的候選永遠不被採用
def test_without_a_locator_behaviour_is_unchanged(c, ref): ...      # 同序列，與加參數前的 samples／segments 完全相等
```

- [ ] **Step 2：確認失敗**。
- [ ] **Step 3：實作**。
- [ ] **Step 4：確認通過**　`.venv/bin/python -m pytest -q tests/test_dynamic_calibration.py tests/test_dynamic_pipeline.py tests/test_stadium_follow.py` → 全過。
- [ ] **Step 5：審查後 commit**　訊息 "Relocate a fixed camera from keypoints after the rim match fails"。

---

### Task 5：手持模式 `HandheldCalibration`

**Files：**
- Create: `src/bxjudge/handheld_calibration.py`
- Test: `tests/test_handheld_calibration.py`

**Interfaces：**
- Consumes：Task 2 的 `KeypointLocator.locate`、`zone_disagreement_px`、`alignment_from_homography`；`StadiumFollower(c, reference, policy).align(frame) -> Alignment | None`、`.set_foreground_mask(mask, minimum_visible)`；`GeometrySample(frame, alignment, segment, status)`；`warp_calibration`；`DynamicCalibration.record()` 的格式；`save_geometry`／`read_geometry`／`recorded_calibration`。
- Produces：
  - `HandheldCalibration(c, reference, fps, follow_policy, handheld_policy, locator, *, minimum_visible: float)`
  - 屬性 `samples: dict[int, GeometrySample]`、`segments: list[dict]`
  - `set_foreground(self, detections) -> None`（框轉成遮罩後交給 follower）
  - `update(self, frame: int, image: np.ndarray) -> GeometrySample`
  - `finish(self) -> None`
  - `record(self) -> dict`：與 `DynamicCalibration.record()` 同鍵（`schema: 1`、`reference_size`、`coordinate_mapping`、`policy`、`temporal_policy: None`、`segments`、`samples`），另加 `"mode": "handheld"` 與 `"refinds": [{"frame", "result", "disagreement_px", "reason"}]`。

**狀態與規則**（簽名決定不了的部分）：

- 兩種狀態：**已確認**（輸出 alignment，status `tracking`，重找確認那一格為 `refound`）與**未確認**（alignment `None`，status `unconfirmed`；跟隨失敗的那一格 status `lost`）。
- 起始：已確認，段落 0 `kind="recorded_snapshot"`，follower 的參考是快照參考畫面，`anchor_h = I`；`last_confirm = 0`。
- 逐格：`p = follower.align(image)` 轉成校準 px → 畫面 px（`p.homography @ anchor_h`）。不套 `DynamicCalibration` 的全域位移限制（手持本來就會大幅移動），只依 follower 本身相對參考格的限制。已確認狀態下 `p` 為 `None` → 這格 `lost`，轉為未確認，保留目前的 follower 當暫定參考。
- 重找：自上次重找起滿 `refind_n = max(1, round(refind_s × fps))` 格時，`k = locator.locate(image)`：
  - `k` 為 `None`：記 `result="no_fit"`，狀態不變。
  - 已確認且 `p` 存在、`zone_disagreement_px(c, p, k) ≤ confirm_px`：記 `agree`，follower 改以這一格為參考、`anchor_h = k.homography`（消除累積漂移），`last_confirm = frame`。
  - 已確認但 `p` 為 `None` 或不一致：記 `disagree`；`(last_confirm, frame]` 已輸出的 sample 全部改為 `None`、status `invalidated`（反目標：錯的區域比未知更糟）；轉為未確認，follower 改以這一格、`k` 為暫定參考。
  - 未確認且 `p` 存在、與 `k` 一致：記 `agree`，開新段落（`kind="refound"`、`start_frame=frame`、`confirmed_frames=[暫定參考格, frame]`），轉為已確認；之前未確認的格**不回填**。
  - 未確認但不一致或 `p` 為 `None`：記 `provisional`，follower 改以這一格、`k` 為暫定參考。
- 已確認狀態超過 `max_unchecked_n = round(max_unchecked_s × fps)` 格沒有成功的 `agree`：轉為未確認（後續格不輸出），已輸出的不動。
- `finish()`：若最後一格不是重找格，對保留的最後一張畫面補做一次重找，套用同樣的規則。
- 不做內插、不做橋接；`normalize_tracks` 看到 `None` 就略過。

- [ ] **Step 1：寫失敗測試**（合成序列：`tests/helpers.py` 的合成場地，以已知單應逐格移動；`FakeLocator` 依格號回傳指定單應或 `None`，也可指定丟例外字串）

```python
def test_follows_a_slow_pan_and_refinds_reset_the_anchor(): ...      # 全部格都有 alignment，segments 只有 0，refinds 全為 agree
def test_disagreeing_refind_invalidates_back_to_the_last_confirmation():
    g = run(seq_pan(120), locator=FakeLocator(wrong_at={60: shift(50)}), fps=30, refind_s=1.0)
    assert [g.samples[f].status for f in (30, 31, 59, 60)] == ["refound", "invalidated", "invalidated", "invalidated"]
    assert g.samples[30].alignment is not None
def test_whip_pan_stays_unknown_until_a_refind_confirms():
    g = run(seq_with_jump(at=45, dx=400, frames=150), locator=FakeLocator(truth=True), fps=30, refind_s=1.0)
    assert all(g.samples[f].alignment is None for f in range(45, 90))
    assert g.samples[90].status == "refound" and g.segments[-1]["start_frame"] == 90
def test_refind_interval_uses_playback_fps():
    assert refind_frames(run(seq_pan(130), fps=30, refind_s=1.0)) == [30, 60, 90, 120]
    assert refind_frames(run(seq_pan(250), fps=120, refind_s=1.0)) == [120, 240]
def test_refind_error_is_recorded_and_tracking_continues(): ...       # result="no_fit"，reason 以 "keypoint_error" 開頭，sample 仍輸出
def test_long_unchecked_tracking_stops_being_emitted(): ...           # locator 一直 None，超過 max_unchecked_s 後 alignment 為 None
def test_finish_checks_the_tail(): ...                                # 尾段被最後一次重找否定 → invalidated
def test_record_round_trips_through_saved_geometry(tmp_path): ...     # save_geometry → read_geometry → recorded_calibration 可用，mode == "handheld"
```

- [ ] **Step 2：確認失敗**。
- [ ] **Step 3：實作**。
- [ ] **Step 4：確認通過**　`.venv/bin/python -m pytest -q tests/test_handheld_calibration.py tests/test_dynamic_calibration.py` → 全過。
- [ ] **Step 5：審查後 commit**　訊息 "Track a handheld camera by refinding the stadium from keypoints"。

---

### Task 6：接入判定流程與 CLI

**Files：**
- Modify: `src/bxjudge/launch_check.py`（`analyze_launch`）、`src/bxjudge/pipeline.py`（`judge_video` 與批次呼叫處約 1204 行）、`src/bxjudge/cli.py`（judge 子命令參數，約 151、169 行附近）
- Test: `tests/test_launch_check.py`、`tests/test_dynamic_pipeline.py`

**Interfaces：**
- Consumes：Task 1–5 全部。
- Produces：
  - `analyze_launch(..., motion_unknown_reason: str | None = None)`：給值時兩側觸碰都是 `unknown`，備註只有一筆 `f"stadium_motion_unknown@{reason}"`，不呼叫 `detect_touch`；其餘不變。
  - `judge_video(..., stadium_tools: StadiumTools | None = None, camera_policy: dict | None = None)`
  - CLI：`--stadium-keypoints`（啟用關鍵點快照與模式判斷；載入一次 `load_stadium_tools(device=0)`，啟動前 log `nvidia-smi` 需求）、`--camera-mode YAML`（預設 `config/camera_mode.yaml`，只在 `--stadium-keypoints` 時讀取）。

`judge_video` 的流程（只在 `stadium_tools` 不為 `None` 時改變）：

1. `select_snapshot` 回傳 `None` 後，呼叫 `keypoint_snapshot(info.path, out_dir, stadium_tools, autocalib, vetoed, hwaccel=th.decode_hwaccel)`；仍為 `None` 才輸出 `recalibrate`。`on_snapshot` 回報實際採用的快照。`FoundStadium.zone_visibility` 低於 `stadium_finder` 現行門檻的袋區，在每回合加上 `zone_out_of_frame@<區域>`（門檻與判斷方式照 `scripts/eval_stadium_finder.py` 計算 `pocket_out_of_frame` 的作法）。provenance 另記 `record["stadium_keypoints"] = {"snapshot_id", "frame_index", "agreeing_frames", "coverage", "precision", "keypoint_error_px", "keypoint_rail_shift_px"}`（spec：記錄方法、格號與擬合指標）。
2. `camera_policy` 不為 `None` 時：`decide_camera_mode(info.path, info.path.parent.name, measure_camera_motion(sample_for_motion(...)), camera_policy)`，結果寫進 provenance 的 `record["camera"] = {"mode", "source", "px_per_s", "pairs"}`。
3. 手持時：
   - `dynamic_config is None` → 整支 `recalibrate`，原因 `handheld_requires_dynamic_calibration`。
   - `KeypointLocator.for_snapshot(...)` 為 `None` → 整支 `recalibrate`，原因 `handheld_reference_unlocated`。
   - 否則 `geometry = HandheldCalibration(c, ref_image, info.fps, follow_policy, camera_policy["handheld"], locator, minimum_visible=load_temporal_policy()["min_visible_fraction"])`（有 `timing` 時用 `converted(...)` 後的值，與現有 `DynamicCalibration` 一致）。
   - `measured()` 不跑 `iter_stadium_displacement`，直接迭代 `chain(prefix, frames)`，每格 `disp[f] = math.nan`（保留逐格帳目，`len(disp)` 檢查不變）。
   - 呼叫 `analyze_launch(..., motion_unknown_reason="camera_handheld")`；每回合 review reasons 加 `camera_handheld`。
4. 固定時：`DynamicCalibration(..., keypoint_locator=KeypointLocator.for_snapshot(...))`（可為 `None`）。
5. 解碼迴圈結束後、`geometry.record()` 之前呼叫 `geometry.finish()`。

- [ ] **Step 1：寫失敗測試**

```python
def test_motion_unknown_reason_makes_touch_unknown_without_scanning(th): ...      # test_launch_check.py
def test_without_stadium_keypoints_results_are_identical(tmp_path, monkeypatch, th, rules): ...
    # 同一支合成影片，judge_video 不給 stadium_tools 與給 None：rows、geometry 證據逐欄相等
def test_keypoint_snapshot_is_the_third_choice(tmp_path, monkeypatch, th, rules): ...
    # select_snapshot → None，keypoint_snapshot 回傳假快照 → 採用它；兩者都 None → recalibrate
def test_handheld_round_marks_touch_unknown_and_notes_camera(tmp_path, monkeypatch, th, rules): ...
    # 假 decide_camera_mode 回傳 handheld；檢查每回合 launch.touch 全為 unknown、review_reasons 含 camera_handheld、
    # provenance camera.mode == "handheld"、geometry 證據 mode == "handheld"
def test_handheld_without_locatable_reference_is_recalibrate(tmp_path, monkeypatch, th, rules): ...
def test_handheld_without_dynamic_calibration_is_recalibrate(tmp_path, monkeypatch, th, rules): ...
```

（monkeypatch 對象：`pipeline.keypoint_snapshot`、`pipeline.decide_camera_mode`、`pipeline.KeypointLocator`，假物件放在測試檔內；照 `tests/test_dynamic_pipeline.py` 現有的假偵測器與合成影片寫法。）

- [ ] **Step 2：確認失敗**。
- [ ] **Step 3：實作**（先確認 Codex 沒有 `pipeline.py`／`launch_check.py` 的未提交修改）。
- [ ] **Step 4：確認通過**　`.venv/bin/python -m pytest -q tests/test_launch_check.py tests/test_dynamic_pipeline.py tests/test_pipeline.py tests/test_two_bey_integration.py tests/test_launch_phase.py tests/test_rules.py` → 全過；再跑全套非 GPU：`YOLO_OFFLINE=1 .venv/bin/python -m pytest -q -m "not gpu" -p no:cacheprovider`（`tests/test_judge_evidence.py` 缺 node 的 2 項失敗為既有狀況，照實記錄）。
- [ ] **Step 5：審查後 commit**　訊息 "Judge with keypoint stadiums and per-camera geometry modes"。

---

### Task 7：覆核顯示

**Files：**
- Modify: `src/bxjudge/review_ui.py`（`repair_steps`）
- Test: `tests/test_review_ui.py`

**Interfaces：**
- Consumes：Task 6 的原因字串 `handheld_requires_dynamic_calibration`、`handheld_reference_unlocated`、`camera_handheld`、`zone_out_of_frame@<區域>`。
- Produces：
  - `repair_steps` 對前兩者給出繁體中文處理步驟（前者：加上 `--dynamic-calibration` 重跑；後者：開頭畫面找不到場地，人工校準或確認影片）；`camera_handheld` 在回合備註中顯示為「手持鏡頭：發射觸碰無法判定」。照檔內現有字串表的寫法。
  - `geometry_gaps(samples: list[dict], first: int, last: int) -> list[dict]`：從幾何證據的 `samples`（`record()` 格式）找出回合 `[first, last]` 內 alignment 為空的連續區間，回傳 `[{"start", "end", "status", "reason"}]`（status／reason 取區間第一格）；回合覆核頁列出這些區間（spec：覆核要看得到哪些區間未知與原因）。

- [ ] **Step 1：寫失敗測試** `test_repair_steps_explain_handheld_reasons`（兩個原因各產生至少一行含指令或說明的步驟）、`test_camera_handheld_note_is_labelled`、`test_geometry_gaps_merge_consecutive_unknown_frames`（samples 第 3–5 格 `invalidated`、第 9 格 `lost` → 兩個區間 `{3,5,"invalidated"}`、`{9,9,"lost"}`，回合外的空格不列）。
- [ ] **Step 2：確認失敗**。
- [ ] **Step 3：實作**。
- [ ] **Step 4：確認通過**　`.venv/bin/python -m pytest -q tests/test_review_ui.py`。
- [ ] **Step 5：審查後 commit**　訊息 "Explain handheld camera outcomes in review"。

---

### Task 8：驗證腳本、實測與門檻校正

**Files：**
- Create: `scripts/eval_dynamic_calibration.py`、`tests/test_eval_dynamic_calibration.py`
- Modify（依實測結果）：`config/camera_mode.yaml`
- 記錄：`.superpowers/sdd/2026-11-12-dynamic-calibration/progress.md`（ledger，第一行 `# SDD ledger — plan: docs/superpowers/plans/2026-11-12-dynamic-calibration.md`）

**Interfaces：**
- Consumes：Task 2、3、5 的公開介面。
- Produces：`scripts/eval_dynamic_calibration.py` 三個子命令，輸出 JSON 到 `--out`（沿用 `eval_stadium_finder.py` 的 `ensure_inside`、續跑與 SHA 核對寫法）：
  - `modes <資料夾...>`：每支影片 `CameraMode`，依批次彙總手持比例。
  - `refind-accuracy --dataset runs/stadium-kp-20261112 --manifest data/detector-hard-cases-20261030/stadium-keypoints-20261111/manifest.json`：以圖片 SHA 對回 manifest 的批次；對每張圖 `locate` 一次，與標註共平面點（索引 0–5、7）求得的單應比較，報告各批最差區域誤差（中位、p90、最大），train／val 分開，20261101 單獨列出。
  - `geometry <影片...>`：不跑偵測器，只解碼並依模式跑 `DynamicCalibration` 或 `HandheldCalibration`（`set_foreground([])`）；報告每支的有幾何格數比例、status 分布、每次重找的 `disagreement_px`（跟隨漂移）、`invalidated` 格數、處理時間。

- [ ] **Step 1：寫失敗測試**（`tmp_path` 合成影片與假定位器）：`test_modes_summarises_per_batch`、`test_geometry_reports_coverage_and_refind_drift`、`test_outputs_stay_inside_out`。
- [ ] **Step 2：確認失敗**。
- [ ] **Step 3：實作**。
- [ ] **Step 4：確認通過**，審查後 commit（訊息 "Measure camera modes, refind accuracy and geometry coverage"）。
- [ ] **Step 5：實測**（先 `nvidia-smi`；長時間指令放背景）：
  1. `modes` 跑 5 個批次共 434 支：預期 20261101 大多為手持、20261108 大多為固定；誤判的影片逐支抽看畫面。
  2. `refind-accuracy`：20261101 的 35 張圖與其他批次對照。
  3. `geometry` 跑 20261101 全部 61 支＋20261108 前 10 支當對照：記錄覆蓋率、漂移分布、`invalidated` 比例、每支耗時。
  4. 固定影片回歸：以 `--stadium-keypoints --dynamic-calibration config/dynamic_calibration.yaml --two-bey-tracking config/two_bey_tracking.yaml` 重判 Task 4 的 20 回合（作法照 `.superpowers/sdd/launch-phase-deadline-20261112/rejudge-selected.py`），判決欄位必須與目前完全相同；列出處理時間增加量。
- [ ] **Step 6：校正門檻**：依 1–3 的數據調整 `handheld_px_per_s`、`refind_s`、`confirm_px`、`max_unchecked_s`；每項寫一行 ledger `Ruling: <決定> — <理由> — <若錯的代價>`，重跑受影響的量測，commit `config/camera_mode.yaml`（訊息 "Calibrate camera mode thresholds on 434 videos"）。
- [ ] **Step 7：收尾**：`graphify update .`；把全部結果整理成給使用者的摘要（找到率、覆蓋率、誤判、耗時、未解決項目）。
