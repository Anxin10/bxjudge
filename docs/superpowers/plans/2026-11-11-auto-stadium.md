# 自動找場地 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 對每支影片，依綠色軌道自動找出場地，輸出與人工校準同格式的 `Calibration`；不可信時輸出 recalibrate，袋區在畫面外時標記出來。

**Architecture:** 從一組人工範本建立標準場地模型，內容是軌道點集加 8 個區域。每支影片挑遮擋最少的幾張畫面，以自適應綠色門檻抓出軌道，先粗搜旋轉、縮放、位置，再以投影變換精修套上模型。最後用既有的 `warp_calibration` 換算區域。X／B 改以場地左右判定。接入主流程放在最後一個任務。

**Tech Stack:** Python 3.12（uv）、OpenCV、numpy、pytest。沿用 `autocalib.py`、`calibration.py`、`calibration_store.py`，不新增依賴。

**Spec:** `docs/superpowers/specs/2026-11-11-auto-stadium-design.md`

## Global Constraints

- 分支 `feat/local-judge`，不新建分支或工作樹，不 push、不 merge。
- 工作樹裡有別人未提交的修改，不得動：Rust 擱置、HUD、`config/vlm.yaml`。
- Codex 同時在這個工作樹執行 Task 2，會修改 `manifest.py`、`config.py`、`thresholds.yaml`、`launch_check.py`、`finish.py`、`landing.py` 與 `pipeline.py`。Task 1–6 只新增檔案，或修改 `calibration.py`。Task 7、8 會碰到 Codex 的檔案，必須等 Codex 的 Task 2 提交後才開始。
- 提交時用 `git commit -o <自己的檔案>`，只提交列出的路徑，不碰共用 index 裡別人 stage 的內容。
- 原始影片、`calib/`、人工真值與 811 val／test 一律唯讀。新輸出另存。
- 門檻全部放在 `config/stadium_finder.yaml`。證據不足時回傳 `None`，由呼叫端標成 recalibrate，不使用後備值。
- X／B：貓耳（軌道近端 Xtreme 缺口兩側的凸起）朝向觀看者時，場地左邊為 X、右邊為 B（使用者 2026-11-11 以圖確認）。
- 邊界誤差門檻沿用 `autocalib.yaml` 的 `max_boundary_error_px: 8`，單位是長邊 1280 的推論像素。
- 程式修改先寫失敗的測試，再寫實作；全套測試通過後交 Codex 唯讀審查，親自核實每條發現後再提交。
- Commit 結尾：`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`。

## Review Focus

1. **綠色手套或其他綠色物體靠近軌道**（11/01 `113324`）：不得把區域套歪。分數不夠就回傳 `None`。→ Task 3 測試 `test_green_glove_does_not_shift_fit`。
2. **前幾秒的畫面都被手擋住**：要往後找畫面，找不到才回傳 `None`。→ Task 4 測試 `test_skips_occluded_frames`。
3. **鏡頭旋轉 90° 以上**（11/01 `115144`、`120734`）：方向必須由貓耳決定，X／B 不能對調。→ Task 3 測試 `test_rotated_and_mirrored_orientation`。
4. **袋區在畫面外**（11/08）：仍輸出區域，並標出可見比例。→ Task 5 測試 `test_zone_visibility_out_of_frame`。
5. **沒有場地的畫面**（只拍到腳，或 `114513` 那種全黑畫面）：回傳 `None`，不崩潰。→ Task 4 測試 `test_no_stadium_returns_none`。

---

### Task 1：設定檔與標準場地模型

> 建模會用到 Task 2 的 `rail_mask`，所以 Task 1 與 Task 2 合併成一次提交：先寫 Task 1 Step 1 與 Task 2 的測試，接著實作設定檔與 `rail_mask`，最後完成建模。

**Files:**
- Create: `config/stadium_finder.yaml`、`src/bxjudge/stadium_finder.py`（模型部分）、`scripts/build_stadium_model.py`、`config/stadium_model.json`（腳本產生）
- Test: `tests/test_stadium_finder.py`

**Interfaces:**
- Produces:
  - `StadiumFinderConfig`（frozen dataclass，所有 YAML 鍵）
  - `load_stadium_finder_config(path: Path = DEFAULT_STADIUM_FINDER_CONFIG) -> StadiumFinderConfig`
  - `StadiumModel(rail: np.ndarray  # (N,2) template px, calibration: Calibration, source_sha256: str, image_size: tuple[int, int])`
  - `load_stadium_model(path: Path = DEFAULT_STADIUM_MODEL) -> StadiumModel`
  - `build_stadium_model(template: Calibration, image: np.ndarray, cfg: StadiumFinderConfig) -> StadiumModel`

- [ ] **Step 1：寫失敗的測試**
  - `test_config_rejects_missing_or_nonpositive_keys`：缺鍵、負值或 NaN 都拋 `ValueError`。
  - `test_build_model_from_synthetic_rail`：用合成影像（灰底、亮綠色的「橢圓 + 近端凹口」軌道），模型點數 ≥ `cfg.model_points`，且所有點都在綠色像素 2 px 內。
  - `test_model_roundtrip`：存檔再讀回，點集與 8 個區域完全相同，`source_sha256` 保留。
- [ ] **Step 2：執行測試**，確認失敗。指令：`uv run pytest tests/test_stadium_finder.py -q`
- [ ] **Step 3：實作**
  - YAML 至少包含：
    - `model_template: 20261014_202713`
    - `model_points: 400`
    - `hue_search: [30, 90]`
    - `min_saturation: 60`、`min_value: 60`
    - `component_min_elongation`
    - `sample_window_s: 6.0`、`sample_step_s: 0.25`、`candidate_frames: 5`
    - `coarse_rotation_step_deg: 5`、`coarse_scales: [0.6, 1.6]`
    - `refine_iterations: 30`、`inlier_px: 6`
    - `min_coverage: 0.5`、`min_precision: 0.5`
    - `agreement_px: 8`
    - `zone_visible_min: 0.8`
  - 範本影像的軌道萃取共用 Task 2 的 `rail_mask`。建模時用範本的 `battle_area` 外擴一點作為 ROI，排除場地外的綠色。
  - 沿軌道骨架等距取樣 `model_points` 點。
  - `scripts/build_stadium_model.py --template calib/20261014_202713.json --out config/stadium_model.json`，只讀 `calib/`。
- [ ] **Step 4：執行測試**，確認通過。執行建模腳本，並把軌道點疊在範本圖上，存一張預覽到 `.superpowers/sdd/auto-stadium-20261111/model-preview.jpg` 供人工查看。
- [ ] **Step 5：提交**：`git commit -o config/stadium_finder.yaml config/stadium_model.json src/bxjudge/stadium_finder.py scripts/build_stadium_model.py tests/test_stadium_finder.py`

### Task 2：自適應軌道萃取

**Files:** Modify `src/bxjudge/stadium_finder.py`；Test `tests/test_stadium_finder.py`

**Interfaces:**
- Produces: `rail_mask(frame: np.ndarray, cfg: StadiumFinderConfig, roi: np.ndarray | None = None) -> np.ndarray`，回傳 uint8 0/255 的遮罩，尺寸與畫面相同。

- [ ] 測試先寫：
  - `test_rail_mask_adapts_to_hue_shift`：合成軌道色相在 45 與 70 時都能抓到 ≥ 90% 的軌道像素。
  - `test_rail_mask_drops_compact_blobs`：同色的實心橢圓塊（模擬手套）被濾掉，細長的軌道保留。
  - `test_rail_mask_empty_on_gray`：無綠色的畫面回傳全 0。
- [ ] 實作：
  1. 在 `hue_search` 範圍內，對飽和度與亮度達門檻的像素做色相直方圖，取主峰 ±寬度作為本張畫面的門檻。
  2. 開運算去雜點。
  3. 連通元件依細長度（骨架長度／面積）過濾。
- [ ] 測試通過後提交，作法同 Task 1。

### Task 3：模型套合

**Files:** Modify `src/bxjudge/stadium_finder.py`；Test `tests/test_stadium_finder.py`

**Interfaces:**
- Produces:
  - `StadiumFit(homography: tuple[tuple[float, float, float], ...], coverage: float, precision: float, mean_error_px: float)`
  - `fit_stadium(mask: np.ndarray, model: StadiumModel, cfg: StadiumFinderConfig) -> StadiumFit | None`
  - `homography` 為模型像素到畫面像素；行列式為正，不允許鏡像。

- [ ] 測試先寫：
  - `test_fit_recovers_known_homography`：把合成軌道以已知的旋轉、縮放、位移與輕微透視投影到畫面上，換算後的區域頂點誤差 < 2 px。
  - `test_rotated_and_mirrored_orientation`：旋轉 0、90、180、270° 都能正確還原。左右鏡像的軌道必須回傳 `None` 或得到正確方向，不能把 `over_left` 換到右側。
  - `test_partial_rail_still_fits`：遮掉 40% 的軌道仍能套合，誤差 < 4 px。
  - `test_green_glove_does_not_shift_fit`：軌道旁加一塊細長綠色雜訊，誤差仍 < 4 px，否則回傳 `None`。
  - `test_low_coverage_returns_none`：只剩 20% 軌道時回傳 `None`。
- [ ] 實作：
  1. 粗搜：在縮小的距離轉換圖上，對旋轉（步進 `coarse_rotation_step_deg`，全 360°）、縮放與位移做 chamfer 比對，取前幾名候選。
  2. 精修：反覆以最近點對應加 `cv2.findHomography(RANSAC)` 更新，最多 `refine_iterations` 次。
  3. 計算 `coverage`（模型點在證據 `inlier_px` 內的比例）與 `precision`（證據點被模型解釋的比例）。任一低於門檻，或行列式 ≤ 0，就回傳 `None`。
- [ ] 測試通過後提交。

### Task 4：挑畫面與整支影片的套合

**Files:** Modify `src/bxjudge/stadium_finder.py`；Test `tests/test_stadium_finder.py`

**Interfaces:**
- Consumes: `bxjudge.video` 的解碼工具。實作前先讀它，沿用既有的 ffmpeg 讀取方式，不自行開檔。
- Produces:
  - `FoundStadium(calibration: Calibration, fit: StadiumFit, frame_index: int, frame: np.ndarray, zone_visibility: dict[str, float], agreeing_frames: int)`
  - `find_stadium(video: Path, model: StadiumModel, cfg: StadiumFinderConfig) -> FoundStadium | None`

- [ ] 測試先寫，用合成影片（`tests/helpers.make_video` 或同等工具）：
  - `test_skips_occluded_frames`：前 2 秒軌道被大塊遮住，之後清楚，結果取自清楚的畫面。
  - `test_no_stadium_returns_none`：全黑影片與無綠色影片都回傳 `None`，不拋例外。
  - `test_disagreeing_frames_return_none`：候選畫面之間的套合互相矛盾（區域差 > `agreement_px`）時回傳 `None`。
- [ ] 實作：
  1. 在 `sample_window_s` 內每 `sample_step_s` 秒取一格，依軌道像素量排序，取前 `candidate_frames` 格分別套合。
  2. 至少兩格的區域差距在 `agreement_px` 內才算數，取分數最高的一格。
  3. 用 `autocalib.warp_calibration` 產生 `Calibration`，名稱為 `found-<video stem>`。它只吃 `Alignment`，所以要從 `StadiumFit.homography` 組一個 `Alignment`：
     - `score = min(coverage, precision)`；
     - `shift_frac`、`rotation_deg`、`scale`、`perspective` 由 `autocalib._decompose` 算出。
- [ ] 測試通過後提交。

### Task 5：袋區可見度

**Files:** Modify `src/bxjudge/stadium_finder.py`；Test `tests/test_stadium_finder.py`

**Interfaces:**
- Produces: `zone_visibility(c: Calibration) -> dict[str, float]`，計算 `stadium`、`battle_area`、`over_left`、`over_right`、`xtreme` 各自落在畫面內的面積比例。

- [ ] 測試先寫：
  - `test_zone_visibility_out_of_frame`：把 `xtreme` 移到一半在畫面外，比例約 0.5（誤差 0.02 內），其他區域為 1.0。
  - `test_zone_visibility_degenerate`：面積為 0 的區域回傳 0.0，不得除以 0。
- [ ] 實作後，在 `find_stadium` 填入此欄位。測試通過後提交。

### Task 6：5 區人工真值工具與評估

**Files:**
- Modify: `src/bxjudge/calibration.py`（`interactive_calibrate` 加上 `zones` 參數，預設為原本的 8 區，原行為不變）
- Create: `scripts/stadium_truth.py`、`scripts/eval_stadium_finder.py`
- Test: `tests/test_calibration.py`（新增一個測試）、`tests/test_stadium_finder.py`

**Interfaces:**
- Produces:
  - `TRUTH_ZONES = ("stadium", "battle_area", "over_left", "over_right", "xtreme")`
  - `zone_boundary_errors(found: Calibration, truth: dict[str, Polygon], long_edge: int) -> dict[str, float]`，沿用 `autocalib._boundary_distance` 的距離定義。

- [ ] 測試先寫：
  - `test_polygon_collector_with_truth_zones`：`PolygonCollector(zones=...)` 依序收完 5 區後 `finished` 為真。
  - `test_zone_boundary_errors_identical_is_zero`：兩份相同的區域，誤差全為 0。
  - `test_zone_boundary_errors_shifted`：平移 10 px，在長邊 1280 換算下誤差約 10 × 1280 ÷ 1920。
- [ ] `scripts/stadium_truth.py <video> --at <秒> --out data/stadium-truth-20261111/`：開視窗讓使用者點 5 區，存 JSON 與參考畫面。輸出目錄不得在 `calib/` 內，否則拒絕。
- [ ] 挑出約 15 支要點真值的影片，涵蓋 10/24 黑白兩種箱子、11/01 大角度、11/08 袋區被切。清單寫進 `.superpowers/sdd/auto-stadium-20261111/truth-list.md`，交給使用者。
- [ ] `scripts/eval_stadium_finder.py`：對全部 434 支影片唯讀執行 `find_stadium`，輸出：
  - 每批成功率；
  - 失敗原因；
  - 袋區可見度分布；
  - 每支的疊圖預覽；
  - 有真值的影片，各區邊界誤差與 PASS／FAIL（門檻 8 px）；
  - 舊影片與現行範本或範本對位結果的誤差。
  
  輸出到 `.superpowers/sdd/auto-stadium-20261111/eval/`。
- [ ] 測試通過後提交。完成後停下，交 Codex 審查，並把評估結果給使用者看。

### Task 7：以場地左右判定 X／B（等 Codex Task 2 提交後）

**Files:**
- Modify: `src/bxjudge/launch_check.py`（`assign_sides` 的後備判定）、`src/bxjudge/vlm.py`（提示文字）、`src/bxjudge/geometry.py`（新增函式；擱置的 Rust 修改不得被 stage）
- Test: `tests/test_launch_check.py`、`tests/test_geometry.py`

**Interfaces:**
- Produces: `stadium_side(p: Point, c: Calibration) -> Side`。以 `battle_area` 形心為原點、`over_left → over_right` 形心向量為右方，內積 < 0 為 X、≥ 0 為 B。

- [ ] 實作前先列出所有以影像 x 座標分左右的地方。目前已知 `launch_check.py:53` 的 `mid_x`，其他用 `rg` 全找出來，寫進 ledger。
- [ ] 測試先寫：
  - `test_stadium_side_matches_image_side_for_upright`：正擺的範本與 `mid_x` 判定結果相同。
  - `test_stadium_side_follows_rotation`：把範本旋轉 90° 與 180° 後，原本左側的點仍判為 X。
- [ ] 回歸：以修改前後的程式重判 20 回合真值，判決欄位逐欄相同；有差異就停下回報。
- [ ] `vlm.py` 的提示文字從「畫面左方為 X 方」改為「場地左側（貓耳朝向鏡頭時的左邊）為 X 方」。
- [ ] 只 stage 本任務的 hunk（`git add -p`），全套測試通過後提交。

### Task 8：接入主流程（等 Codex Task 2 提交後）

**Files:**
- Modify: `src/bxjudge/calibration_store.py`（`select_snapshot`）、`src/bxjudge/pipeline.py`（袋區可見度 note）
- Test: `tests/test_calibration_store.py`、`tests/test_pipeline.py`

**Interfaces:**
- Consumes: `find_stadium`、`zone_visibility`。
- Produces:
  - `Snapshot.kind` 新增 `"found"`，`_checked_meta` 接受它。
  - `select_snapshot` 的順序改為：人工範本 → 範本對位 → `find_stadium` → `None`。

- [ ] 測試先寫：
  - `test_select_snapshot_falls_back_to_found`：範本對位失敗、找場地成功時回傳 `kind == "found"`。
  - `test_found_snapshot_veto`：veto 機制對 `found` 一樣有效。
  - `test_pocket_out_of_frame_forces_review`：袋區可見度低於 `zone_visible_min` 時，回合帶 `zone_out_of_frame@xtreme` 等 note，且涉及該袋區的終結會 `needs_review`。
- [ ] 回歸：20 回合真值逐欄相同。舊影片原本就有人工範本或範本對位，不會走到 `found`。
- [ ] 新影片 smoke：每批抽 3 支跑完整流程，只確認能跑完、區域疊圖正確，不評準確率。
- [ ] 全套測試、Codex 審查後提交。
