# Beyblade X 本地裁判 Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在使用者的 RTX 3090 / WSL2 上建立本地 AI 裁判，逐回合判斷 Beyblade X 比賽影片的發射檢查（觸碰盤面、落入戰鬥區、空判）與勝負（Spin / Over / Burst / Xtreme），依設定檔規則計分，低信心者交人工覆核。

**Architecture:** 影片經 rclone 唯讀同步進 WSL → ffmpeg 解碼 → 自訓練偵測模型（RF-DETR 或 YOLO26）+ ByteTrack 產生軌跡 → 純 Python 的事件邏輯（落地、觸碰、空碰、終結）以校準多邊形判定 → YAML 規則引擎計分 → 低信心片段交 Ollama VLM 給參考意見 → Streamlit 覆核介面 → 修正紀錄回饋評估與調參。判定邏輯全部吃「軌跡 + 盤面位移序列」，與模型解耦，可用合成資料做單元測試。

**Tech Stack:** WSL2 Ubuntu、Python 3.12（uv 管理）、PyTorch 最新穩定版（CUDA 13.x wheel）、ffmpeg、rclone、OpenCV、numpy、PyYAML、supervision（ByteTrack）、rfdetr、ultralytics（YOLO26）、SAM 3.1、Label Studio、Ollama（qwen3.8:27b / qwen3-vl:8b）、Streamlit、pytest。

**Spec:** `docs/specs/2026-10-21-beyblade-x-local-judge-design.md`（實作時複製到 repo 的 `docs/spec.md`，以 repo 內版本為準）

## Global Constraints

- 所有程式、模型、工具在 **WSL2** 內執行；repo 位於 `~/bxjudge`。從 Windows 端執行指令：`wsl.exe -- bash -lc 'cd ~/bxjudge && <cmd>'`。
- 影片同步到 WSL 檔案系統 `~/bx/videos/<批次>`；**不得**從 `/mnt/c`、`/mnt/g` 直接讀影片。
- 雲端硬碟只讀：rclone remote 使用 `drive.readonly` scope；程式不得寫入、移動、刪除任何來源影片。
- 不上傳影片或畫面到任何雲端服務（Ollama 只用本地模型，不用 `-cloud` 標籤）。
- 畫面左方 = `"X"`，右方 = `"B"`。
- 計分（`config/rules.yaml`）：spin 1、burst 2、over 2、xtreme 3；先到 `win_points: 4` 勝。規則與門檻**只能**放在 `config/*.yaml`，程式碼中不得寫死數值。
- 不使用音軌；不判斷搶拍。
- 觸碰盤面唯一依據：手或發射器造成的盤面晃動；與陀螺落地 / 撞壁 / 互撞同時（±`impact_coincide_frames`）的晃動不算觸碰。
- 空判：在**兩顆陀螺都尚未落地**的期間（早於較早的落地時間點）發生碰撞。
- 發射時間窗與空碰檢查期間**不降幀**（保留 60fps 每一格）；推論解析度長邊 1280。
- VLM 答案**永不覆蓋** `finish` / `winner`，只寫入 `vlm_note`。
- 低信心門檻 `review_confidence: 0.7`；低於門檻或任一欄位為 unknown / maybe → `needs_review = True`。
- 成功標準（spec §11）：未標覆核的回合終結方式 + 得分方正確率 ≥ 95%；觸碰召回 ≥ 90%；空判召回 ≥ 90%；需覆核比例 ≤ 30%；處理時間 ≤ 影片長度。
- Git commit 訊息結尾加上 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`。

## Review Focus

1. **影片裡偵測不到任何回合**（鏡頭被碰歪、非比賽片段、陀螺從未被偵測到）→ 不可崩潰，輸出一筆 `needs_review`、原因 `no_round`。測試加在 Task 10 與 Task 13。
2. **兩顆陀螺碰撞時追蹤 ID 互換** → 方別判反卻高信心。期望：框重疊連續 ≥ `id_swap_overlap_frames` 格即降信心並標 `id_swap_risk`。測試加在 Task 9。
3. **鏡頭位置與校準不符**（換場地、支架被移動）→ 不可用舊多邊形硬判。期望：相似度低於 `calibration_min_similarity` 時輸出 `recalibrate` 並跳過該影片。測試加在 Task 4 與 Task 13。
4. **影片損毀 / WSL 下 NVDEC 解碼失敗** → CPU 重試；仍失敗則記錄 `decode_error`，批次繼續處理其他影片。測試加在 Task 5 與 Task 13。
5. **重跑批次或 rclone 新增檔案** → 不重複處理未變更影片、不產生重複列；**已覆核的列永遠不被覆寫**。測試加在 Task 5 與 Task 13。

---

## File Structure

```
~/bxjudge/
  pyproject.toml                 # 套件 bxjudge，pytest markers: gpu, slow
  config/rules.yaml              # 規則（spec §7）
  config/thresholds.yaml         # 所有門檻與時間窗
  scripts/setup_wsl.sh           # 安裝 ffmpeg / rclone / uv / Python / torch
  scripts/sync_videos.sh         # rclone 唯讀同步
  docs/spec.md                   # 設計文件副本
  src/bxjudge/
    types.py                     # 共用資料型別
    config.py                    # 載入 thresholds.yaml
    geometry.py                  # 多邊形、框運算、軌跡速度
    rules.py                     # 規則引擎：單回合判定、分場累計
    calibration.py               # 校準存取、相似度、互動標註
    manifest.py                  # 掃描影片、ffprobe、已處理紀錄
    video.py                     # ffmpeg 解碼（NVDEC → CPU 後備）
    stadium_motion.py            # 盤面位移序列（ECC）
    landing.py                   # 落地時間點
    launch_check.py              # 方別、觸碰、落入、空碰
    finish.py                    # 停轉、終結判定、ID 互換風險
    segment.py                   # 回合切分
    detect.py                    # Detector 介面、RF-DETR / YOLO26 adapter
    track.py                     # ByteTrack → Track
    train.py                     # 訓練與模型比較
    labeling/sample_frames.py    # 事件導向抽格
    labeling/sam3_prelabel.py    # SAM 3.1 預標 → Label Studio JSON
    labeling/export.py           # Label Studio → COCO（依影片切分）
    pipeline.py                  # 單支影片 → RoundResult
    output.py                    # rounds.json / rounds.csv / matches.csv，合併保護覆核列
    vlm.py                       # Ollama 覆核
    review_app.py                # Streamlit 覆核介面
    corrections.py               # 修正紀錄存取與套用
    evaluate.py                  # 指標與門檻推導
    cli.py                       # bxjudge 指令
  tests/
    helpers.py                   # make_track(), make_calibration(), make_video()
    test_*.py
```

---

### Task 1: WSL 環境與專案骨架

**Files:**
- Create: `scripts/setup_wsl.sh`, `pyproject.toml`, `.gitignore`, `src/bxjudge/__init__.py`, `docs/spec.md`, `tests/test_env.py`

**Interfaces:**
- Produces: 可 `uv run pytest` 的套件 `bxjudge`；pytest markers `gpu`、`slow`；指令入口 `bxjudge = "bxjudge.cli:main"`。

- [ ] **Step 1: 建立 repo 並複製設計文件**

```bash
mkdir -p ~/bxjudge && cd ~/bxjudge && git init
cp "/mnt/c/Users/<user>/VSC/Edge/docs/specs/2026-10-21-beyblade-x-local-judge-design.md" docs/spec.md
```

- [ ] **Step 2: 撰寫 `scripts/setup_wsl.sh`**

依序：`apt install ffmpeg rclone`；安裝 uv；`uv python install 3.12`；`uv sync`；從 pytorch.org「Get Started」頁目前**穩定版**的 CUDA 13.x index（cu130 或 cu132，以頁面為準）安裝 torch / torchvision，並把 index URL 寫進 `pyproject.toml` 的 `[tool.uv.sources]`。`.gitignore` 排除 `data/`、`runs/`、`weights/`、`*.mp4`、`.venv/`。

- [ ] **Step 3: 寫環境測試 `tests/test_env.py`**

```python
import shutil, subprocess, pytest

def test_ffmpeg_present():
    assert shutil.which("ffmpeg") and shutil.which("ffprobe")

@pytest.mark.gpu
def test_torch_sees_3090():
    import torch
    assert torch.cuda.is_available()
    assert "3090" in torch.cuda.get_device_name(0)

@pytest.mark.gpu
def test_ffmpeg_lists_cuda_hwaccel():
    out = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True, text=True).stdout
    assert "cuda" in out
```

- [ ] **Step 4: 執行 `bash scripts/setup_wsl.sh && uv run pytest tests/test_env.py -v`**

Expected: 全部 PASS。若只有 `test_ffmpeg_lists_cuda_hwaccel` 失敗，在 `config/thresholds.yaml` 設 `decode_hwaccel: cpu` 並把該測試標 `xfail`（spec 允許 CPU 後備），於 commit 訊息註明。

- [ ] **Step 5: Commit** — `git add -A && git commit -m "chore: WSL env and project scaffold"`

---

### Task 2: 共用型別、設定與幾何

**Files:**
- Create: `src/bxjudge/types.py`, `src/bxjudge/config.py`, `src/bxjudge/geometry.py`, `config/thresholds.yaml`, `tests/helpers.py`, `tests/test_geometry.py`

**Interfaces:**
- Produces（`types.py`，全部 `@dataclass`）：
  - `Side = Literal["X","B"]`、`Finish = Literal["spin","over","burst","xtreme"]`、`TouchState = Literal["yes","no","unknown"]`、`AirState = Literal["yes","no","maybe"]`、`Point = tuple[float,float]`、`Box = tuple[float,float,float,float]`（x1,y1,x2,y2）、`Polygon = list[Point]`
  - `Detection(frame: int, cls: str, box: Box, score: float)`；`cls ∈ {"bey","bey_part","launcher"}`
  - `Track(track_id: int, cls: str, dets: list[Detection])`，方法 `first_frame() -> int`、`last_frame() -> int`、`box_at(frame) -> Box | None`、`center_at(frame) -> Point | None`、`centers() -> dict[int, Point]`
  - `Zone = Literal["battle_area","lid","over_left","over_right","xtreme","stadium_other","outside"]`
  - `Calibration(name: str, frame_size: tuple[int,int], stadium, lid, battle_area, over_left, over_right, xtreme, launch_left, launch_right: Polygon, ref_frame: str)`（`battle_area` = 下盤戰鬥區，陀螺正常跑的盤底；`lid` = 上盤，透明蓋子；`stadium` 外框包含蓋子）
  - `LaunchCheck(touch: dict[Side,TouchState], landed: dict[Side,bool], landing_frame: dict[Side,int|None], landing_zone: dict[Side, Zone | None], air_collision: AirState, notes: list[str])`
  - `RoundResult(video: str, round: int, t_launch: float, t_end: float, launch: LaunchCheck | None, finish: Finish | None, winner: Side | None, points: int, confidence: float, needs_review: bool, review_reasons: list[str], vlm_note: str | None = None, reviewed: bool = False, reviewer_fix: dict | None = None)`
- Produces（`config.py`）：`Thresholds`（dataclass，欄位與 yaml 鍵同名）、`load_thresholds(path="config/thresholds.yaml") -> Thresholds`
- Produces（`geometry.py`）：`point_in_polygon(p: Point, poly: Polygon) -> bool`、`zone_of(p: Point, c: Calibration) -> Zone`（檢查順序：xtreme → over_left → over_right → battle_area → lid → stadium_other → outside；蓋子與戰鬥區在畫面上重疊時以戰鬥區為準）、`box_center(b: Box) -> Point`、`boxes_touch(a: Box, b: Box, margin: float = 0.0) -> bool`、`polygon_mask(poly: Polygon, size: tuple[int,int]) -> np.ndarray`、`velocities(centers: dict[int, Point]) -> dict[int, tuple[float,float]]`（相鄰格差分）、`is_abrupt_change(v_before, v_after, angle_deg: float, speed_ratio: float) -> bool`
- Produces（`tests/helpers.py`）：`make_track(track_id, cls, centers: dict[int, Point], size=20.0, score=0.9) -> Track`、`make_calibration() -> Calibration`（640×360 的矩形示意版：stadium=(40,20)-(600,340)、lid=(60,25)-(580,120)、battle_area=(120,60)-(520,280)、over_left=(40,200)-(110,300)、over_right=(530,200)-(600,300)、xtreme=(280,290)-(360,340)、launch_left=(40,0)-(320,120)、launch_right=(320,0)-(600,120)）

`config/thresholds.yaml` 初始值（spec §13 允許實作者定初值，Task 16 再以資料校正）：

```yaml
fps_default: 60
infer_long_edge: 1280
decode_hwaccel: cuda            # cuda | cpu
launch_pre_s: 0.5
launch_post_s: 0.5
shake_px: 2.0
impact_coincide_frames: 3
air_collision_frames: 2
abrupt_angle_deg: 45
abrupt_speed_ratio: 0.5
settle_frames: 6
fall_speed_px: 6.0
rim_margin_px: 15
still_px: 1.5
still_activity: 4.0
still_frames: 30
simultaneous_stop_frames: 3
burst_part_window_s: 0.3
id_swap_overlap_frames: 3
round_gap_s: 2.0
calibration_min_similarity: 0.8
review_confidence: 0.7
```

- [ ] **Step 1: 寫失敗測試 `tests/test_geometry.py`**

```python
from bxjudge.geometry import point_in_polygon, zone_of, boxes_touch, is_abrupt_change, velocities
from tests.helpers import make_calibration

def test_point_in_polygon():
    c = make_calibration()
    assert point_in_polygon((300, 150), c.battle_area)
    assert not point_in_polygon((50, 10), c.battle_area)

def test_zone_of():
    c = make_calibration()
    assert zone_of((300, 200), c) == "battle_area"
    assert zone_of((300, 100), c) == "battle_area"      # 與蓋子重疊處以戰鬥區為準
    assert zone_of((80, 60), c) == "lid"
    assert zone_of((320, 320), c) == "xtreme"
    assert zone_of((70, 250), c) == "over_left" and zone_of((560, 250), c) == "over_right"
    assert zone_of((150, 320), c) == "stadium_other" and zone_of((5, 5), c) == "outside"

def test_boxes_touch_with_margin():
    assert boxes_touch((0,0,10,10), (10,0,20,10))
    assert not boxes_touch((0,0,10,10), (15,0,25,10))
    assert boxes_touch((0,0,10,10), (15,0,25,10), margin=5)

def test_abrupt_change():
    assert is_abrupt_change((10,0), (0,10), angle_deg=45, speed_ratio=0.5)      # 90° 轉向
    assert is_abrupt_change((10,0), (4,0), angle_deg=45, speed_ratio=0.5)       # 速度降 60%
    assert not is_abrupt_change((10,0), (9,1), angle_deg=45, speed_ratio=0.5)

def test_velocities():
    v = velocities({0:(0,0), 1:(3,4), 2:(6,8)})
    assert v[1] == (3,4) and v[2] == (3,4)
```

- [ ] **Step 2: `uv run pytest tests/test_geometry.py -v`** — Expected: FAIL（模組不存在）
- [ ] **Step 3: 實作 `types.py`、`config.py`、`geometry.py`、`tests/helpers.py`、`config/thresholds.yaml`**（`point_in_polygon` 用 `cv2.pointPolygonTest`）
- [ ] **Step 4: `uv run pytest tests/test_geometry.py -v`** — Expected: PASS
- [ ] **Step 5: Commit** — `feat: core types, thresholds config, geometry`

---

### Task 3: 規則引擎

**Files:**
- Create: `config/rules.yaml`, `src/bxjudge/rules.py`, `tests/test_rules.py`

**Interfaces:**
- Consumes: `RoundResult`, `LaunchCheck`, `Side`（Task 2）
- Produces:
  - `Rules`（dataclass）、`load_rules(path="config/rules.yaml") -> Rules`
  - `finish_points(rules: Rules, finish: Finish) -> int`
  - `RoundOutcome(action: Literal["score","redo"], scorer: Side | None, points: int, reasons: list[str])`
  - `decide_round(r: RoundResult, rules: Rules, misses: dict[Side,int]) -> RoundOutcome`（會就地累加 `misses`）
  - `Match(rounds: list[tuple[RoundResult, RoundOutcome]], score: dict[Side,int], winner: Side | None)`
  - `group_matches(results: list[RoundResult], rules: Rules) -> list[Match]`（依 `t_launch` 及影片順序排序後累計；任一方 ≥ `win_points` 即結束該場，下一回合開新場；每場重置 `misses`）

`config/rules.yaml` 照 spec §7 原文，另加來源註解：

```yaml
rules_version: "takara-tomy-bx-2025-reg8"   # TODO(使用者)：以官方最新規則 PDF 核對
win_points: 4
finish_points: {spin: 1, burst: 2, over: 2, xtreme: 3}
launch_check:
  stadium_touch: redo
  bey_not_in_battle_area: redo
  mid_air_collision: redo
  repeat_miss_penalty: none        # none | opponent_1pt_on_2nd_miss
draw:
  action: redo
```

判定順序（`decide_round`）：發射失誤（touch=="yes"、landed 為 False、air_collision=="yes"）→ 依 `launch_check` 設定 redo，並把失誤方記入 `misses`（空碰不記任何一方）；若 `repeat_miss_penalty == "opponent_1pt_on_2nd_miss"` 且該方 misses 達 2 → `score` 對手 1 分。無發射失誤時：`finish is None` → `draw.action`；否則 `score`，`scorer = winner`、`points = finish_points`。`touch == "unknown"` 與 `air_collision == "maybe"` **不**觸發 redo（只影響 needs_review）。

- [ ] **Step 1: 寫失敗測試 `tests/test_rules.py`**（在 `tests/helpers.py` 新增 `rr(finish, winner, touch=None, landed=None, air="no") -> RoundResult`，未指定的 touch / landed 預設雙方 `"no"` / `True`；後續 Task 的 `res(...)`、`rev(...)`、`read_rounds(...)` 等測試 helper 同樣放在 `tests/helpers.py`）

```python
def test_finish_points(rules):
    assert [finish_points(rules, f) for f in ("spin","over","burst","xtreme")] == [1,2,2,3]

def test_xtreme_scores_three(rules):
    o = decide_round(rr("xtreme", "X"), rules, {"X":0,"B":0})
    assert (o.action, o.scorer, o.points) == ("score", "X", 3)

def test_touch_yes_redo_and_counts_miss(rules):
    m = {"X":0,"B":0}
    o = decide_round(rr("spin", "B", touch={"X":"yes","B":"no"}), rules, m)
    assert o.action == "redo" and m == {"X":1,"B":0}

def test_not_landed_redo(rules):
    assert decide_round(rr("spin","X", landed={"X":True,"B":False}), rules, {"X":0,"B":0}).action == "redo"

def test_air_collision_yes_redo_no_miss(rules):
    m = {"X":0,"B":0}
    assert decide_round(rr("spin","X", air="yes"), rules, m).action == "redo" and m == {"X":0,"B":0}

def test_unknown_touch_and_maybe_air_still_score(rules):
    o = decide_round(rr("over","B", touch={"X":"unknown","B":"no"}, air="maybe"), rules, {"X":0,"B":0})
    assert (o.action, o.scorer, o.points) == ("score","B",2)

def test_draw_redo(rules):
    assert decide_round(rr(None, None), rules, {"X":0,"B":0}).action == "redo"

def test_second_miss_penalty(rules):
    rules.launch_check["repeat_miss_penalty"] = "opponent_1pt_on_2nd_miss"
    m = {"X":1,"B":0}
    o = decide_round(rr("spin","X", touch={"X":"yes","B":"no"}), rules, m)
    assert (o.action, o.scorer, o.points) == ("score","B",1)

def test_group_matches_first_to_four(rules):
    rs = [rr("xtreme","X"), rr("spin","B"), rr("spin","X"), rr("over","B")]
    ms = group_matches(rs, rules)
    assert ms[0].winner == "X" and ms[0].score == {"X":4,"B":1}
    assert len(ms) == 2 and ms[1].winner is None and ms[1].score == {"X":0,"B":2}
```

- [ ] **Step 2: 執行確認 FAIL**
- [ ] **Step 3: 實作 `rules.py` 與 `config/rules.yaml`**
- [ ] **Step 4: `uv run pytest tests/test_rules.py -v`** — Expected: PASS
- [ ] **Step 5: Commit** — `feat: YAML rules engine and match grouping`

---

### Task 4: 鏡頭校準

**Files:**
- Create: `src/bxjudge/calibration.py`, `tests/test_calibration.py`
- Modify: `src/bxjudge/cli.py`（新增 `calibrate` 子指令）

**Interfaces:**
- Consumes: `Calibration`, `Thresholds`（Task 2）
- Produces:
  - `save_calibration(c: Calibration, path: Path) -> None`、`load_calibration(path: Path) -> Calibration`（JSON；`ref_frame` 存 PNG 相對路徑）
  - `calibration_similarity(frame: np.ndarray, c: Calibration) -> float`（0–1；灰階縮至 320 寬，僅比對 `stadium` 多邊形內，正規化互相關）
  - `find_calibration(frame: np.ndarray, calib_dir: Path, min_similarity: float) -> Calibration | None`（回傳相似度最高且 ≥ 門檻者）
  - `interactive_calibrate(frame: np.ndarray, name: str) -> Calibration`：OpenCV 視窗（WSLg），依序點選 `stadium（場地，含蓋子）, lid（上盤／蓋子）, battle_area（下盤戰鬥區）, over_left / over_right（擊飛區域）, xtreme（極限區域）, launch_left, launch_right`，Enter 結束一個多邊形，`u` 復原上一點；畫面上方顯示目前要標的區域名稱。
  - CLI：`bxjudge calibrate VIDEO --at 1.0 --name 20261001 --out calib/`（取第 `--at` 秒畫面）

- [ ] **Step 1: 寫失敗測試**

```python
def test_roundtrip(tmp_path):
    c = make_calibration(); save_calibration(c, tmp_path/"c.json")
    assert load_calibration(tmp_path/"c.json") == c

def test_similarity_identical_high(ref_img):           # fixture：隨機紋理 640x360 並存為 c.ref_frame
    assert calibration_similarity(ref_img, cal_with(ref_img)) > 0.95

def test_similarity_shifted_low(ref_img):               # Review Focus 3
    shifted = np.roll(ref_img, 40, axis=1)
    assert calibration_similarity(shifted, cal_with(ref_img)) < 0.8

def test_find_calibration_none_when_no_match(tmp_path, ref_img):
    save_calibration(cal_with(ref_img), tmp_path/"a.json")
    assert find_calibration(np.roll(ref_img, 40, axis=1), tmp_path, 0.8) is None
```

- [ ] **Step 2: 執行確認 FAIL**
- [ ] **Step 3: 實作 `calibration.py` 與 CLI 子指令**
- [ ] **Step 4: 測試 PASS；手動驗證**：`uv run bxjudge calibrate ~/bx/videos/20261001/20261001_205842.mp4 --name 20261001 --out calib/`，WSLg 視窗可開啟、7 個多邊形存成 `calib/20261001.json` 與 `calib/20261001.png`。
- [ ] **Step 5: Commit** — `feat: camera calibration and similarity check`

---

### Task 5: 影片同步、清單與解碼

**Files:**
- Create: `scripts/sync_videos.sh`, `src/bxjudge/manifest.py`, `src/bxjudge/video.py`, `tests/test_manifest.py`, `tests/test_video.py`
- Modify: `tests/helpers.py`（`make_video(path, seconds=2, fps=60, audio=False)` 用 `ffmpeg -f lavfi testsrc` 產生）

**Interfaces:**
- Produces:
  - `VideoInfo(path: Path, size: int, mtime: float, duration: float, fps: float, width: int, height: int, codec: str, has_audio: bool, recorded_at: datetime | None)`
  - `parse_recorded_at(name: str) -> datetime | None`（`YYYYMMDD_HHMMSS`）
  - `probe(path: Path) -> VideoInfo`（ffprobe JSON）
  - `scan(folder: Path, db_path: Path) -> list[VideoInfo]`（只回傳新增或 size/mtime 變更者；`mark_processed(info, db_path)` 寫入）
  - `class VideoDecodeError(Exception)`
  - `iter_frames(path: Path, start: float = 0.0, end: float | None = None, long_edge: int = 1280, hwaccel: str = "cuda") -> Iterator[tuple[int, np.ndarray]]`（ffmpeg rawvideo BGR pipe；cuda 失敗自動改 cpu 重試一次；仍失敗拋 `VideoDecodeError`；frame 編號以來源 fps 計，不降幀）

`scripts/sync_videos.sh <drive_folder_id> <batch_name>`：
```bash
rclone copy gdrive: ~/bx/videos/"$2" --drive-root-folder-id "$1" --include "*.mp4" --progress
```
README 註記：使用者先自行執行 `rclone config` 建立 remote `gdrive`，scope 選 `drive.readonly`，在瀏覽器完成授權。10/1 批次的資料夾 ID：`1ExampleDriveFolderId0123456789ab`。

- [ ] **Step 1: 寫失敗測試**

```python
def test_parse_recorded_at():
    assert parse_recorded_at("20261001_205842.mp4") == datetime(2026,10,1,20,58,42)
    assert parse_recorded_at("clip.mp4") is None

def test_probe(tmp_path):
    p = make_video(tmp_path/"20261001_000001.mp4", seconds=2, fps=60, audio=True)
    i = probe(p); assert i.fps == 60 and i.has_audio and abs(i.duration-2) < 0.1

def test_scan_idempotent(tmp_path):                    # Review Focus 5
    make_video(tmp_path/"a.mp4"); db = tmp_path/"db.json"
    first = scan(tmp_path, db); [mark_processed(i, db) for i in first]
    assert len(first) == 1 and scan(tmp_path, db) == []

def test_iter_frames_keeps_all_frames(tmp_path):
    p = make_video(tmp_path/"a.mp4", seconds=2, fps=60)
    assert sum(1 for _ in iter_frames(p, hwaccel="cpu")) == 120

def test_corrupt_video_raises(tmp_path):              # Review Focus 4
    bad = tmp_path/"bad.mp4"; bad.write_bytes(b"\x00"*1024)
    with pytest.raises(VideoDecodeError):
        list(iter_frames(bad, hwaccel="cpu"))
```

- [ ] **Step 2: 執行確認 FAIL**
- [ ] **Step 3: 實作 `manifest.py`、`video.py`、`sync_videos.sh`**
- [ ] **Step 4: 測試 PASS；手動驗證**：同步 10/1 批次，`uv run bxjudge scan ~/bx/videos/20261001` 列出全部影片與 fps / 解析度 / 編碼。在 spec §4 表格旁記下實際的解析度、fps、編碼，與「4K 60fps、一支一回合」推測比對。
- [ ] **Step 5: Commit** — `feat: rclone sync, manifest, ffmpeg decoding`

---

### Task 6: 盤面位移序列

**Files:**
- Create: `src/bxjudge/stadium_motion.py`, `tests/test_stadium_motion.py`

**Interfaces:**
- Consumes: `Calibration`, `polygon_mask`（Task 2）
- Produces: `stadium_displacement(frames: Iterable[tuple[int, np.ndarray]], c: Calibration, ref: np.ndarray) -> dict[int, float]`（每格相對參考畫面的平移量，像素）

做法：遮罩 = `stadium` 減 `battle_area`（保留盤緣與蓋子的邊緣特徵，排除盤底的陀螺）；灰階、`cv2.findTransformECC` 搭配 `MOTION_TRANSLATION`，回傳平移向量長度。ECC 不收斂時該格回傳 `nan`。

- [ ] **Step 1: 寫失敗測試**

```python
def test_detects_shift(ref_img):
    frames = [(0, ref_img), (1, shift(ref_img, 3, 0)), (2, shift(ref_img, 0, -5))]
    d = stadium_displacement(frames, make_calibration(), ref_img)
    assert abs(d[0]) < 0.5 and abs(d[1]-3) < 0.5 and abs(d[2]-5) < 0.5

def test_ignores_motion_inside_battle_area(ref_img):
    moved = draw_blob(ref_img.copy(), (300,170), r=15)        # 戰鬥區內的「陀螺」
    assert stadium_displacement([(0, moved)], make_calibration(), ref_img)[0] < 0.5
```

- [ ] **Step 2–4: FAIL → 實作 → PASS**
- [ ] **Step 5: Commit** — `feat: stadium displacement via ECC on rim mask`

---

### Task 7: 落地時間點

**Files:**
- Create: `src/bxjudge/landing.py`, `tests/test_landing.py`

**Interfaces:**
- Consumes: `Track`, `Calibration`, `Thresholds`, `velocities`, `point_in_polygon`
- Produces:
  - `find_settle(t: Track, c: Calibration, th: Thresholds) -> tuple[int | None, Zone | None]`（第一個停穩格與其區域）
  - `find_landing_frame(t: Track, c: Calibration, th: Thresholds) -> int | None`（停穩區域為 `battle_area` 時回傳停穩格，否則 None）

演算法（spec 6.4「落地不能只看 2D 位置」）：停穩格 = 第一個滿足以下條件的格 f：(1) f 之前曾出現垂直速度 `vy > fall_speed_px`（下落階段；從未下落則省略此條件）；(2) 從 f 起連續 `settle_frames` 格的 `|vy|` 都 ≤ `fall_speed_px`；(3) 這些格的 `zone_of` 都相同。回傳 f 與該區域。停在 `lid`（蓋子上）、`stadium_other`（卡在盤緣）、`outside` → 未落入戰鬥區。

- [ ] **Step 1: 寫失敗測試**

```python
def test_parabola_then_orbit_lands_near_truth(th):
    t = make_track(1, "bey", parabola_then_orbit(land_at=30))   # helper：0–29 格自 launch_left 下落，30 起繞圈
    assert abs(find_landing_frame(t, make_calibration(), th) - 30) <= 2

def test_projected_inside_while_falling_not_landed(th):
    centers = {f: (300, 70 + 10*f) for f in range(15)}          # 投影已進戰鬥區但仍快速下落
    centers.update({f: (300+f, 220) for f in range(15, 40)})
    assert find_landing_frame(make_track(1,"bey",centers), make_calibration(), th) >= 15

def test_settles_on_lid_not_landed(th):
    centers = {f: (80, 30 + 3*f) for f in range(10)}              # 下落到蓋子區（不與戰鬥區重疊處）
    centers.update({f: (80, 60) for f in range(10, 40)})
    t = make_track(1, "bey", centers)
    assert find_settle(t, make_calibration(), th)[1] == "lid"
    assert find_landing_frame(t, make_calibration(), th) is None

def test_never_enters_battle_area(th):
    centers = {f: (20, 10 + f) for f in range(40)}
    assert find_landing_frame(make_track(1,"bey",centers), make_calibration(), th) is None
```

- [ ] **Step 2–4: FAIL → 實作 → PASS**
- [ ] **Step 5: Commit** — `feat: landing frame detection`

---

### Task 8: 發射檢查（方別、觸碰、落入、空碰）

**Files:**
- Create: `src/bxjudge/launch_check.py`, `tests/test_launch_check.py`

**Interfaces:**
- Consumes: Tasks 2, 6, 7
- Produces:
  - `assign_sides(bey_tracks: list[Track], c: Calibration) -> dict[int, Side]`（首次出現中心在 `launch_left` → X、`launch_right` → B；都不在則以 `stadium` 重心的左右判定）
  - `impact_frames(bey_tracks: list[Track], landings: dict[int, int | None], c: Calibration, th: Thresholds) -> set[int]`（落地格、兩陀螺框接觸格、陀螺中心距 `stadium` 邊界 < `rim_margin_px` 且 `is_abrupt_change` 的格）
  - `detect_touch(disp: dict[int, float], launchers: list[Track], sides_of_launcher: dict[int, Side], impacts: set[int], window: tuple[int,int], c: Calibration, th: Thresholds) -> tuple[dict[Side, TouchState], list[str]]`
  - `detect_air_collision(tx: Track, tb: Track, land_x: int | None, land_b: int | None, th: Thresholds) -> AirState`
  - `check_launch(bey_tracks, launchers, disp, c, th, fps) -> tuple[LaunchCheck, dict[int, Side]]`（`landing_zone[side]` = `find_settle` 的區域；`landed[side] = landing_zone[side] == "battle_area"`；從未停穩為 None 且 landed 為 False）（發射窗 = 第一顆陀螺出現前 `launch_pre_s` 到兩顆都落地後 `launch_post_s`；任一方未落地時窗口結束於第一顆出現後 3 秒）

`detect_touch` 規則（spec 6.4 三條件）：在窗內找 `disp > shake_px` 的晃動起點 s。若 s 與 `impacts` 中任一格距離 ≤ `impact_coincide_frames` → 不算觸碰，notes 加 `shake_by_bey_impact@{s}`。否則找 s 時刻框與 `stadium` 邊界距離 ≤ `rim_margin_px` 的 launcher：有 → 該 launcher 所屬方 `yes`；s 時刻**有** launcher 入鏡但都不在盤緣附近 → 雙方 `no`，notes 加 `shake_not_by_launcher_body@{s}`（launcher 只代表發射器本體，拉繩棉線、拉條碰盤不算觸碰）；s 時刻**無** launcher 入鏡 → 雙方 `unknown`，notes 加 `shake_cause_unknown@{s}`。無晃動 → 雙方 `no`。launcher 的方別以其中心在 `launch_left` / `launch_right` 判定。

`detect_air_collision`：檢查期間 = 兩軌跡共同存在、且早於 `min(land_x, land_b)` 的格（任一為 None 時用另一個；皆 None 用整段）。條件 A：`boxes_touch`；條件 B：在同一格 ±`air_collision_frames` 內兩者都 `is_abrupt_change`。A 且 B → `yes`；僅其一 → `maybe`；皆無 → `no`。

- [ ] **Step 1: 寫失敗測試**

```python
def test_assign_sides():
    x = make_track(1,"bey",{0:(100,50)}); b = make_track(2,"bey",{0:(500,50)})
    assert assign_sides([x,b], make_calibration()) == {1:"X", 2:"B"}

def test_touch_left_launcher_at_rim(th):
    disp = {f: (5.0 if f == 10 else 0.0) for f in range(30)}
    launcher = make_track(9, "launcher", {10: (100, 22)})       # 貼近 stadium 上緣、位於 launch_left
    touch, _ = detect_touch(disp, [launcher], {9:"X"}, set(), (0,30), make_calibration(), th)
    assert touch == {"X":"yes", "B":"no"}

def test_shake_with_landing_is_not_touch(th):
    disp = {f: (5.0 if f == 10 else 0.0) for f in range(30)}
    touch, notes = detect_touch(disp, [], {}, {12}, (0,30), make_calibration(), th)
    assert touch == {"X":"no","B":"no"} and any(n.startswith("shake_by_bey_impact") for n in notes)

def test_shake_without_launcher_is_unknown(th):
    disp = {f: (5.0 if f == 10 else 0.0) for f in range(30)}
    touch, _ = detect_touch(disp, [], {}, set(), (0,30), make_calibration(), th)
    assert touch == {"X":"unknown","B":"unknown"}

def test_shake_with_launcher_body_away_from_rim_is_no(th):   # 拉繩 / 拉條碰盤不算
    disp = {f: (5.0 if f == 10 else 0.0) for f in range(30)}
    launcher = make_track(9, "launcher", {10: (150, 5)})          # 本體入鏡但離盤緣遠
    touch, notes = detect_touch(disp, [launcher], {9:"X"}, set(), (0,30), make_calibration(), th)
    assert touch == {"X":"no","B":"no"} and any(n.startswith("shake_not_by_launcher_body") for n in notes)

def test_shake_outside_window_ignored(th):
    disp = {f: (5.0 if f == 50 else 0.0) for f in range(60)}
    assert detect_touch(disp, [], {}, set(), (0,30), make_calibration(), th)[0] == {"X":"no","B":"no"}

def test_landing_zone_recorded(th):
    tx = make_track(1, "bey", parabola_then_orbit(land_at=30, land_point=(300, 200)))
    tb = make_track(2, "bey", fall_then_rest(rest_at=32, rest_point=(560, 60), start=(500, 10)))   # 停在蓋子上
    lc, _ = check_launch([tx, tb], [], {f: 0.0 for f in range(120)}, make_calibration(), th, 60)
    assert lc.landing_zone == {"X": "battle_area", "B": "lid"}
    assert lc.landed == {"X": True, "B": False}

def test_air_collision_yes_maybe_no(th):
    assert detect_air_collision(*crossing_tracks(bounce=True), 40, 42, th) == "yes"
    assert detect_air_collision(*crossing_tracks(bounce=False), 40, 42, th) == "maybe"   # 前後重疊但軌跡不變
    assert detect_air_collision(*separate_tracks(), 40, 42, th) == "no"

def test_collision_after_first_landing_is_not_air(th):
    tx, tb = crossing_tracks(bounce=True, at=45)
    assert detect_air_collision(tx, tb, 40, 50, th) == "no"
```

- [ ] **Step 2–4: FAIL → 實作 → PASS**
- [ ] **Step 5: Commit** — `feat: launch check - sides, touch, landing, air collision`

---

### Task 9: 旋轉方向、停轉與終結判定

**Files:**
- Create: `src/bxjudge/finish.py`, `tests/test_finish.py`
- Modify: `src/bxjudge/types.py`（新增 `SpinDir`、`RoundResult.spin_dir`）、`src/bxjudge/config.py` 與 `config/thresholds.yaml`（新增下列門檻鍵）

新增門檻（初始值，Task 16 以資料校正）：

```yaml
spin_dir_window_s: 2.0        # 落地後量繞行方向的時間窗
spin_dir_min_turn_deg: 180    # 繞行累計角度未達此值 → unknown
orbit_to_spin: {cw: left, ccw: right}   # 繞行方向 → 旋轉方向；驗收時用已知陀螺影片確認
reverse_frames: 6             # 反轉需持續的格數
reverse_max_rate_deg: 20      # 只有畫面旋轉速率（度/格）低於此值時看到的反轉才算數
reverse_max_move_px: 3.0      # 且中心移動（px/格）低於此值
```

**Interfaces:**
- Consumes: Tasks 2, 8
- Produces:
  - `types.py`：`SpinDir = Literal["right","left","unknown"]`；`RoundResult` 新增 `spin_dir: dict[Side, SpinDir] | None = None`（放在既有預設欄位之後）
  - `spin_activity(crops: dict[int, np.ndarray]) -> dict[int, float]`（輸入為「格號 → 該陀螺框裁切後縮至 64×64 的灰階圖」；輸出每格與前一格的平均絕對差）
  - `rotation_rate(crops: dict[int, np.ndarray]) -> dict[int, float]`（相鄰兩格 Farneback 光流，對裁切中心取切向分量的加權平均換算成**帶正負號的畫面旋轉速率**（度/格；正 = 畫面上順時針））
  - `orbit_direction(t: Track, c: Calibration, landing_frame: int | None, th: Thresholds, fps: float) -> SpinDir`（落地後 `spin_dir_window_s` 秒內，中心相對 `battle_area` 重心的累計角度；|累計| < `spin_dir_min_turn_deg` → unknown；否則依 `orbit_to_spin` 換算）
  - `stop_frame(t: Track, activity: dict[int, float], rate: dict[int, float], spin_dir: SpinDir, th: Thresholds) -> tuple[int | None, str | None]`：回傳最早的轉停格與原因 `"still"` 或 `"reversed"`。
    - still：中心移動 < `still_px` 且 activity < `still_activity`，連續 `still_frames` 格的第一格。
    - reversed：連續 `reverse_frames` 格滿足「`|rate|` < `reverse_max_rate_deg`、中心移動 < `reverse_max_move_px`、rate 的符號與基準方向相反」的第一格。基準方向 = `spin_dir` 經 `orbit_to_spin` 反查後的畫面旋轉方向；`spin_dir == "unknown"` 時，基準 = 第一段連續 `reverse_frames` 格滿足速率與移動條件、符號一致的方向。**快速旋轉時的符號變化一律忽略**（混疊）。
  - `id_swap_risk(tx: Track, tb: Track, th: Thresholds) -> bool`（框重疊連續 ≥ `id_swap_overlap_frames` 格）
  - `FinishResult(finish: Finish | None, loser: Side | None, frame: int | None, confidence: float, reasons: list[str])`
  - `detect_finish(bey: dict[Side, Track], parts: list[Track], stops: dict[Side, int | None], c: Calibration, th: Thresholds, fps: float) -> FinishResult`

判定：對每方找最早的終結事件：Xtreme（中心進入 `xtreme`）、Over（進入 `over_left` / `over_right`）、Burst（該方軌跡結束後 `burst_part_window_s` 內、末位置附近出現 ≥ 1 條 `bey_part` 軌跡）、Spin（雙方都在戰鬥區，`stops` 較早者；`stops` 由 `stop_frame` 的格號組成，原因寫入 reasons，例如 `B_stop_reversed`）。取時間最早的事件；winner = 對方。兩方 stop 相差 ≤ `simultaneous_stop_frames`，或兩方在同一窗內同時出局 → `finish=None`，reasons 加 `draw`。信心 = 相關偵測平均分數；`id_swap_risk` 為真 → 信心 ×0.5 並加 `id_swap_risk`。

- [ ] **Step 1: 寫失敗測試**

```python
def test_xtreme(th):  r = detect_finish(*scene(b_path_into="xtreme"), th, 60); assert (r.finish, r.loser) == ("xtreme","B")
def test_over(th):    r = detect_finish(*scene(x_path_into="over_left"), th, 60); assert (r.finish, r.loser) == ("over","X")
def test_burst(th):   r = detect_finish(*scene(b_burst_at=80), th, 60);        assert (r.finish, r.loser) == ("burst","B")
def test_spin(th):    r = detect_finish(*scene(stops={"X":300,"B":250}), th, 60); assert (r.finish, r.loser) == ("spin","B")

def test_earliest_event_wins(th):
    r = detect_finish(*scene(b_path_into="xtreme", b_enter_at=100, stops={"X":200,"B":None}), th, 60)
    assert r.finish == "xtreme"

def test_simultaneous_stop_is_draw(th):
    r = detect_finish(*scene(stops={"X":250,"B":252}), th, 60)
    assert r.finish is None and "draw" in r.reasons

def test_id_swap_risk_lowers_confidence(th):            # Review Focus 2
    r = detect_finish(*scene(overlap_frames=5, stops={"X":300,"B":250}), th, 60)
    assert "id_swap_risk" in r.reasons and r.confidence < th.review_confidence

def test_rotation_rate_sign():
    crops = rotating_crops(deg_per_frame=5, n=10)          # helper：合成紋理每格順時針轉 5°
    assert all(v > 0 for v in list(rotation_rate(crops).values())[1:])
    crops = rotating_crops(deg_per_frame=-5, n=10)
    assert all(v < 0 for v in list(rotation_rate(crops).values())[1:])

def test_orbit_direction(th):
    c = make_calibration()
    cw = make_track(1, "bey", circle_path(center=(320,170), r=60, deg_per_frame=4, n=120))   # 畫面順時針繞行
    assert orbit_direction(cw, c, 0, th, 60) == th.orbit_to_spin["cw"]
    wobble = make_track(2, "bey", circle_path(center=(320,170), r=60, deg_per_frame=1, n=60))  # 只繞 60°
    assert orbit_direction(wobble, c, 0, th, 60) == "unknown"

def test_stop_by_reversal_when_slow(th):
    t, act, rate = slowing_bey(reverse_at=200)              # 慢速且幾乎不動時方向反轉，持續 ≥ reverse_frames
    assert stop_frame(t, act, rate, "right", th) == (200, "reversed")

def test_fast_sign_flips_ignored(th):                       # 混疊防護
    t, act, rate = slowing_bey(reverse_at=None, aliasing_flips_at=[50, 90])   # 快速階段 |rate| 大、符號亂跳
    assert stop_frame(t, act, rate, "right", th) == (None, None)

def test_still_detected(th):
    t, act, rate = slowing_bey(reverse_at=None, still_at=260)
    assert stop_frame(t, act, rate, "right", th) == (260, "still")
```

（`scene(...)`、`rotating_crops`、`circle_path`、`slowing_bey` 都在 `tests/helpers.py`，以 `make_calibration()` 座標生成。`rotating_crops` 以 `cv2.warpAffine` 旋轉一張 64×64 隨機紋理。）

- [ ] **Step 2–4: FAIL → 實作 → PASS**
- [ ] **Step 5: 校正記錄**：在報告中記下 `rotation_rate` 對 `rotating_crops(±5)` 的實測值，供 Task 16 調整 `reverse_max_rate_deg`。
- [ ] **Step 6: Commit** — `feat: spin direction, stop-by-reversal, finish detection`

---

### Task 10: 回合切分

**Files:**
- Create: `src/bxjudge/segment.py`, `tests/test_segment.py`

**Interfaces:**
- Produces: `segment_rounds(bey_tracks: list[Track], fps: float, th: Thresholds) -> list[tuple[int, int]]`（回合開始 = 無活躍 bey 軌跡 ≥ `round_gap_s` 後第一條新軌跡出現；結束 = 該回合所有軌跡的最後一格）

- [ ] **Step 1: 寫失敗測試**

```python
def test_single_round(th):  assert segment_rounds([trk(0,300), trk(2,310)], 60, th) == [(0,310)]
def test_two_rounds(th):    assert segment_rounds([trk(0,300), trk(0,290), trk(600,900), trk(602,880)], 60, th) == [(0,300),(600,900)]
def test_no_tracks(th):     assert segment_rounds([], 60, th) == []          # Review Focus 1
```

- [ ] **Step 2–4: FAIL → 實作 → PASS**
- [ ] **Step 5: Commit** — `feat: round segmentation`

---

### Task 11: 標註流程（抽格、SAM 3.1 預標、匯出）

**Files:**
- Create: `src/bxjudge/labeling/sample_frames.py`, `src/bxjudge/labeling/sam3_prelabel.py`, `src/bxjudge/labeling/export.py`, `tests/test_labeling.py`
- Modify: `src/bxjudge/cli.py`（`bxjudge label sample|prelabel|export`）

**Interfaces:**
- Produces:
  - `change_energy(path: Path, long_edge: int = 640) -> dict[int, float]`（相鄰格灰階平均絕對差）
  - `select_frames(energy: dict[int, float], fps: float, n: int, uniform_ratio: float = 0.3, min_gap_s: float = 0.1) -> list[int]`（(1-uniform_ratio)·n 取能量峰值，其餘均勻取樣；任兩格間隔 ≥ min_gap_s）
  - `to_label_studio(image: str, w: int, h: int, boxes: list[tuple[str, Box, float]]) -> dict`（Label Studio task，`predictions[0].result` 為 rectanglelabels，座標為百分比）
  - `filter_launcher_boxes(boxes: list[tuple[str, Box, float]], max_aspect: float = 4.0) -> list[tuple[str, Box, float]]`（只過濾 `launcher` 類：長邊 / 短邊 > max_aspect 的框丟掉；其他類別原樣保留）
  - `prelabel(video: Path, frames: list[int], prompts: dict[str, str], out_json: Path) -> int`（SAM 3.1 文字提示，影片模式在選定格之間傳播；寫出 Label Studio 匯入 JSON 與 JPG；回傳框數）
  - `label_studio_to_coco(export_json: Path, out_dir: Path, val_ratio: float = 0.2) -> None`（**依影片**切 train/val，同一支影片不跨集合；類別固定 `["bey","bey_part","launcher"]`）

初始提示詞（spec §13 留給實作者調整）：`{"bey": "spinning top", "bey_part": "small plastic part", "launcher": "toy launcher"}`。

**標註準則（寫進 README 並在 Label Studio 標籤說明中顯示）**：`launcher` 只框**發射器本體**；拉繩式發射器的棉線、拉條式發射器的拉條**一律不框**。SAM 3.1 預標若把拉繩或拉條框進去，使用者修正時要縮回本體。`prelabel` 對 `launcher` 預標框做一次後處理：去掉細長比 > 4 的框（多半是拉繩或拉條），並在報告中列出被去掉的數量。

- [ ] **Step 1: 寫失敗測試**

```python
def test_select_frames_hits_peaks():
    e = {f: (100.0 if f in (60, 300, 600) else 1.0) for f in range(900)}
    s = select_frames(e, 60, n=10)
    assert {60, 300, 600} <= set(s) and len(s) == 10

def test_to_label_studio_percent_units():
    t = to_label_studio("a.jpg", 1000, 500, [("bey", (100,50,200,150), 0.9)])
    v = t["predictions"][0]["result"][0]["value"]
    assert (v["x"], v["y"], v["width"], v["height"], v["rectanglelabels"]) == (10.0, 10.0, 10.0, 20.0, ["bey"])

def test_prelabel_drops_elongated_launcher_boxes():
    boxes = [("launcher", (100,100,140,130), 0.9), ("launcher", (100,100,400,110), 0.8)]   # 第二個是細長拉繩 / 拉條
    kept = filter_launcher_boxes(boxes, max_aspect=4.0)
    assert kept == [("launcher", (100,100,140,130), 0.9)]

def test_coco_split_by_video(tmp_path):
    label_studio_to_coco(fixture_export(videos=["a","a","b","b","c"]), tmp_path)
    train, val = load_coco_images(tmp_path)
    assert not ({video_of(i) for i in train} & {video_of(i) for i in val})
```

- [ ] **Step 2–4: FAIL → 實作 → PASS**
- [ ] **Step 5: 使用者準備步驟（文件化於 README）**：在 Hugging Face 申請 SAM 3 權重存取並 `huggingface-cli login`；`uv tool install label-studio`。
- [ ] **Step 6: 手動驗證**：對 10/1 批次執行 `bxjudge label sample --per-video 8`（約 50 支 × 8 ≈ 400 張）→ `bxjudge label prelabel` → `label-studio start` 匯入，確認預標框出現在 Windows 瀏覽器 `http://localhost:8080`。**交由使用者標註**完成後 `bxjudge label export`。
- [ ] **Step 7: Commit** — `feat: event-aware sampling, SAM 3.1 prelabel, COCO export`

---

### Task 12: 偵測器、追蹤與訓練比較

**Files:**
- Create: `src/bxjudge/detect.py`, `src/bxjudge/track.py`, `src/bxjudge/train.py`, `tests/test_track.py`
- Modify: `src/bxjudge/cli.py`（`bxjudge train`、`bxjudge compare`）

**Interfaces:**
- Consumes: COCO 資料集（Task 11）、`Detection` / `Track`（Task 2）
- Produces:
  - `class Detector(Protocol): def __call__(self, frame: np.ndarray, frame_idx: int) -> list[Detection]`
  - `RFDETRDetector(weights: Path, conf: float = 0.3)`、`YOLODetector(weights: Path, conf: float = 0.3)`、`load_detector(kind: Literal["rfdetr","yolo26"], weights: Path) -> Detector`
  - `track_frames(frames: Iterable[tuple[int, np.ndarray]], detector: Detector, fps: float) -> list[Track]`（supervision ByteTrack，各類別分開追蹤）
  - `train(kind, dataset: Path, out: Path, epochs: int) -> Path`（回傳最佳權重路徑）
  - `compare(weights: dict[str, Path], dataset: Path) -> dict[str, dict]`（各模型 val 的 `bey` mAP50-95、各類別 AP、每格推論毫秒數；寫入 `runs/compare.json`）
  - `weights/selected.json`：`{"kind": ..., "weights": ...}`，`pipeline` 由此載入

選定規則：`bey` 類 mAP50-95 較高者勝；差距 < 1 點時選推論較快者。

- [ ] **Step 1: 寫失敗測試**

```python
def test_track_frames_keeps_identity():
    det = FakeDetector({f: [Detection(f,"bey",(10+5*f,10,30+5*f,30),0.9)] for f in range(20)})
    tracks = track_frames(((f, blank()) for f in range(20)), det, 60)
    assert len(tracks) == 1 and tracks[0].first_frame() == 0 and tracks[0].last_frame() == 19

def test_tracks_split_by_class():
    det = FakeDetector({f: [Detection(f,"bey",(0,0,20,20),0.9), Detection(f,"launcher",(100,0,140,30),0.9)] for f in range(5)})
    assert sorted(t.cls for t in track_frames(((f, blank()) for f in range(5)), det, 60)) == ["bey","launcher"]
```

- [ ] **Step 2–4: FAIL → 實作 → PASS**
- [ ] **Step 5: 訓練與比較（GPU，手動）**：`bxjudge train --kind rfdetr` 與 `--kind yolo26`（同一份 COCO），`bxjudge compare`。Expected：`runs/compare.json` 兩模型都有數值，`weights/selected.json` 已寫入；訓練期間 `nvidia-smi` VRAM < 21 GB。
- [ ] **Step 6: Commit** — `feat: detector adapters, ByteTrack, training comparison`

---

### Task 13: 單支影片管線、輸出與 CLI

**Files:**
- Create: `src/bxjudge/pipeline.py`, `src/bxjudge/output.py`, `tests/test_pipeline.py`, `tests/test_output.py`
- Modify: `src/bxjudge/cli.py`（`bxjudge run BATCH_DIR --calib calib/ --out results/BATCH`）

**Interfaces:**
- Consumes: Tasks 3–10、12
- Produces:
  - `judge_video(info: VideoInfo, calib_dir: Path, detector: Detector, rules: Rules, th: Thresholds) -> list[RoundResult]`
  - `write_results(new: list[RoundResult], out_dir: Path, rules: Rules) -> None`（`rounds.json` 為權威資料；`rounds.csv` 欄位照 spec §9：`video, round, t_launch, t_end, x_touch, b_touch, x_landed, b_landed, x_land_zone, b_land_zone, x_spin_dir, b_spin_dir, air_collision, finish, winner, points, confidence, needs_review, review_reasons, vlm_note, reviewed, reviewer_fix`；`matches.csv` 由 `group_matches` 以 effective 值重建）
  - `effective(row: dict) -> dict`（`reviewer_fix` 覆蓋原欄位後的結果）、`row_to_result(row: dict) -> RoundResult`
  - `record_video_timing(out_dir: Path, video: str, duration_s: float, processing_s: float) -> None`（寫入 `videos.json`，供 Task 16 計算速度）
  - `write_trajectory(out_dir: Path, r: RoundResult, bey: dict[Side, Track], parts: list[Track], c: Calibration, fps: float, events: dict[str, int | None]) -> tuple[Path, Path]`：
    - `trajectories/<video stem>_r<round>.csv`，欄位 `frame, t, side, track_id, cls, x, y, zone, score`（`side` 對零件為空；每格每條軌跡一列）
    - `trajectories/<video stem>_r<round>.png`：在校準 PNG 上畫 7 個區域外框、X 路徑藍色、B 路徑紅色，`events` 鍵 `landing_X, landing_B, air_collision, finish` 的位置畫標記與文字
    - 重跑時覆寫同名檔案（軌跡不屬於覆核資料）
  - `run_batch(folder: Path, calib_dir: Path, out_dir: Path) -> None`（`scan` → 逐支 `judge_video` → `write_results` → `record_video_timing` → `mark_processed`；單支例外不中斷批次）

`judge_video` 流程：取第 1 秒畫面 → `find_calibration`，None → 回傳單筆 `needs_review`、reasons `["recalibrate"]`；解碼一次，同時產生 `stadium_displacement`（以校準 PNG 為參考）、`track_frames`，並把每條 `bey` 軌跡每格的框裁切縮成 64×64 灰階存入 `crops[track_id]`（供 `spin_activity`）→ `segment_rounds`，空 → 單筆 `["no_round"]`；每回合：`check_launch` → 每方 `orbit_direction(t, c, landing_frame, th, fps)` 得 `spin_dir` → `stop_frame(t, spin_activity(crops[t.track_id]), rotation_rate(crops[t.track_id]), spin_dir, th)` → `detect_finish` → 組 `RoundResult`（`points = finish_points`）→ `write_trajectory`。`needs_review` = 信心 < `review_confidence`，或任一 touch 為 unknown，或 air 為 maybe，或 finish 為 None，或 reasons 非空。`VideoDecodeError` → 單筆 `["decode_error"]`。

`write_results` 合併規則：以 `(video, round)` 為鍵；既有列 `reviewed == True` 時**保留不動**，否則以新結果取代；不得產生重複鍵。

- [ ] **Step 1: 寫失敗測試**

```python
def test_recalibrate_when_camera_moved(monkeypatch, info):         # Review Focus 3
    monkeypatch.setattr(pipeline, "find_calibration", lambda *a, **k: None)
    [r] = judge_video(info, Path("calib"), FakeDetector({}), rules, th)
    assert r.needs_review and r.review_reasons == ["recalibrate"]

def test_no_round(monkeypatch, info):                               # Review Focus 1
    stub_calibrated(monkeypatch)
    [r] = judge_video(info, Path("calib"), FakeDetector({}), rules, th)
    assert r.review_reasons == ["no_round"]

def test_decode_error_does_not_stop_batch(tmp_path):               # Review Focus 4
    make_video(tmp_path/"ok.mp4"); (tmp_path/"bad.mp4").write_bytes(b"\x00"*1024)
    run_batch_with_fakes(tmp_path)
    reasons = {r["video"]: r["review_reasons"] for r in read_rounds(tmp_path/"out")}
    assert reasons["bad.mp4"] == ["decode_error"] and "ok.mp4" in reasons

def test_write_trajectory(tmp_path):
    c = make_calibration_with_png(tmp_path)
    bey = {"X": make_track(1,"bey",{f:(300,100) for f in range(10)}), "B": make_track(2,"bey",{f:(320,320) for f in range(10)})}
    csv_p, png_p = write_trajectory(tmp_path, res("a.mp4", 1), bey, [], c, 60, {"landing_X": 2, "landing_B": 3, "air_collision": None, "finish": 9})
    rows = list(csv.DictReader(open(csv_p)))
    assert len(rows) == 20 and {r["zone"] for r in rows if r["side"] == "X"} == {"battle_area"}
    assert {r["zone"] for r in rows if r["side"] == "B"} == {"xtreme"} and png_p.exists()

def test_rerun_keeps_reviewed_rows(tmp_path):                      # Review Focus 5
    write_results([res("a.mp4", 1, finish="spin")], tmp_path, rules)
    mark_reviewed(tmp_path, "a.mp4", 1, {"finish": "over"})
    write_results([res("a.mp4", 1, finish="burst"), res("a.mp4", 2)], tmp_path, rules)
    rows = read_rounds(tmp_path)
    assert len(rows) == 2 and row(rows, "a.mp4", 1)["reviewer_fix"] == {"finish": "over"}
```

- [ ] **Step 2–4: FAIL → 實作 → PASS**
- [ ] **Step 5: 實機冒煙測試**：`bxjudge run ~/bx/videos/20261001 --limit 3`。Expected：`results/20261001/rounds.csv` 有 3 支影片的列；每支處理時間 ≤ 影片長度（印在 log）；`nvidia-smi` 無 OOM。
- [ ] **Step 6: Commit** — `feat: per-video pipeline, merge-safe outputs, run command`

---

### Task 14: VLM 覆核

**Files:**
- Create: `src/bxjudge/vlm.py`, `tests/test_vlm.py`
- Modify: `src/bxjudge/pipeline.py`（`run_batch` 結束後、偵測模型卸載後呼叫）

**Interfaces:**
- Produces:
  - `VlmAnswer(answer: str, reason: str, model: str)`
  - `ask_vlm(images: list[np.ndarray], question: str, model: str = "qwen3.8:27b", fallback: str = "qwen3-vl:8b", host: str = "http://localhost:11434") -> VlmAnswer`（`/api/chat`，圖片 base64，`format` 為 JSON schema `{answer, reason}`；主模型回傳錯誤或 OOM 時改用 fallback）
  - `question_for(r: RoundResult) -> str`（依 review_reasons：`shake_cause_unknown` → 「發射時是否有發射器或手碰到盤面？是哪一側？」；`id_swap_risk` / 低信心 → 「左方開始的陀螺最後停在哪個區域，還是先停止旋轉？」；air `maybe` → 「兩顆陀螺落地前是否在空中相撞？」）
  - `review_low_confidence(results: list[RoundResult], get_frames: Callable[[RoundResult], list[np.ndarray]]) -> list[RoundResult]`（只處理 `needs_review` 列；只寫 `vlm_note`）

- [ ] **Step 1: 寫失敗測試**

```python
def test_vlm_note_only(monkeypatch):
    monkeypatch.setattr(vlm, "_post", lambda *a, **k: {"message": {"content": '{"answer":"over","reason":"..."}'}})
    r = res("a.mp4", 1, finish="spin", winner="X", needs_review=True)
    [out] = review_low_confidence([r], lambda _: [blank()])
    assert out.finish == "spin" and out.winner == "X" and "over" in out.vlm_note

def test_fallback_on_error(monkeypatch):
    calls = []
    def fake(host, payload):
        calls.append(payload["model"])
        if payload["model"] == "qwen3.8:27b": raise VlmError("out of memory")
        return {"message": {"content": '{"answer":"no","reason":"x"}'}}
    monkeypatch.setattr(vlm, "_post", fake)
    assert ask_vlm([blank()], "q").model == "qwen3-vl:8b" and calls == ["qwen3.8:27b","qwen3-vl:8b"]
```

- [ ] **Step 2–4: FAIL → 實作 → PASS**
- [ ] **Step 5: 實機驗證**：在 WSL 安裝 Ollama（官方安裝程式），`ollama pull qwen3.8:27b && ollama pull qwen3-vl:8b`；對冒煙測試結果執行 `bxjudge vlm results/20261001`，確認需覆核列出現 `vlm_note`。
- [ ] **Step 6: Commit** — `feat: local VLM second opinion`

---

### Task 15: 覆核介面與修正紀錄

**Files:**
- Create: `src/bxjudge/corrections.py`, `src/bxjudge/review_app.py`, `tests/test_corrections.py`
- Modify: `src/bxjudge/cli.py`（`bxjudge review results/BATCH` 啟動 Streamlit）

**Interfaces:**
- Produces:
  - `Correction(video: str, round: int, field: str, old: Any, new: Any, at: str)`
  - `apply_correction(out_dir: Path, c: Correction) -> None`（更新 `rounds.json` 對應列：`reviewer_fix[field] = new`、`reviewed = True`；追加一行到 `corrections.jsonl`；重建 csv）
  - `load_corrections(out_dir: Path) -> list[Correction]`
  - 使用 Task 13 的 `effective()` 與 `write_results` 的 csv 重建邏輯，不另寫一份

介面：側欄篩選（全部 / 需覆核 / 已覆核）；每列顯示發射瞬間與結束瞬間截圖並疊上校準多邊形、**該回合軌跡圖**（`write_trajectory` 的 PNG）、`review_reasons`、`vlm_note`；`st.video` 從 `t_launch - 1` 秒開始播放；可編輯 `x_touch, b_touch, x_landed, b_landed, x_spin_dir, b_spin_dir, air_collision, finish, winner`，以及「確認無誤」按鈕（只設 `reviewed = True`）。

- [ ] **Step 1: 寫失敗測試**

```python
def test_apply_correction_marks_reviewed(tmp_path):
    write_results([res("a.mp4", 1, finish="spin", winner="X")], tmp_path, rules)
    apply_correction(tmp_path, Correction("a.mp4", 1, "winner", "X", "B", "2026-10-21T00:00:00"))
    row = read_rounds(tmp_path)[0]
    assert row["reviewed"] and effective(row)["winner"] == "B"
    assert len(load_corrections(tmp_path)) == 1
```

- [ ] **Step 2–4: FAIL → 實作 → PASS**
- [ ] **Step 5: 手動驗證**：`bxjudge review results/20261001`，在 Windows 瀏覽器開 `http://localhost:8501`，修改一筆後重新整理，修改仍在；`matches.csv` 比分依修正更新。
- [ ] **Step 6: Commit** — `feat: review UI and corrections log`

---

### Task 16: 評估、門檻校正與整批驗收

**Files:**
- Create: `src/bxjudge/evaluate.py`, `tests/test_evaluate.py`
- Modify: `src/bxjudge/cli.py`（`bxjudge eval results/BATCH`、`bxjudge tune results/BATCH`）

**Interfaces:**
- Consumes: `rounds.json`、`videos.json`、`effective()`（Task 13）、`corrections.jsonl`（Task 15）
- Produces:
  - `Metrics(finish_accuracy_unflagged: float, touch_recall: float, air_recall: float, review_ratio: float, speed_ratio: float, n_reviewed: int)`
  - `evaluate(rows: list[dict]) -> Metrics`（只用 `reviewed` 列當真值；`finish_accuracy_unflagged` = 原判 `needs_review == False` 的列中，原 finish 與 winner 都等於 effective 值的比例；`touch_recall` = 真值有觸碰的方中，原判為 yes 或該列 needs_review 的比例；`air_recall` 同理；`speed_ratio` = 處理秒數 / 影片秒數的中位數）
  - `passes(m: Metrics) -> dict[str, bool]`（門檻：0.95、0.90、0.90、≤ 0.30、≤ 1.0）
  - `derive_shake_px(samples: dict[str, list[float]]) -> float`（samples 鍵 `normal`, `touch`, `bey_impact`；回傳 `touch` 最小值與 `normal` 最大值的中點；兩者重疊時回傳 `normal` 的 99 百分位並在報告中警告）
  - `bxjudge eval` 寫 `reports/eval-YYYYMMDD.md`（各指標、是否達標、錯誤列清單依原因分組）；`bxjudge tune` 依覆核資料重算 `shake_px` 並輸出建議的 `thresholds.yaml` diff（**不**自動覆寫）

- [ ] **Step 1: 寫失敗測試**

```python
def test_finish_accuracy_unflagged():
    rows = [rev(finish="spin", fix={}), rev(finish="spin", fix={"finish":"over"}), rev(finish="spin", needs_review=True, fix={"finish":"burst"})]
    assert evaluate(rows).finish_accuracy_unflagged == 0.5

def test_touch_recall_counts_flagged_as_caught():
    rows = [rev(x_touch="no", needs_review=True, fix={"x_touch":"yes"}), rev(x_touch="no", fix={"x_touch":"yes"})]
    assert evaluate(rows).touch_recall == 0.5

def test_passes_thresholds():
    assert all(passes(Metrics(0.96, 0.92, 0.91, 0.25, 0.8, 50)).values())
    assert passes(Metrics(0.90, 0.92, 0.91, 0.25, 0.8, 50))["finish_accuracy_unflagged"] is False
    assert passes(Metrics(0.96, 0.92, 0.91, 0.35, 0.8, 50))["review_ratio"] is False

def test_derive_shake_px_midpoint():
    assert derive_shake_px({"normal":[0.5,1.0], "touch":[3.0,4.0], "bey_impact":[2.0]}) == 2.0
```

- [ ] **Step 2–4: FAIL → 實作 → PASS**
- [ ] **Step 5: 整批驗收（與使用者一起）**：`bxjudge run ~/bx/videos/20261001` → `bxjudge vlm` → 使用者在覆核介面覆核全部回合 → `bxjudge eval`。Expected：報告列出 spec §11 五項指標與是否達標。未達標項目交由 Claude 依報告調整 `config/thresholds.yaml` / 校準 / 補標註後重跑（spec §10 的優化迴圈）。
- [ ] **Step 5b: 確認旋轉方向對應**：用已知右旋 / 左旋陀螺的回合，檢查 `x_spin_dir` / `b_spin_dir` 是否正確；錯了就對調 `orbit_to_spin` 並重跑；`tune` 報告列出覆核後 spin_dir 的正確率與 `reversed` 轉停被推翻的比例。
- [ ] **Step 6: 核對官方規則**：取得 Takara Tomy 官方最新規則 PDF，逐項核對 `config/rules.yaml`，更新 `rules_version` 並移除 TODO 註解。
- [ ] **Step 7: Commit** — `feat: evaluation report and threshold tuning`

---

## Spec 對照

| Spec 段落 | Task |
|---|---|
| §3 發射檢查（觸碰、落入、空判） | 6、7、8 |
| §3 勝負結果、4 分制 | 3、9 |
| §3 信心與覆核 | 9、13、14、15 |
| §4 限制（斜拍、無手、無聲、4K60、雲端硬碟） | 4、5、6、8 |
| §5 WSL、PyTorch CUDA 13、rclone、模型選擇、VRAM 依序 | 1、5、11、12、14 |
| §6.1–6.5 流程 | 5、4、10、8、9 |
| §7 規則設定檔 | 3、16（核對官方） |
| §8 VLM 不覆蓋 | 14 |
| §9 輸出欄位與覆核介面 | 13、15 |
| §6.5 轉停 = 靜止或反轉（混疊防護）、§6.6 旋轉方向 flag | 9、13（輸出欄位）、15（可修正）、16（對應表確認） |
| §9 陀螺軌跡紀錄（CSV + 軌跡圖） | 2（`zone_of`）、13（`write_trajectory`）、15（顯示） |
| §6.2 校準：場地、上盤（蓋子）、下盤戰鬥區、擊飛區域、極限區域 | 2、4 |
| §6.4 停在蓋子上 = 未落入戰鬥區 | 2（`zone_of`）、7、8 |
| §10 持續優化 | 16 |
| §11 成功標準 | 16 |
