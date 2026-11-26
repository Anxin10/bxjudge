# BXJudge 120 FPS 時間基準與勝方阻擋：Codex 工作清單

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 由 Codex 在現有分支執行；Claude 負責審查每個任務並親自核實每條發現。

**Goal:** 讓同一套裁判程式能正確處理舊的 240 FPS 慢動作檔與新的 120 FPS 檔，並在 20 回合人工真值上找出、修掉「沒判出勝方」的第一個阻擋點。

**Architecture:** 不換模型、不改 ROI。先盤點每個時間門檻的實際物理意義，由使用者決定數值，再改成以物理時間戳判定；另一條線逐回合找出勝方被擋在哪一層（盤面對位、X／B 身分、落地、終結、規則）。

**Tech Stack:** 既有 WSL Ubuntu、uv／Python 3.12、pytest、ffprobe、RF-DETR／TensorRT 正式權重。不新增依賴。

**Spec:** `docs/spec.md`；本機規劃頁 `BXJudge_1080p120_發射後ROI_系統規劃.html`（第 3 版，第 5 節時間基準、第 9 節執行順序）。

## Global Constraints

- 分支 `feat/local-judge`，HEAD `d426130`。不新建分支或工作樹，不 push、不 merge。
- **工作樹不乾淨。** 以下是別人未提交的修改，不得覆蓋、還原或 stage：`.gitignore README.md config/vlm.yaml pyproject.toml scripts/build_reid_review.py scripts/setup_wsl.sh src/bxjudge/finish.py src/bxjudge/geometry.py src/bxjudge/web/reid_review.css src/bxjudge/web/reid_review.js src/bxjudge/workbench_server.py tests/test_geometry.py tests/test_workbench_server.py`，以及未追蹤的 `rust/`、`judge_view.py`、`bx_fx.js`、`bx_theme.css`、`judge_view.js`。只 `git add` 自己改的檔案，不用 `git add -A` 或 `git add .`。
- 受保護的未追蹤檔不動：`.agents/ AGENTS.md CLAUDE.md .claude.code-workspace antigravity.db .graftignore .graphifyignore graphify-out/ src/graphify-out/`。
- 原始影片（`~/bx/videos/*`）、人工真值、`calib/`、Label Studio 原始匯出、811 val／test 一律唯讀。新輸出另存，不覆寫舊結果。
- 規則與門檻只放在 `config/*.yaml`。證據不足維持 unknown／needs_review，不為了產生勝方放寬門檻。
- 每個程式修改：先寫會失敗的測試 → 最小修正 → 全套測試 → Claude 審查 → 修正後再測。全套基準：`d426130` 為 1658 pass／4 strict-xfail。
- 不同時跑兩個 GPU 重負載工作；啟動前確認 GPU 與 WSL 可用記憶體。
- Commit 訊息結尾：`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`。

## 已知現況（Claude 2026-11-11 核查）

- `manifest.probe()` 取 `avg_frame_rate`（無值時 `r_frame_rate`）當 `info.fps`，所有判定都用這個**播放幀率**。
- 以秒定義的門檻（`launch_pre_s`、`launch_post_s`、`launch_missing_landing_s`、`spin_dir_window_s`、`pocket_confirm_s`、`burst_part_window_s`、`burst_part_overlap_s`、`round_gap_s`、`launch_phase.yaml` 的 `held_s`／`separation_s`、`reid_appearance` 的 `window_s`）都是 `秒 × info.fps`，也就是**播放秒**。
- 以格數定義的門檻（`settle_frames`、`still_frames`、`reverse_frames`、`impact_coincide_frames`、`air_collision_frames`、`simultaneous_stop_frames`、`id_swap_overlap_frames`，以及 `launch_phase.yaml`、`two_bey_tracking.yaml`、`dynamic_calibration.yaml` 的 `*_frames`）直接用格數。
- `fps_default: 60` 只在 `_validate_thresholds` 檢查為正數，沒有其他用途。
- 影片批次：`20261001` 107 支（240 FPS 拍攝、檔案 30 FPS，慢動作 8 倍）；`20261014` 2 支（120 FPS，檔案帶 `com.android.capture.fps=120`）；新匯入 `20261024`、`20261101`、`20261108` 共 325 支（120 FPS，盤點結果見 `.superpowers/sdd/video-inventory-20261111/`）。
- 後果：舊檔的 `launch_missing_landing_s: 3.0` 等於物理 0.375 秒；套到 120 FPS 檔會變成物理 3 秒。格數門檻在 120 FPS 檔代表的物理時間是舊檔的 2 倍。

## Review Focus

1. **舊影片判決靜默改變：** 時間基準改寫後，20 回合真值切片的 `rounds.json` 判決欄位必須與改寫前逐欄相同，除非使用者核准某個門檻改值。
2. **慢動作倍率猜錯：** 檔案沒有 `com.android.capture.fps` 或與批次設定矛盾時，必須停下回報，不從檔案幀率推測。
3. **可變幀率：** 舊檔 `avg_frame_rate` 有 23–30 不等；以時間戳判定時，「持續 ≥ X 毫秒」＝最後一格與第一格的物理時間戳差 ≥ X，不能用格數乘平均間隔。
4. **空中被當落地：** D265／G555（陀螺仍在空中卻被 settle 選為落地起點）修正後，C225 等原本正確的空中例不能變成落地。
5. **新影片 Baseline 不可混入舊數據：** 120 FPS 驗證集的成績和 811 val 分開報告，不合併成一個數字。

---

### Task 1：時間門檻盤點報告（唯讀，不改程式）

**Files:**
- Create: `.superpowers/sdd/timebase-audit-20261111/thresholds.md`、`thresholds.json`

- [ ] 列出 `config/*.yaml` 每個與時間有關的鍵：名稱、檔案、目前值、單位（播放秒／格數）、使用它的函式與行號。
- [ ] 對每個鍵換算三種物理時間：舊檔（物理幀率 240、慢動作 8 倍）、10/14 檔與新檔（120）。
- [ ] 讀 `.superpowers/sdd/video-inventory-20261111/summary.json` 與 `probe.json`（Claude 已盤點：434 支都帶 `com.android.capture.fps`，`20261001` 為 240、其餘為 120；舊檔 84／107 支、新檔 24／325 支有間隔偏差 > 25% 的影格），在報告中引用，不重跑。
- [ ] 每個鍵標出「物理意義在兩種影片不同」的項目，並提出建議值（以舊檔目前的物理意義為預設），但**不改 YAML**。
- [ ] 完成後停下，交 Claude 審查；建議值由使用者決定後才進行 Task 2。

### Task 2：以物理時間戳判定（使用者核准數值後才開始）

**前置條件：** 未提交的 `finish.py`／`geometry.py`（Rust 擴充）由使用者決定提交或擱置。這個任務會改 `finish.py`，不能和別人未提交的修改混在同一個 commit。

**Files:**
- Modify: `src/bxjudge/manifest.py`、`src/bxjudge/config.py`、`config/thresholds.yaml`、`config/launch_phase.yaml`，以及 Task 1 列出的使用端
- Create: `config/video_sources.yaml`（每個批次的拍攝幀率與慢動作倍率）
- Test: `tests/test_timebase.py`

**Interfaces:**
- Produces: `VideoInfo.capture_fps: float`、`VideoInfo.slowmo_factor: float`（播放時間 × slowmo_factor⁻¹＝物理時間）；`physical_time_s(info, frame_index) -> float`；`span_at_least(timestamps_s, start, end, min_ms) -> bool`。

- [ ] 測試先寫：`span_at_least` 在 120 FPS 下 7 格（6 間隔＝50 ms）為真、6 格為假；舊檔可變幀率用實際時間戳；`video_sources.yaml` 缺批次或與檔案標籤矛盾時拋錯。
- [ ] 門檻改以毫秒（物理）定義，換算由時間戳決定；移除未使用的 `fps_default` 或改為只在 ffprobe 失敗時報錯，不作後備值。
- [ ] 回歸：用 `bxjudge eval runs/truth-slice-20261027` 與改寫前比對；判決欄位逐欄相同。另存新輸出目錄，不覆寫原 run。
- [ ] 全套測試，交 Claude 審查。

### Task 3：20 回合勝方阻擋點逐回合核查（唯讀分析）

**Files:**
- Create: `.superpowers/sdd/winner-blockers-20261111/per-round.json`、`REPORT.md`

- [ ] 輸入：`runs/truth-slice-20261027`（20 回合，15 有勝方、5 無結果）與人工 `reviewer_fix`。先讀既有輸出；需要重跑時另存新目錄，記錄 HEAD 與權重 SHA。
- [ ] 對 15 個有勝方回合，依序檢查：盤面校準／對位 → 雙方偵測 → X／B 身分 → 落地時點（對照人工 physical_floor）→ 終結事件 → 規則計分。記錄**第一個**失敗的層級、影格號與證據檔路徑。
- [ ] 5 個無結果回合也記錄系統輸出是否同樣無結果。
- [ ] 報告按層級彙總：每層擋住幾回合。
- [ ] 完成後停下，交 Claude 審查，依結果決定 Task 4 的修正範圍。

### Task 4：修正最多回合的阻擋層（預期是落地判定）

**Files:**（依 Task 3 結果確定；預期）
- Modify: `src/bxjudge/landing.py`、`src/bxjudge/launch_check.py`
- Test: `tests/test_landing.py`，加入 D265、G555、C225、E303、F417 的真實 fixture

- [ ] 先寫失敗測試：D265、G555 不得被選為落地起點；C225 維持空中；E303、F417 維持接觸盤面。
- [ ] 最小修正；新門檻放 YAML。
- [ ] 重跑 20 回合真值，回報有勝方的回合數與每回合是否和人工一致；舊判決變化逐條列出。
- [ ] 全套測試，交 Claude 審查。

### Task 5：120 FPS 偵測驗證集（抽樣與預標，不訓練）

**Files:**
- Create: `data/labeling-120fps-val-20261111/`（影像、`tasks.json`、`provenance.json`）

- [x] 從 `20261024`、`20261101`、`20261108` 抽 50–100 張，按批次與影片分層；包含發射中、碰撞、入袋、發射器在盤面上方的影格。記錄來源影片 SHA 與影格號。（已於 Project 5 完成 120 張：`20261024`×40、`20261101`×40、`20261108`×40）
- [x] 用正式模型產生預標，匯入本機 Label Studio 新專案，等使用者修正 Submit；預標不算人工答案。（120/120 已完成人工標註）
- [ ] 這批影片之後若要進訓練，必須整支影片排除在這個驗證集之外。
- [ ] 不訓練、不換正式權重。

> **抽樣與訓練補充註記（2026-11-23）**：下次處理影片時，先裁減掉大部分沒有對上戰鬥盤的片段，要訓練的部分只取剩下有效對準盤面的片段。

## 任務順序與停點

Task 1 →（停：使用者決定數值）→ Task 2；Task 3 可與 Task 1 同時進行（都是唯讀）→（停：Claude 審查）→ Task 4；Task 5 在影片盤點完成後隨時可做，需要使用者標註。ROI 與 Seg-XLarge 不在本清單，等 Task 4 有基準後再排。
