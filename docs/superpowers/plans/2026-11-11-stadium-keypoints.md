# 場地關鍵點模型 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 訓練 10 點的場地關鍵點模型，以可見點求投影變換、換算區域，讓自動找場地能把袋區定位到 8 px 內。

**Architecture:**
1. 從 434 支影片抽 150 格，加上 10/14 範本格，以「只靠軌道」的結果投影範本點作預標。
2. 在本機既有的 Label Studio 開新專案，由使用者標註後匯出。
3. 轉成 YOLO pose 資料集，以整支影片切分，並用 `yolo26n.pt` 主幹訓練。
4. 預測點經 RANSAC 求出投影變換，換算區域，再用軌道交叉驗證，最後接入 `find_stadium`。

**Tech Stack:** ultralytics 8.4.183（YOLO pose）、OpenCV、本機 Label Studio（8081，Django ORM 建專案，比照 `.superpowers/sdd/training-hard-cases-v2-20261031/import_followup12.py`）、pytest。

**Spec:** `docs/superpowers/specs/2026-11-11-stadium-keypoints-design.md`

## Global Constraints

- 10 點的順序與名稱固定：`notch_left, notch_right, xtreme_left, xtreme_right, over_left_inner, over_right_inner, field_left_upper, field_left_lower, field_right_upper, field_right_lower`。
- 翻轉時互換的成對點：`flip_idx = [1,0,3,2,5,4,8,9,6,7]`。
- 只標看得到的點；被擋住、在畫面外或看不清的不標。預標不算人工答案。
- 原始影片、`calib/`、既有 Label Studio 專案與匯出檔一律唯讀。新資料放 `data/detector-hard-cases-20261030/stadium-keypoints-20261111/`，這是 Label Studio 允許讀取的資料夾。
- 測試一律用 `tmp_path` 副本，不得以真實的受保護路徑執行會寫入的程式（2026-11-11 事故）。
- 以整支影片切分 train／val／test = 80／10／10；test 凍結到最後。
- 不下載權重；GPU 訓練前確認沒有其他 GPU 工作。
- Commit 以 `git commit -o <paths>` 只提交自己的檔案，結尾 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`。

## Review Focus

1. **左右點弄反：** 翻轉增強或匯出時把 left／right 對調，會讓 X／B 與 Over 袋對調。→ K3 測試 `test_flip_idx_swaps_left_right_pairs`、`test_export_keeps_point_identity`。
2. **只有一側的點：** 例如只看到左邊 4 點時，投影變換無法約束另一側。→ K4 測試 `test_points_on_one_side_return_none`。
3. **點標在錯的特徵上：** 偶發的離群點必須被 RANSAC 排除，或使結果被拒絕。→ K4 測試 `test_one_wild_point_is_rejected_by_ransac`。
4. **預標被當成真值：** 使用者沒有改動的預標，不能變成訓練資料。→ K3 只匯出有人工 Submit 的任務，測試 `test_unsubmitted_tasks_are_not_exported`。
5. **同一支影片同時出現在 train 與 val／test。** → K3 測試 `test_split_never_shares_a_video`。

---

### K1：抽格、預標、建立 Label Studio 專案

**Files:** Create `scripts/keypoint_labeling.py`、`config/label_studio_keypoints.xml`；Test `tests/test_keypoint_labeling.py`

**Interfaces:**
- `KEYPOINTS: tuple[str, ...]`（10 個名稱）
- `FLIP_IDX`
- `sample_keypoint_frames(videos: list[Path], n: int, seed: int) -> list[FrameRef]`：分層抽樣，每支影片最多 2 格，避開前 0.5 秒。
- `prelabel_points(frame, model, cfg, template_points) -> dict[str, tuple[float, float]]`：只靠軌道的找場地成功時，投影範本點；失敗則回傳空預標。

步驟：
- [ ] **測試：** 分層後每批都有、每支影片至少 1 格且最多 2 格、同一 seed 的結果相同；預標只在套合成功時產生；XML 內含 10 個 `KeyPointLabels` 與 1 個 `stadium` 框。
- [ ] **實作與輸出：**
  - 腳本 `sample` 會輸出 150 格與 10/14 範本格（task 0，作為範本參考點），寫入 `images/`、`tasks.json`、`manifest.json`（含來源 SHA、影格號與批次）。
  - 腳本 `create-project` 以 ORM 建專案「BXJUDGE 場地關鍵點 150 張」。
  - 建立後確認每張圖都回 HTTP 200，且 bytes 與來源相同。
- [ ] 提交。停下等使用者標註。

### K2：範本參考點

- [ ] 從匯出檔取 task 0 的 10 點，存為 `config/stadium_keypoints_reference.json`，記錄範本 SHA。缺任一點則拒絕。
- [ ] 測試：讀寫往返；缺點時拋錯。

### K3：匯出轉 YOLO pose 資料集

**Files:** Create `src/bxjudge/stadium_keypoints.py`（資料集部分）；Test `tests/test_stadium_keypoints.py`

- [ ] **測試：**
  - `test_flip_idx_swaps_left_right_pairs`；
  - `test_export_keeps_point_identity`；
  - `test_unsubmitted_tasks_are_not_exported`；
  - `test_split_never_shares_a_video`；
  - 沒標的點寫成 `0 0 0`（可見性 0）。
- [ ] `export_pose_dataset(export_json, images_dir, out_dir, seed) -> Path`：
  - 輸出 `images/{train,val,test}`、`labels/...` 與 `data.yaml`（`kpt_shape: [10, 3]`、`flip_idx`）；
  - 不含 task 0；
  - 另存 `split.json` 與 `manifest.json`。

### K4：訓練與「點 → 區域」

- [ ] **訓練：** `scripts/train_stadium_keypoints.py`。
  - 以 `yolo26n-pose.yaml` 載入 `weights/yolo26n.pt` 主幹。
  - imgsz 960、seed 42，epochs 先 150。
  - 輸出 `runs/stadium-keypoints-<date>/`。
  - 回報 val 的每點誤差與可見性。
- [ ] **測試：**
  - `test_points_on_one_side_return_none`；
  - `test_one_wild_point_is_rejected_by_ransac`；
  - `test_exact_points_recover_zones`（< 1 px）；
  - `test_fewer_than_four_points_return_none`。
- [ ] **實作 `zones_from_keypoints(points: dict[str, (x, y, conf)], reference, model, frame_size, cfg) -> StadiumFit | None`：**
  - 只用信心 ≥ `kp_min_conf` 的點；
  - `cv2.findHomography(RANSAC, kp_ransac_px)`，內點至少 4 點，且左右、遠近都要有；
  - 檢查重投影誤差、`_zones_valid` 與透視上限；
  - 最後以 `_score` 做軌道交叉驗證。
  - 新門檻放 `config/stadium_finder.yaml`。

### K5：接入找場地與評估

- [ ] `find_stadium` 優先用關鍵點；關鍵點失敗時才退回「只靠軌道」；兩格一致的規則不變。
- [ ] 以 `scripts/eval_stadium_finder.py` 重跑 434 支，報告每批找到率與 val／test 上的區域誤差。
- [ ] Codex 唯讀審查，修正後再提交。
