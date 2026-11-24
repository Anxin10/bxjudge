# BXJudge 競爭專案技術差距分析與分階段改善計畫

> 日期：2026-11-24　｜　審查方式：只讀原始碼、測試與資料，不執行競品程式或權重，不修改 BXJudge 正式程式碼、權重或部署設定。
>
> **BXJudge 基準**：`feat/local-judge` commit `95f4d9a`。本文的行號都以 `git show HEAD:<path>` 為準。
> 工作目錄另有**尚未提交**的修改（見 §0.3），行號可能和工作目錄版本不同。
>
> **競品快照**（`git clone --depth 1`，大於 2 MB 的檔案沒有下載）：
>
> | 專案 | commit | 日期 |
> |---|---|---|
> | Trafitto/open-beybladex-ar-core | `c07bf4b` | 2026-05-19 |
> | Efradgalio/Beyblade-Battle-Video-Analysis-System | `fcee2dd` | 2025-06-02 |
> | ds-raw/BeybladeDetection | `10a5925` | 2024-09-21 |
> | ihsanabdulhakim/beyblade-battle-tracking | `64581aa` | 2024-09-30 |
> | AndreaBContarini/Computer_Vision_Beyblade | `0ab8ec5` | 2025-01-13 |
> | ammarmuflih/AssistX_Assessment | `9588835` | 2025-06-03 |
> | RaffelRavionaldo/Beyblade-Battle | `89f13e6` | 2024-04-10 |

---

## 0. 重點結論

### 0.1 五個關鍵發現

1. **競品都是單一腳本的原型，BXJudge 在工程與證據嚴謹度上大幅領先。**
   - 7 個競品 repo 合計 **0 個測試檔**。
   - 7 個都**沒有 LICENSE**：repo 內沒有授權檔，GitHub License API 也回傳 404。依法預設保留所有權利。
   - 只有 4 個 Roboflow 資料集在 `README.dataset.txt` 或 `data.yaml` 標示 `CC BY 4.0`。
   - BXJudge 在相關領域有 387 個測試：
     | 領域 | 測試數 |
     |---|---|
     | calibration | 113 |
     | launch | 86 |
     | finish | 61 |
     | two_bey | 48 |
     | reid | 36 |
     | identity | 30 |
     | collision | 6 |
     | timebase | 7 |
   - 可借鏡的只有少數**演算法概念**，而且都必須無塵室重寫。

2. **identity-test 目前的瓶頸不在運動模型。**
   - 3 支 identity-test 影片（`20261024_201331`、`_201410`、`_201444`）在 v7 端到端測試中**全部判 `redo`**（`reports/v7-120fps-eval-20261124.md` §5）。
   - 原因是 `calibration_geometry_missing`、發射器漏框，以及 `round_releases={}`，都不是 ID 交換。
   - 在這 3 支影片能產出勝負之前，任何追蹤改善都無法反映在勝方正確率或 Finish Macro-F1 上。

3. **要求的 8 項指標中，3 項目前沒有真值，也沒有計算程式。**
   - IDF1、HOTA、ID Switch：整個 `src/`、`config/` 搜不到 `idf1|hota|idsw|mota`，也沒有逐幀身份真值。
   - 停轉時間誤差：`config/evaluation.yaml` 只有 `truth_t_launch_tolerance_s`，沒有停轉時間真值。
   - 所以 **E0（建立量測）必須排在所有改動之前**。

4. **Axis 沒有任何可查核的技術資料。** [inertia.cx/products/axis.html](https://inertia.cx/products/axis.html) 是行銷頁：
   - 寫了「60-120 FPS capture」「<10 min calibration」，以及 Spin／KO／Burst／Extreme 偵測；
   - 沒有運動模糊、晃動、位置穩定性或 X-Line 校準的描述，也沒有準確率數字；
   - 文件頁的 FAQ 顯示「Coming Soon」。
   - 所以本文中所有 Axis 方法都標 `UNVERIFIED`，只當作題目指定的參考概念。

5. **E1（圓弧預測）已經在量測基準建立前開始實作。** 工作目錄的 `two_bey_tracking.py` 有 +129 行未提交修改，`circular_prediction_enabled` **預設是 `true`**（§0.3）。建議在 E0 完成前把預設改成 `false`，避免基準被污染。

### 0.2 前一版草稿需要更正的地方

| 原本的說法 | 實際查核結果 |
|---|---|
| `Beyblade-Battle` 只有影格擷取（`extract_pict.py`） | 判定邏輯在 `run.ipynb` cell 0：手動框選區域（`create_region`）、`check_spin_status`（註解寫光流）、BoT-SORT 追蹤、輸出勝者圖 |
| `Computer_Vision_Beyblade` 有 Spin/Still 分類器 | `spin_still_detection.py` 是 Colab 匯出檔，內容是把 `SPINNING`／`STILL` 當成 **YOLOv8 偵測類別**訓練（L12–13、L84、L102），不是時序分類器 |
| 預期改善「ID Switch −20~35%」「IDF1 +3~5%」等 | 沒有任何實測依據，本版改為只寫預期方向，數字要由 E0 量出來 |
| 計畫以「`120fps-identity-test`」量 IDF1／HOTA | 這個名稱在 repo 中只出現在草稿本身；實際對應的是 3 支端到端影片，以及 `tests/test_identity*.py`、`tests/test_collision_identity*.py` 的單元測試，都沒有逐幀身份真值 |

前一版草稿已另存為 `reports/competitive-gap-analysis-draft1-20261124.md`。

### 0.3 工作目錄現況（未提交，基準仍為 `95f4d9a`）

| 檔案 | 變更 | 來源 |
|---|---|---|
| `src/bxjudge/two_bey_tracking.py` | +129 行 | 新增 `_fit_circle_lsq`、`_predict_circular_arc`、`_clamp_to_rim`；`TwoBeyTracker(..., arena_circle=None)` |
| `config/two_bey_tracking.yaml` | +6 行 | `circular_prediction_enabled: true` 等 5 個參數 |
| `tests/test_two_bey_tracking.py` | +56 行 | E1 測試 |
| `src/bxjudge/launch_phase.py`、`config/launch_phase.yaml`、`tests/test_launch_phase.py` | — | v7 報告 Phase 5 的 120 FPS 發射器容錯修改（`launcher_gap_s`、`prefix_wait_s`） |

本次分析沒有動到上表任何檔案。

---

## 1. 第一階段：BXJudge 現況盤點（12 項）

狀態定義：
- `IMPLEMENTED`：有程式碼也有測試；
- `PARTIAL`：有部分實作；
- `MISSING`：程式碼不存在；
- `UNVERIFIED`：無法用程式碼或資料確認。

### 1. 場地校準時序記憶、短暫遮擋與手動校準備援 — `IMPLEMENTED`

**實作：**
- `calibration_store.py`：
  - `Snapshot`（L67）有 manual、auto、keypoints 三種校準快照（L59）；
  - 會依影片第一秒的畫面挑選校準，手動校準優先；
  - reviewer 可以 veto 自動產生的校準，veto 記錄在 `calibration_vetoes.json`（模組說明 L24–25）。
- `dynamic_calibration.py`：
  - `DynamicCalibration`（L69）處理固定機位；
  - 固定機位每約 1/15 秒檢查一次（L91–92）；
  - 短暫模糊容忍 `_blur_grace_s = 0.5`（L95、L311–323）；
  - 遮擋時以 `bridged` 狀態補位（L256）。
- `normalize_tracks`（L450–461）和 `recorded_calibration`（L553–555）都把 `bridged` 當成**沒有幾何**。
- `pipeline.py` L727–739：判決前任一幀沒有幾何或是 `bridged`，就加上 `calibration_geometry_missing`，不做確定判決。

**測試：** calibration 相關 6 個檔、113 個測試。

**限制（實測）：** 3 支 identity-test 影片尾端各有約 0.07 秒的遮擋，會因此觸發 `redo`（v7 報告 §5.1）。也就是政策很安全，但**覆蓋率不足**，而且沒有統計 bridged 比例對 redo 率的影響。

### 2. X/B Player ID 與 Track ID 是否獨立 — `IMPLEMENTED`

**實作：**
- `track.py`（`track_frames`，L15–80）：每個類別各自跑 ByteTrack，產生 `native_id`。
- `two_bey_tracking.py`（HEAD）：
  - `TwoBeyTracker`（L138）維護 `entity_id`；
  - `_own`（L239）處理 `native_id` 的歸屬；
  - `identity_unknown_since`／`identity_unknown_intervals`（L161–162）和 `identity_unknown_at`（L119）明確記錄身份不確定的區間。
- `identity.py`：`IdentityAnchors`（L92）、`anchor_sides`（L106）、`inherited_sides`（L276）把 entity 對應到 X／B。

**測試：** identity 4 個檔、30 個測試；two_bey 5 個檔、48 個測試。

**對照：** 競品都直接把 BoT-SORT／ByteTrack 的 ID 當成玩家（例如 Efradgalio `beyblade_tracker.py` L58 `assign_stable_id`、L222–226），沒有「身份不確定」這個狀態。

### 3. HSV、外觀 Embedding、ReID 與身份恢復 — `PARTIAL`

**實作：**
- `reid_appearance.py`：
  - `describe_crop`（L99）用整框 3D HSV 直方圖（`calcHist`，L142）、梯度方向直方圖，以及 `laplacian_variance`（L159，只記錄品質）；
  - `distances`（L166–182）用 Bhattacharyya 距離。
- `two_bey_tracking.py`：
  - `_reanchor`（L246）結合運動與外觀投票，沒有共識就標為 unknown；
  - `_eliminate`（L198）處理只剩一條 tracklet 的情況。

**測試：** reid 7 個檔、36 個測試。

**限制：**
- 沒有光照正規化：HEAD 全庫搜不到 `CLAHE`；
- 沒有反光遮罩：搜不到 `glare`、`specular`；
- 沒有只取中心區域的色相特徵；
- 沒有深度學習的外觀 embedding。

### 4. Kalman 圓弧預測、速度衰減與物理運動約束 — `PARTIAL`

**HEAD 的實作：**
- `_create`（L173–174）用 `cv2.KalmanFilter(4, 2)` 線性常速模型；
- `_reanchor`（L246 起）以線性外推計算運動代價。

**工作目錄：** 圓弧預測和邊界夾限已經實作（§0.3），但**還沒有失鎖時的速度衰減**，也還沒有任何量測結果。

**限制：** 碰撞後的軌跡不是圓弧，競品的做法也只在殘差夠小時才採用圓弧（見矩陣第 4 項）。

### 5. 離線 Bidirectional Tracking — `MISSING`

- `two_bey_tracking.py` L329–330 的 `update` 要求 `frame > self.previous`；
- `track.py` L33–35 也一樣，只能往前追蹤。
- 全庫唯一的 backward 是 `temporal_geometry.py` L381 對**場地幾何**做的 LK 前後向檢查，不是用在追蹤。
- 競品也都沒有這個功能。

### 6. Frame Differencing、Motion Blur、Optical Flow、Spin/Still — `PARTIAL`

**實作：** `finish.py`
- `spin_activity`（L80–86）：64×64 灰階的 `absdiff`；
- `rotation_rate`（L91–109）：在內切圓上用 Farneback 光流擬合角速度 ω；
- `orbit_direction`（L120）；
- `stop_frame`（L160）。

**測試：** finish 2 個檔、61 個測試。

**限制：**
- 停轉判定沒有用到模糊程度：`laplacian_variance` 只用在 ReID；
- 沒有用到位置穩定性或晃動；
- 沒有停轉時間真值，無法算誤差。

### 7. 碰撞偵測、碰撞前後速度及方向變化 — `PARTIAL`

- `launch_check.py`：`_air_collision_evidence`（L185）和 `detect_air_collision`（L212）只處理**落地前的空中碰撞**。
- `finish.py`：`id_swap_risk`（L195）只看兩個框連續接觸幾幀。
- 對戰中的碰撞事件（接近速度、速度反轉）在 `pipeline.py` 和 `rules.py` 都**沒有**；`rules.py` 只有 `mid_air_collision`（L23、L185）。

### 8. Blade、Ratchet、Bit 零件級 Burst 辨識 — `PARTIAL`

- `detector_config.py` L11：`CLASSES = ("bey", "bey_part", "launcher")`，沒有區分零件種類。
- `config/rules.yaml` 有區分脫落的 barrel（不算 burst）的規則說明。
- **前置問題（實測）：** `bey_part` 本身還不可靠。
  - v7 在 `v7/test` 上 bey_part 為 0 TP、67 FP（v7 報告 §3.1）；
  - 偵測器報告也顯示誤報都打在放在場外的 bey 中心（`reports/120fps-detector-20261124/REPORT.md`）。

### 9. Rail Mask、Dome Glare Mask、場地座標正規化 — `PARTIAL`

- `stadium_finder.py`：`rail_mask`（L169）擷取軌道遮罩；`_refine`（L392）、`_ecc`（L420）做精細對齊；`_zones_valid`（L495）檢查區域。
- 用 Homography 把場地正規化成 8 區，`xtreme` 和 over 口袋判定在 `finish.py` L328、L389–395。
- **沒有** dome 反光遮罩。

### 10. 120fps／240fps 時間門檻一致性 — `IMPLEMENTED`

- `timebase.py`：
  - `Timeline`（L23）；
  - `probe_timeline`（L90）；
  - `_FRAME_MINIMA`（L118）；
  - `JudgmentTime`（L147–161，取時間換算的格數和最小格數兩者中較大的值）。
- `config/physical_time.yaml` 以 240 FPS 為基準；`config/video_sources.yaml` 定義 `capture_fps: 240, playback_fps: 30` 的慢動作來源。
- **測試：** `tests/test_timebase.py` 有 7 個測試，L80 參數化測試了各種 fps 的格數換算。
- **注意：** v7 報告 Phase 5 新加的 `launcher_gap_s`、`prefix_wait_s` 也依 fps 換算格數，符合這個契約。

### 11. 終結候選事件的局部高解析度重分析 — `PARTIAL`

- **偵測器層級沒有：** HEAD 的 `src/` 搜不到 ROI 裁切後重新推論、tile、SAHI 這類機制。`pipeline.py` L1079 的 `needs_reanalysis` 是校準改變後整支影片重跑。
- **有 VLM 層級的事件覆核：**
  - `vlm.py` L56 把事件前後的取樣畫面送給 VLM；
  - `vlm_events.py`（`request_events` L135、`corroborate_events` L172）會和追蹤結果交叉驗證；
  - `vlm_judgment.py` 規定身份不明時必須回 `unknown`。
  - 但這些都是覆核，不會提高偵測器在終結瞬間的定位精度。

### 12. 同一測試集上的競爭模型評估 — `UNVERIFIED`

- **類別不相容：**
  | 專案 | 類別 |
  |---|---|
  | Efradgalio | `beyblade`、`broken beyblade` |
  | ds-raw | `Beyblade`、`hand` |
  | ihsan | `'0'`、`'1'`、`'2'` |
  | AssistX | `Beyblade` |
  | CV_Beyblade | `SPINNING`、`STILL` |
- **刻意沒有執行競品權重：** 競品權重是 Ultralytics `.pt`，載入時必須 unpickle，等於執行來源不明的程式碼。
- 安全的替代做法見 E8。

---

## 2. 第二階段：技術比較矩陣

競品路徑都以上方快照 commit 為準；BXJudge 路徑以 HEAD `95f4d9a` 為準。授權欄的 `NO_LICENSE` 代表只能參考概念、無塵室重寫，不能複製程式碼。

| # | 技術項目 | 競品與實際原始碼路徑 | BXJudge 檔案:行號 | 狀態 | 現有限制 | 可移植的方法 | 預期改善指標 | 成本／風險 | 授權 | 優先級 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 校準時序記憶與備援 | open-beybladex `arena.py:12 setup_arena_roi`、`:91 _try_reference_detection`、`tracker.py:670 auto_detect_roi_from_reference`、`:789 auto_detect_roi_hough`；Beyblade-Battle `run.ipynb` cell 0 `create_region`（手動點選）；Axis「<10 min calibration」(`UNVERIFIED`) | `calibration_store.py:59,67`；`dynamic_calibration.py:69,91-95,256,311-323,450-461`；`pipeline.py:727-739` | IMPLEMENTED | 短暫遮擋就整局 redo（identity-test 3/3） | 競品只有一次性的 ROI，沒有比 BXJudge 好的方法；應改善的是 bridged 的**覆蓋統計與分級**，而不是放寬政策 | redo 率、人工覆核率 | 低／低 | 不需移植 | **P0**（見 E1） |
| 2 | X/B 與 Track ID 獨立 | Efradgalio `beyblade_tracker.py:58 assign_stable_id`、`:222-226`；ds-raw `beyblade_battle_tracker.py:189`；ihsan `main.py:201` | `track.py:15-80`；`two_bey_tracking.py:119,138,161-162,239`；`identity.py:92,106,276` | IMPLEMENTED | — | 不需移植，BXJudge 領先 | — | — | — | P2 |
| 3 | HSV／外觀／ReID | open-beybladex `preprocess.py:21 preprocess_frame_hsv`（V 通道 CLAHE，`config.py:198-199`）、`tracker.py:1094 _sample_center_color`（中心晶片飽和像素的色相中位數）、`:1160 _build_cost_matrix`（距離加權色相）；CV_Beyblade `BEY.py:137,144,209`（直方圖／平均色配對） | `reid_appearance.py:99,142,159,166-182`；`two_bey_tracking.py:198,246` | PARTIAL | 沒有光照正規化，整框直方圖容易被反光與背景污染 | V-CLAHE 前處理加中心區色相特徵，**只當成附加特徵**，仍維持「沒有共識就 unknown」 | ID Switch↓、reanchor 召回↑、`identity_unknown` 時長↓ | 低／低（要防止 CLAHE 放大雜訊） | NO_LICENSE | P1 |
| 4 | 圓弧預測、速度衰減、物理約束 | open-beybladex `tracker.py:25 _fit_circle_lsq`、`:76 _clamp_velocity_to_rim`、`:106 _predict_circular`（殘差 >15% 就退回線性）、`:178 kalman_predict`（`KALMAN_MAX_PREDICTION_DRIFT` 限制漂移）、`config.py:358 KALMAN_LOSS_VELOCITY_DECAY=0.5`；CV_Beyblade `BEY.py:103 init_kalman`（filterpy 線性） | HEAD `two_bey_tracking.py:173-174`；工作目錄已加入圓弧預測與邊界夾限（未提交） | PARTIAL | 線性外推；工作目錄版本還沒有失鎖衰減，也沒有量測 | 圓弧僅在殘差達標時作為關聯閘門，加上失鎖衰減；**預設關閉**，由 E3 量測後再決定 | ID Switch↓（碰撞以外的短暫遮擋） | 低／中（碰撞後軌跡不是圓弧，可能誤接） | NO_LICENSE | P1 |
| 5 | 離線雙向追蹤 | 競品都是單向逐幀處理 | `two_bey_tracking.py:329-330`；`track.py:33-35`；只有幾何用了前後向：`temporal_geometry.py:381` | MISSING | 管線本來就是離線處理，卻沒用到碰撞後的清楚軌跡 | 對 `identity_unknown` 區段做反向追蹤，正反兩個方向一致才恢復身份 | IDF1↑、`identity_unknown` 時長↓、人工覆核率↓ | 中／中 | 自研 | **P0** |
| 6 | Frame Diff／Blur／Flow／Spin-Still | AssistX `run.py:98-100 calculate_blur_score`（Laplacian 變異數）、`:133 motion_threshold=5500`、`:103-128 filter`（連續 6 幀遲滯）；ds-raw `beyblade_battle_tracker.py:61-105`（框內 absdiff 平均 <1 視為停止）；ihsan `main.py:311-318`（位置差 <1 且外觀差 <0.2）；Efradgalio `beyblade_tracker.py:186-193`（30 幀）、`:153,175` 停轉判定標明「not yet implemented」；Beyblade-Battle `run.ipynb` cell 0 `check_spin_status`；CV_Beyblade `spin_still_detection.py:12-13`（YOLO 類別）；Axis 模糊／晃動／位置穩定性 (`UNVERIFIED`) | `finish.py:80-86,91-109,120,160` | PARTIAL | 停轉判定只用 absdiff 和光流；沒有停轉時間真值 | 把 Laplacian 模糊、中心位置變異（晃動）當成**輔助特徵**，依 `JudgmentTime` 換成物理秒數後做遲滯；只能延後或標註，不能單獨判定停轉 | 停轉時間誤差↓、spin 的 Finish F1↑ | 低／低 | NO_LICENSE（Laplacian 是公知方法） | P0（等 E0 有停轉真值後） |
| 7 | 對戰碰撞偵測 | open-beybladex `physics.py:35 compute_closing_speed`、`:70 has_velocity_reversal`、`:96 detect_wall_collisions`、`:122 CollisionDetector`；CV_Beyblade `v2/collision.py:3-96`（IoU 加 Kalman 偏差）；ds-raw `:107-135`、ihsan `main.py:134,327`、Efradgalio `:119-134`（都只用重疊） | `launch_check.py:185,212`；`finish.py:195`；`rules.py:23,185` | PARTIAL | 只有空中碰撞，以及用框接觸判定交換風險 | 接近速度、速度反轉、Kalman 殘差突增，用來**切分身份風險區段**；不作為計分證據 | `id_swap_risk` 誤報↓、ReID 區段精度↑ | 低／低 | NO_LICENSE | P1 |
| 8 | 零件級 burst | Efradgalio `data.yaml`（`broken beyblade`，CC BY 4.0 資料集）；其餘都沒有 | `detector_config.py:11`；`finish.py` burst 判定；`config/rules.yaml` | PARTIAL | `bey_part` 本身就不可靠（v7 test 0 TP／67 FP） | **先修 bey_part**：統一定義、補標、處理場外桌面的誤報；細分零件放到之後 | burst 精確率↑ | 高（標註）／中 | 資料集 CC BY 4.0（需署名） | P2 |
| 9 | Rail／Dome Glare 遮罩 | open-beybladex `tracker.py:1305 _build_dome_mask`（V ≥ v_min 且 S < s_max，`config.py:299-301` 預設 185／65，可加扇形排除）、`:530 build_rail_mask` | `stadium_finder.py:169,392,420,495`；`finish.py:328,389-395` | PARTIAL | 沒有反光遮罩 | High-V／Low-S 遮罩只用在 ReID 取樣與對齊打分時排除像素 | 強光片段的 ReID 穩定度↑ | 低／低 | NO_LICENSE | P1 |
| 10 | 120／240 fps 時間一致 | 競品都用固定格數（AssistX `:105` 6 幀、Efradgalio `:190` 30 幀） | `timebase.py:23,90,118,147-161`；`config/physical_time.yaml`；`tests/test_timebase.py:80` | IMPLEMENTED | 新門檻必須持續遵守這個契約 | 不需移植；E1–E7 的新門檻一律用毫秒 | 跨 fps 一致 | — | — | P2 |
| 11 | 終結候選的局部重分析 | 競品都沒有；Axis「instant replay」(`UNVERIFIED`) | `vlm.py:56`；`vlm_events.py:135,172`；`vlm_judgment.py`；`pipeline.py:1079`（只有整支重跑） | PARTIAL | 偵測器解析度固定；畫面邊緣的小型 launcher 是已知的漏框和定位誤差來源 | 對終結候選前後 ±0.25 秒和畫面邊緣的 launcher，裁切 ROI 後用原解析度重新推論，只用來**補足證據或觸發覆核** | Finish F1↑、`finish_unknown` 覆核↓、launcher 召回↑ | 中（只在候選窗口推論）／低 | 自研 | **P0** |
| 12 | 同測試集競品評估 | 各 repo 的 `.pt` 權重與 `data.yaml`；Roboflow 資料集（CC BY 4.0） | `evaluate.py`；v7 報告 §3 | UNVERIFIED | 類別不相容；權重需要 unpickle，有安全風險 | 不載入競品權重，改成在**隔離容器**跑，或用 CC BY 資料集的 test split 評估 BXJudge（bey 類別對應） | 只當參考基準 | 低／低（隔離後） | 權重 NO_LICENSE、資料 CC BY 4.0 | P2 |

---

## 3. 第三階段：分階段改善計畫與消融實驗

### 3.1 護欄（全部實驗都適用）

- **不替換**以下元件，所有新增功能都是可關閉的設定開關：
  - RF-DETR、YOLO26、ByteTrack、YAML 規則引擎；
  - `weights/selected.json`（目前是 `train-rfdetr-20261023b`；v7 因 launcher 驗收沒過，沒有部署）。
- **軟性預測不能當證據：** 圓弧預測、雙向縫合、模糊分數、ROI 重推論只能用來：
  - 輔助關聯配對；
  - 延後判定；
  - 新增 `review_reasons`。
  它們不能單獨產生計分，也不能把 `identity_unknown` 指派成 X／B。
- **資料集固定不變，避免洩漏：**
  - `v7/test`（68 張，2026-11-24 已修正標註）；
  - `dataset-120fps-701-20261124/val`（96 張）；
  - identity-test 的 3 支影片。
  E0 新標的真值只用於評估，不能進入訓練。
- **時間一律用物理毫秒，經 `JudgmentTime` 換成格數。**

### 3.2 指標與資料可得性

| 指標 | 現在能量嗎 | 缺什麼 | 由誰補 |
|---|---|---|---|
| IDF1、HOTA、ID Switch | **不能** | 逐幀的身份真值，以及計算工具 | E0-a、E0-b |
| 停轉時間誤差 | **不能** | 每局敗方的停轉時間真值 | E0-c |
| Finish Macro-F1、勝方正確率 | 可以，但 identity-test 是 3/3 redo，樣本實際上是 0 | identity-test 要先能產出判決；另需要更多已審核回合 | E1，加上 `evaluate.py` 的 truth |
| 人工覆核率 | 可以 | — | `evaluate.py`（`unflagged_accuracy`、`review_reasons`） |
| 端到端延遲 | 部分可以 | 已有 TRT 單幀 13.84 ms／72 FPS（v7 報告 §4）；端到端要用 `evaluate.py` 的 `speed_gate`、`_wall` | E0-d |

### 3.3 實驗順序

| 代號 | 改動（最小侵入，可關閉） | 模組 | 主要指標 | 通過門檻 |
|---|---|---|---|---|
| **E0** 量測基準 | (a) 標註 identity-test 3 支影片的身份真值：每 0.25 秒一個關鍵幀，加上碰撞前後各 ±0.5 秒的逐幀標註，估計約 400 幀；(b) 新增唯讀評估工具，計算 IDF1、ID Switch、HOTA，用 TrackEval 演算法自行實作或在隔離環境使用；(c) 標註停轉與終結時間；(d) 以 production 和 v7 候選模型各跑一次，記錄全部 8 項指標 | 新增 `tools/` 或 `evaluate.py` 擴充；不改判決路徑 | 全部 | 基準數字入庫 |
| **E1** 校準缺口分級 | 統計 `bridged` 幀在判決窗口的位置與長度；只針對「固定機位、遮擋 ≤ N 毫秒、前後對齊殘差一致」的情況，研究能否改為**附條件可用**，否則維持 redo。不改預設，先產出報告 | `dynamic_calibration.py`、`pipeline.py`（只統計） | redo 率、勝方正確率 | 0 個新增錯判 |
| **E2** 離線雙向縫合 | 對 `identity_unknown` 區間從區間尾端反向跑 `TwoBeyTracker`；正反兩個方向和外觀都一致才恢復，否則維持 unknown | `two_bey_tracking.py` | IDF1、`identity_unknown` 時長、覆核率 | 誤恢復 = 0 |
| **E3** 圓弧關聯閘門 | 使用工作目錄中的實作，預設改為 `false`，再加上失鎖速度衰減；只用在線性和圓弧預測都在閘門內的配對 | `two_bey_tracking.py` | ID Switch（排除碰撞區段後） | IDF1 不下降 |
| **E4** 停轉多特徵 | `stop_frame` 加入 Laplacian 模糊與中心位置變異（毫秒遲滯）作為延後或覆核條件 | `finish.py` | 停轉時間誤差、spin 的 F1 | 誤差中位數下降，F1 不下降 |
| **E5** CLAHE＋反光遮罩＋中心色相 | 加在 ReID 取樣前處理 | `reid_appearance.py` | reanchor 召回、ID Switch | 誤恢復 = 0 |
| **E6** 對戰碰撞事件 | 接近速度、速度反轉、殘差突增，作為交換風險區段的邊界 | `finish.py`、`two_bey_tracking.py` | `id_swap_risk` 誤報、覆核率 | 交換風險召回不下降 |
| **E7** ROI 原解析度重推論 | 終結候選 ±0.25 秒，以及畫面邊緣的 launcher，裁切後用原解析度再推論一次 | `pipeline.py`（新增可關閉的步驟） | Finish F1、launcher 召回、延遲 | 延遲增加 ≤ 15% |
| **E8** 外部基準（選做） | 在隔離環境跑競品權重，或用 CC BY 資料集的 test split 評估 BXJudge 的 bey 類別 | 唯讀 | bey 召回 | 只當參考 |

**建議順序：** E0 → E1 → E2 → E7 → E4 → E3 → E5 → E6 → E8。理由：
- E1 和 E7 直接對應 identity-test 的實際失敗原因：校準缺口，以及畫面邊緣的 launcher；
- E2 是離線管線特有的優勢，風險可以控制；
- E3 的效益受碰撞後軌跡不是圓弧限制，排在後面；
- E4 需要 E0-c 的停轉真值。

### 3.4 驗收與回滾

- 每個實驗都必須：
  1. 用獨立的設定開關，預設 `false`；
  2. 通過完整 `uv run pytest`（目前是 1978 passed、4 xfailed）；
  3. 在 E0 的基準上**單獨開啟**做消融，最後再測整體組合。
- 出現下列任一情況就關閉該開關並回滾：
  - 誤恢復或錯判大於 0；
  - 覆核率上升卻沒有換來正確率；
  - 端到端延遲超出預算；
  - 不確定的身份被指派成 X／B。

---

## 附錄 A：競品一覽（實際程式碼查核）

| 專案 | 偵測 | 追蹤 | 停轉判定 | 碰撞 | 勝負 | 測試 | 授權 |
|---|---|---|---|---|---|---|---|
| open-beybladex-ar-core | Hough 圓偵測＋輪廓，HSV CLAHE | 自製的 Kalman、圓弧預測、色相配對 | **沒有**（AR 特效系統，不判勝負） | `physics.py` 接近速度、反轉 | 無 | 0 | 無 |
| Beyblade-Battle-Video-Analysis-System | YOLO11（beyblade／broken） | BoT-SORT | 位移 30 幀（L186–193）；說明寫「not yet implemented」（L153） | 距離／IoU（L119–134） | broken 或停止 | 0 | 無（資料集 CC BY） |
| BeybladeDetection | YOLO（Beyblade／hand） | BoT-SORT | 框內 absdiff 平均 <1（L61–105） | 重疊比例（L107–135） | 另一顆停止 | 0 | 無（資料集 CC BY） |
| beyblade-battle-tracking | YOLO（3 類） | Ultralytics 預設 | 位置差 <1 且外觀差 <0.2（L311–318） | 重疊 ≥3%（L134、L327） | 先停者輸 | 0 | 無（資料集 CC BY） |
| Computer_Vision_Beyblade | YOLOv8（SPINNING／STILL 類別） | filterpy Kalman＋顏色配對 | 偵測類別 | IoU＋Kalman 偏差（`v2/collision.py`） | 生命值條 | 0 | 無 |
| AssistX_Assessment | YOLO11／v8（Beyblade） | ByteTrack | Laplacian 變異數 <5500，連續 6 幀遲滯（L98–133） | 無 | 持續時間較長者勝（L78–90、L186） | 0 | 無（資料集 CC BY） |
| Beyblade-Battle | YOLOv8 | BoT-SORT | `check_spin_status`（`run.ipynb` cell 0） | 無 | 輸出勝者圖 | 0 | 無 |

## 附錄 B：查核方法

- **競品：** `git clone --depth 1 --filter=blob:limit=2m`，只讀檔案，不執行任何程式、notebook 或權重。授權同時查了 repo 內的 LICENSE 檔，以及 `api.github.com/repos/<repo>/license`（7 個都是 404）。
- **BXJudge：** 一律以 `git show HEAD:` 和 `git grep HEAD` 取得行號，避免工作目錄未提交的修改造成偏差。測試數以 `^def test` 計算。
- **Axis：** 讀了 inertia.cx 的 Axis 產品頁、文件頁和運作說明頁，都沒有技術細節。
- **報告引用：** `reports/v7-120fps-eval-20261124.md`、`reports/120fps-detector-20261124/REPORT.md`。
