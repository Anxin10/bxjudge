# E1b 實驗報告：身份錨定與對戰前缺口容錯（2026-11-25）

> **實驗順序：** E0（真值基準） → E1（校準缺口補齊） → **E1b（本報告）** → E7 → E4 → E2 → E3 → E5 → E6 → E8  
> **基準程式碼：** Commit `95f4d9a` 基準 + E1 校準補齊 + E1b 發射空域錨定與槽位保護（`runs/e1b-eval-20261125/`）  
> **真值基準：** `runs/e0-baseline-20261124/truth/e0-truth.json`（已由使用者人工複核確認）  
> **測試影片：** `20261024_201331`、`_201410`、`_201444`（皆為 120 FPS）  
> **硬體環境：** WSL2 RTX 3090 (24GB VRAM), TensorRT

---

## 1. 核心指標對比（E0 vs E1b）

| 指標 | E0 基準 (Baseline) | E1b 實施後 | 改善幅度 |
|---|---|---|---|
| **side-IDF1** | 0.4351 | **0.9201** | **+0.4850 (+111.5%)** |
| **side_coverage**（身份覆蓋率） | 27.80% (114/410) | **85.61% (351/410)** | **+57.81 pp (從 1 支解鎖至全數指派)** |
| **side_precision_when_assigned** | 1.0000 (100%) | **1.0000 (100%)** | **保持 100% 零誤指派** |
| **side_wrong**（指派錯誤） | 0 | **0** | **0 次（零誤判）** |
| **side_abstain**（未指派） | 296 | **59** | **大幅降低 -237** |
| **ID_switches_side** | 0 | **0** | **0 次（零換身）** |
| **track_IDF1_pooled** | 0.9976 | **0.9976** | **無損（0.998）** |
| **HOTA_side_mean** | 0.4119 | **0.9007** | **+0.4888 (+118.7%)** |
| **HOTA_track_mean** | 0.9963 | **0.9963** | **無損（0.996）** |

---

## 2. 影片逐回合表現與改善分析

| 影片 | E0 身份狀態 | E1b 身份狀態 | 阻擋原因變化 |
|---|---|---|---|
| **`20261024_201331`** | 未指派（`side_coverage = 0%`）<br>`identity_unanchored@32, 33`<br>`track_missing@X, B` | **X 錨定 + B 推斷成功**<br>`identity_inferred_from_observed_pair`<br>`track_missing` 全部消除 | 身份層級完全通暢，進入終結判定層：`X_stop_reversed`、`B_stop_reversed`、`finish_unknown`（待 E4 解鎖） |
| **`20261024_201410`** | X 錨定 + B 推斷成功<br>覆蓋率 ~83% | **X 錨定 + B 推斷成功**<br>覆蓋率維持穩定 | 維持完美指派，阻擋原因落在停轉判定：`X_stop_reversed`、`B_stop_reversed`（待 E4 解鎖） |
| **`20261024_201444`** | 未指派（`side_coverage = 0%`）<br>`identity_unanchored@29, 30`<br>`track_missing@X, B` | **X 錨定成功（覆蓋率大幅提升）**<br>`identity_unanchored@29` 消除 | X 選手成功錨定，B 選手受限於發射初期邊緣偵測缺損維持 unknown（符合不強行指派原則） |

---

## 3. 根因分析與三大技術突破

### 突破 1：發射空域幾何識別（Top Border Prelaunch Entry）
- **現象**：俯視鏡頭中，陀螺在 `y=0~220` 的空中預備區進場，戰鬥區（盤底）始於 `y ≥ 220`。舊邏輯以嚴格多邊形頂部（`y=58`）判定，導致陀螺在 `y=60` 時被誤標為「已自由運動（`observed_free`）」而永久取消錨定資格。
- **解法**：在 [src/bxjudge/identity.py](../src/bxjudge/identity.py) 引入空域邊界判定：若陀螺位於戰鬥區頂部以上（`point[1] < min(battle_area)`）且水平位於左右發射區範圍內，視為發射準備進場，允許等待發射器上下文最長 `prelaunch_setup_wait_s: 1.5`；一旦進入戰鬥區（`point_in_polygon(point, c.battle_area)`）則嚴格判定為自由運動，確保不放寬對戰中安全標準。

### 突破 2：左右側發射器非對稱隔離（Lateral Launcher Decoupling）
- **現象**：舊代碼在判定 `below_bodies`（是否在發射器下方）與 `measured`（已觀測發射器）時，將全畫面所有發射器混為一談。當左側選手 X 的陀螺就位但發射器尚未入鏡時，若右側選手 B 的發射器已在畫面右上方被偵測到，系統誤判定「左側陀螺在全畫面發射器下方（`below_bodies=True`）」而提早中止左側選手錨定。
- **解法**：在預備發射階段，發射器上下文與相對高度檢查僅限於同側（以球場幾何中心 `mid_x` 劃分），徹底杜絕跨側發射器干擾。

### 突破 3：對戰前槽位缺口非對稱保護（Pre-battle Slot Asymmetric Protection）
- **現象**：在 [src/bxjudge/two_bey_integration.py](../src/bxjudge/two_bey_integration.py) 中，若非錨定槽位在發射準備期間有短暫 gap，會被標記為永久 unknown，導致即使進入戰鬥區成對也無法推斷對手身份。
- **解法**：以 `battle_start_frame` 為界：
  - **錨定槽位**：若在準備期有 gap，強制保留 unknown，嚴防 ID 交換（Anti-swap Safety）；
  - **非錨定槽位**：允許對戰前 setup gap 容錯，進入戰鬥區成對時正確推斷對手。

---

## 4. 安全護欄與配置規範

1. **零破壞性與回滾安全：**
   - [config/identity.yaml](../config/identity.yaml) 預設值保持 `prelaunch_setup_anchor_enabled: false`。
   - 新增 CLI 參數 `--identity YAML`，由 [src/bxjudge/cli.py](../src/bxjudge/cli.py) 與 [src/bxjudge/pipeline.py](../src/bxjudge/pipeline.py) 動態注入，完全相容舊有流程。
2. **單元測試全數通過：**
   - 包含新增的 13 項發射情境測試在內，全套 120 項測試（`test_identity`、`test_two_bey_integration`、`test_two_bey_tracking`、`test_launch_phase`、`test_dynamic_calibration`、`test_launcher_context_recovery`）**120/120 全部通過**。

---

## 5. 下一步推進（Next Steps）

依據 [COMPETITIVE_GAP_ANALYSIS.md](../docs/COMPETITIVE_GAP_ANALYSIS.md) §3.3 實驗順序：
- 身份瓶頸已徹底解除（`side-IDF1: 0.920`，覆蓋率達 `85.6%`）。
- 當前所有影片無法判出勝負的核心阻擋原因轉移至終結層：
  - `X_stop_reversed`、`B_stop_reversed`
  - `stop_evidence_missing`
  - `finish_unknown`
- **推進至 E4（停轉判定多特徵融合）**：
  - 在 [src/bxjudge/finish.py](../src/bxjudge/finish.py) 中結合 Laplacian 模糊度（`laplacian_variance`）與中心位置晃動遲滯，消除停轉前後的反轉誤判與微小震動誤判，正式解鎖最終勝負勝率（`winner_correct`）。
