# 標註、評估與規則核對

> 原 README 的使用說明章節。

## 本機標註：抽格 → LocateAnything → 人工修正 → COCO

在 WSL 執行 `bash scripts/setup_labeling.sh`。安裝會建立獨立的 `scripts/locateanything-runtime/.venv`，
依 lockfile 固定 CUDA、PyTorch、Transformers 等套件版本，並取得
[NVIDIA LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B/tree/c32291ca5e996f5a7a485845b4f57a233936bba0)
固定版本。權重預設放在 `weights/locateanything/LocateAnything-3B`。若已有同版本副本，可以用
`bash scripts/setup_labeling.sh --from-snapshot /path/to/LocateAnything-3B` 匯入。
模型載入前核對本機檔案雜湊，每次預標只啟動一個離線工作程序，模型只載入一次。

所有抽格與標註設定在 `config/labeling.yaml`，可以另存一份並以 `--config` 指定。
預設 10/1 批次，每支 8 張，30% 均勻取樣、其餘優先能量峰值，最小間隔 0.1 秒：

```bash
uv run bxjudge label sample --per-video 8
uv run bxjudge label sample ~/bx/videos/20261001 --out data/labeling
uv run bxjudge label prelabel --out data/labeling
```

抽格先寫 `samples.json`（影片絕對路徑、內容 SHA256、fps、原始格號）。預標重驗來源，
只對選定格做單圖推論，輸出 `tasks.json`、各影片 JSON／`.report.json` 與
`images/<video_sha256>/<原始格號>.jpg`。報告記錄模型版本、提示詞、框數與推論時間。
模型輸出沒有校準過的框置信分數。匯入 Label Studio 後，請核對陀螺、碎片與發射器，
補漏框、刪誤框並修正邊界；訓練匯出只使用人工完成的標註。

Label Studio 需人工建立本機帳號與專案。以下在 WSL 啟動，明確綁定 loopback，Windows
瀏覽器開啟 [http://localhost:8080](http://localhost:8080)：

```bash
export PYTHONPATH="$PWD/src"
export DJANGO_SETTINGS_MODULE=bxjudge.labeling.label_studio_settings
export COLLECT_ANALYTICS=false LATEST_VERSION_CHECK=false
export SENTRY_DSN= FRONTEND_SENTRY_DSN=
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$(realpath data/labeling)"
label-studio start --internal-host 127.0.0.1 --port 8080 --debug --no-browser \
  --label-config config/label_studio.xml
```

本機設定透過 Django 支援的 settings module 停用 Community 版不使用的 Redis queues；
固定版本仍會在啟動時探測 queue，預設 localhost:6379 在此 WSL 環境會阻塞。
此固定版本的 CLI `--debug` 僅略過另一次會阻塞的 localhost 埠號探測，直接使用指定埠；
Django DEBUG 仍為 false。若指定埠已被占用，請先停止占用程式或明確換埠。
以上設定也停用分析遙測、錯誤回報與版本查詢。

建立專案後，將 `config/label_studio.xml` 貼入 Labeling Interface（若啟動參數未替新專案
套用）。在專案 Settings → Cloud Storage → Add Source Storage 選 **Local Files**，
Absolute local path 填 `$(realpath data/labeling/images)` 的實際輸出路徑，儲存即可，
**不要 Sync**（已有 tasks JSON，Sync 會再建立圖片任務）。這個名稱中的 Cloud Storage
也是本機儲存設定入口，選 Local Files 不會上傳。接著匯入 `data/labeling/tasks.json`。
僅設環境變數而未註冊專案的本機來源儲存，圖片端點會回傳 404。圖片使用官方 local-file URL
`/data/local-files/?d=images/...`；DOCUMENT_ROOT 必須是**此次標註輸出資料夾**。
瀏覽器取用的是本機 Label Studio，影片與畫面不傳到雲端。

標註類別固定 `bey`、`bey_part`、`launcher`。**launcher 只框發射器本體；棉線與拉條
一律不框**。SAM 若把線或拉條框入，請縮回本體。預標會移除 launcher 長邊／短邊比
超過 4 的細長框，終端和每影片 `.report.json` 都列出 `launcher_filtered` 數量；此過濾
不能保證把所有線排除，人工仍需檢查。三類提示詞也在 YAML 中調整。

人工逐張修正並 Submit（無物件圖片可提交空框，作為背景）。從 Label Studio 匯出
**JSON** 後執行：

```bash
uv run bxjudge label export human-export.json --out data/dataset
```

後續新增難例、需要固定原驗證集比較模型時，使用另一個新輸出目錄：

```bash
uv run bxjudge label extend-training data/dataset new-human-export.json --out data/dataset-v2
```

此命令只接受**既有 train 影片的新影格**與已完成的人工作業；拒絕 val 影片、未分組影片、
重複來源影格及不一致的影片識別。原 val JSON／圖片與既有 train 標註不變，產生新資料版本，
不重新切分、不覆寫原資料集。`--out` 必填，且必須是基底資料集以外的新路徑。這個流程只附加
新影格，不用來改寫既有影格的標註；未人工 Submit 的預標不能拿來擴訓。
影像先以同一份 bytes 解碼、驗證並保存快照，再從快照匯出，避免原路徑更新後把舊框套到新圖；
輸出來源紀錄保留對應的 `image_sha256`。

首次 `label export` 的輸出目錄必須為空。匯出只接受每張圖恰好一筆已完成、未取消的人工作業；未標、取消或
多筆未裁決作業直接報錯，永遠不把 SAM predictions 自動當成訓練真值。旋轉框不支援。
預設依整支影片的內容識別碼排序切 20% 驗證集，同支影片不跨集合；一支影片時全放 train。

Task 12 的唯一 canonical COCO 格式如下（`file_name` 相對於 dataset 根目錄）：

```text
data/dataset/
  images/train/000000001.jpg
  images/val/000000002.jpg
  annotations/instances_train.json
  annotations/instances_val.json
  provenance.json
```

類別 ID 為 `1=bey, 2=bey_part, 3=launcher`；bbox 為實際 JPG 像素的 `[x,y,width,height]`。
每張 COCO image 帶 `video_id` 和原始 `frame`；`provenance.json` 保留來源 export、原始
影片路徑、影片 SHA256、fps、格號與集合。訓練前須保留這份 provenance，不能再隨圖片重切。

不在預設 pytest 重載模型。需要重新驗證 GPU 管線時，明確執行：

```bash
uv run python scripts/smoke_labeling.py --out .superpowers/sdd/plan/scratch/labeling-smoke
```

此人工觸發的自動 smoke 產生四張合成影格，以兩張為一個實際窗口，透過真正 CLI/SAM
預標。零物件只證明管線可運作，不證明真實陀螺辨識準確率。真實批次同步/OAuth、
約 400 張人工標註與 Windows 瀏覽器人工驗收仍須使用者完成。

## 評估與門檻校正（`bxjudge eval` / `bxjudge tune`）

覆核完（`bxjudge review`）後，以人工確認的回合（`reviewed`）為真值評估：

```bash
uv run bxjudge eval results/20261001            # 寫 reports/eval-YYYYMMDD.md
uv run bxjudge tune results/20261001            # 寫 reports/tune-YYYYMMDD.md，印出建議 diff
# 優化迴圈：新設定輸出到新資料夾，再拿已覆核的舊結果當真值（兩邊都只讀）
uv run bxjudge run ~/bx/videos/20261001 --out results/20261001-try2
uv run bxjudge eval results/20261001-try2 --truth results/20261001
```

- 真值 = 已覆核回合的 effective 值（原判 + `reviewer_fix`）；預測 = 原始欄位（含原始
  `needs_review`）。`recalibrate` / `no_round` / `decode_error` / `analysis_error` 不是回合。
- 報告最上方是摘要表：spec §11 五項（未標覆核回合判定正確率、觸碰召回、空碰召回、
  需覆核比例、處理速度比〔最慢影片〕）與是否達標，以及核心判定延遲、VLM 單次請求延遲
  （中位數 / p95 / 最大，冷 / 熱）、VLM 觸發率、端到端延遲、速度達標率、裁判正確率。
  分母為 0 的指標顯示 unverified，一律不算達標。
- 速度以「核心 + VLM」計（R59）；核心時間不含一次性的模型載入與掃描 / probe，核心-only
  比值另列為診斷。影片長度用 videos.json 記錄的播放秒數，慢動作不換算。
- 實際牆鐘時間（R61）：每次 `bxjudge run` 從開始（含設定與模型載入、probe）到 VLM 階段
  結束的 `wall_s`，以及其中的 VLM 階段 `vlm_wall_s`，追加到 `runs.json`；單獨的
  `bxjudge vlm` 也記一筆（不增加影片長度）。端到端同時列「每支影片 核心+VLM 加總」與
  「批次牆鐘 / 影片總長」（分母只算與加總相同條件的影片：有真實回合、已量測、
  ≤ 300 秒，每支只算一次；不合格影片的量測時間從牆鐘扣除），兩者都只是診斷。
  VLM 時間只算一次；加總大於牆鐘時會明列，不隱藏。
- 速度達標判定（R66a，取代 R61 的判定規則）：每支影片以 `judgment_sources` 記錄目前每個
  原始回合所屬的產生批次 `run_id`、回合編號、實測 `wall_s` 與 `processing_s`。重判保留已覆核
  回合時，也保留各代原始計時及批次證據；只有真正被取代的預測才移除其來源。
  每個 wall 都是同一計時器從第一次解碼（含校準選擇）到結果寫完；`judged_wall_s` 與
  `processing_s` 是目前來源中最慢的一次分析，`attempt_wall_s` / `attempt_run_id` 則另外記錄
  最近一次嘗試。每支有真實回合且 ≤ 300 秒的影片，
  max(max(`judgment_sources[].wall_s`) + `vlm_s`, `attempt_wall_s`) ÷ 目前來源檔最近一次 probe 的播放長度，
  最大值必須 ≤ 1.0；**同時**批次牆鐘比（`runs.json` 批次牆鐘 ÷ 已量測影片總長，含模型載入
  與 probe 等每批開銷）也必須 ≤ 1.0。兩者都過才算達標；報告與 CLI 列出最慢的影片。
  最近一次嘗試失敗（`analysis_status` 非 `ok`、保留舊回合）時，快速失敗不會掩蓋舊回合的
  慢分析，舊檔案的長度也不會沿用到被替換的新檔。任何一支真實影片缺牆鐘時間（例如舊版
  結果）、沒有批次紀錄，或批次紀錄沒有涵蓋每一支已量測的真實影片（例如某次 run 判完部分
  影片後被中斷、沒寫出批次紀錄）→ 速度 unverified。每次 run 有一個 `run_id`，記在它寫的每支
  影片計時與它的批次紀錄上；影片目前的計時必須對得上一筆完整的批次紀錄，舊批次紀錄不能替
  被中斷的重判背書；所有仍保留的產生批次及最新嘗試都必須有完整批次證據並通過速度門檻。
  缺少新來源欄位的舊結果一律 unverified，不從舊欄位推測或遷移。
  未變更的影片一般重跑不會重判，請用
  `uv run bxjudge run <資料夾> --rejudge`（忽略已處理紀錄、重判每支影片，已覆核的回合保留）。
  若保留的已覆核回合缺少來源證據，`--rejudge` 也不能重建舊計時；需用新的 `--out` 目錄產生
  新預測及計時，並以舊結果作 `bxjudge eval <新結果> --truth <舊結果>` 的人工真值對照。
  每支比值中位數只作診斷。最近一次狀態不是 `ok` 的影片（暫時失敗或舊版沒有狀態）下次
  `bxjudge run` 一定重判；VLM 階段（含單獨的 `bxjudge vlm`）跳過最近一次失敗的影片，
  不動其保留的回合。執行中被中斷（Ctrl+C、OOM）的嘗試不留紀錄。
- 未標覆核回合正確率需要隨機抽樣（R62）：只覆核被標記的回合會有選擇偏差。

  ```bash
  uv run bxjudge audit-sample results/20261001 --n 30 --seed 1   # 標記隨機抽樣（只加 audit_sample）
  ```

  抽樣只從真實、原判未標覆核的回合抽，不改任何判定；同 seed 重跑不變，n 超過可抽數會拒絕。
  在 `bxjudge review` 用「抽樣稽核」篩選逐一覆核。評估時：全部未標回合都覆核 → 精確值；
  否則用已覆核的抽樣回合估計並附 Wilson 區間（信心水準 `audit_confidence_level`）；
  沒有抽樣 → unverified（selection bias / no random sample）。抽樣外自行覆核的未標回合
  另列為 biased，不納入。全部未標回合都覆核時，此值等於原定義（已覆核未標回合的比例）。
  判定（R64）：精確值直接比目標；抽樣估計只有在抽樣**全部**覆核、覆核數 ≥ `audit_min_n`
  且 Wilson 下界 ≥ 目標時才 PASS，點估計達標但證據不足 → `INCONCLUSIVE (樣本不足)`，
  點估計未達標 → FAIL。CLI 摘要與報告表格的「判定」欄一致。
- 取捨區塊（R63 / R67）並列自動化率（1 − 需覆核比例）、正確率下界與速度達標，不合成
  單一加權分數，也不外推。正確率下界 = 已覆核且原判勝負與終局都正確的真實回合 ÷ 全部
  真實回合（未覆核回合，含被否決校準的，一律不算對，所以不會高估）。整體正確率只在
  每個真實回合都已覆核時才給精確值，否則顯示 `unverified (truth coverage x/y)`。
  CLI 摘要與報告表格都有這兩個欄位。
- 目標值、容許誤差與百分位都在 `config/evaluation.yaml`。
- `--truth` 只用 (video, round) 唯一、該影片回合數相同、`t_launch` 差在容許內的回合；
  重複、缺少、分段不同、時間不符都列在報告並排除，不猜最近的回合，也不改任何檔案。
- `tune` 從已覆核回合的 `motion_evidence` 取晃動樣本（排除撞擊時間窗與未量測格），
  算出 `shake_px` 候選值並印出 `config/thresholds.yaml` 的建議 diff，**不會自動覆寫**；
  也列出旋轉方向 flag 正確率（必要時建議對調 `orbit_to_spin`）與 reversed 轉停被推翻比例。
- 用合成資料訓練的偵測器（synthetic_detector）跑出的結果，報告會標示 SMOKE ONLY：
  只證明流程可運作，不是驗收。

## 規則核對（官方第 12 版）

`config/rules.yaml` 已逐項核對 Takara Tomy 官方「ベイブレードエックス レギュレーション」
第 12 版（2026 年 3 月，2026-03-13 施行，4 頁；
<https://beyblade.takaratomy.co.jp/beyblade-x/_image/regulation.pdf>，
SHA256 `c0950dba96979d99c88a53bd7114c8bf1b2b4ff470875f475cbe1dcdeed26541`）：

- 先取得 4 分勝；spin 1、burst 2、over 2、xtreme 3：與官方一致。
- 同時發生終結 → 平手重賽；兩顆陀螺在接觸盤面前空中相撞 → 重賽：與官方一致。
- 發射落到盤外或蓋子上 = シュートミス → 重賽並計一次失誤：與官方一致。
- `repeat_miss_penalty: opponent_1pt_on_2nd_miss`（選用）照官方計數：同一バトル（重賽到
  有人得分為止）內某方第 2 次失誤 → 對手得 1 分並清除雙方失誤數；雙方在同一次發射同時
  第 2 次失誤 → 該次無效、雙方都回到 1 次；正常得分也清除雙方失誤數；空碰、平手重賽
  不清除。

以下是**使用者自訂（不是官方規定）**，不宣稱與官方完全一致：

- 觸碰只算發射器**本體**；棉線與拉條碰到盤面一律不算（官方禁止身體、發射器、裝飾品碰
  盤面，只在不影響對戰時有條件地豁免繩式發射器的握把與繩子）。偵測到觸碰 → 重賽並計
  失誤（2025 世界大會作法）；官方「故意觸碰判失格」不在自動判定範圍。
- 預設 `repeat_miss_penalty: none`：重複失誤不額外扣分（spec §12.1，可在設定檔切換）。

已知限制：

- 第 12 版追加「Barrel Griffon 分離的 barrel 不算 burst」。偵測類別 `bey_part` 無法分辨
  這個零件，若只因該 barrel 脫落而判 burst 會判錯，相關對戰需人工覆核。
- 不判斷搶拍 / 慢拍（フライング、後出し）：本系統不使用音軌（spec §3）。
