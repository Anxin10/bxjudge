# bxjudge

Beyblade X 比賽影片的本機 AI 裁判（Local AI referee for Beyblade X match videos）。
所有程式都在 WSL2 Ubuntu 內執行；影片不上傳到任何雲端服務。

首次執行依本文順序：環境／同步 → 校準 → 抽格／人工標註／匯出 → 訓練兩種偵測器 →
比較並產生 selection → 安裝／啟動專案 Ollama → run → 選用 standalone vlm →
review → 抽樣稽核／eval／tune。以下命令都從 repo 根目錄執行。
來源影片（含雲端與 WSL 同步副本）只讀，輸出放在 repo 的獨立資料夾。
同一 results 目錄一次只能有一個寫入者：run、vlm、review、audit-sample 不可並行。
一般輸入錯誤回傳非零與 `error:`；加 `--debug` 可保留 traceback，Ctrl+C 回傳 130。

## 環境設定

```bash
cd ~/bxjudge
bash scripts/setup_wsl.sh      # ffmpeg、rclone、uv、Python 依賴
uv run pytest                  # 確認環境正常
```

## 同步影片（rclone，Google Drive 唯讀）

1. 先自行執行一次 `rclone config`，建立名為 **`gdrive`** 的 remote：
   - storage 選 `drive`（Google Drive）
   - **scope 選 `drive.readonly`**（rclone 的 "Read-only access" 選項；程式不會寫入、移動或刪除雲端檔案）
   - 在瀏覽器完成授權。WSL 無瀏覽器時，回答 auto config 為 `n`，WSL 端 `rclone config`
     會印出一行**完整的** `rclone authorize "drive" "<一長串編碼 JSON>"` 指令：把**整行**
     原封不動複製到有瀏覽器的 Windows 端 rclone 執行，再把產生的 token 貼回 WSL。
     **不要**只打 `rclone authorize "drive"`：第二個參數帶有剛才選的設定（含
     `drive.readonly` scope），省略它會以預設的完整讀寫權限授權。
     官方說明：<https://rclone.org/commands/rclone_authorize/>
   - `scripts/setup_wsl.sh` 與 `scripts/sync_videos.sh` 一律把 `~/.local/bin` 放進 PATH，
     所以裝在那裡的 rclone / uv 在一般登入 shell 也找得到。
2. 同步一個批次到 WSL 檔案系統（不要從 `/mnt/c`、`/mnt/g` 直接讀影片）：

   ```bash
   scripts/sync_videos.sh <drive_folder_id> <batch_name>
   # 10/1 批次：
   scripts/sync_videos.sh 1ExampleDriveFolderId0123456789ab 20261001
   ```

   影片會放在 `~/bx/videos/<batch_name>/`（只複製 `*.mp4`，可重複執行，已下載的不會重抓）。
   `<batch_name>` 必須是單一資料夾名稱（英數字與 `.` `_` `-`，不可含 `/`、`..` 或以 `.`/`-`
   開頭），否則腳本直接拒絕；腳本只用 `rclone copy`，不會同步刪除或改動雲端檔案。
3. 列出新增或變更過的影片（解析度 / fps / 編碼 / 是否有音軌）；此指令不會標記為已處理：

   ```bash
   uv run bxjudge scan ~/bx/videos/20261001
   ```

   已處理紀錄預設存在 `data/processed_videos.json`（以絕對路徑為 key，記錄 size + mtime），可用 `--db` 指定其他位置。

## Setup (English summary)

Run `bash scripts/setup_wsl.sh`, then once `rclone config` to create a remote named `gdrive`
with scope `drive.readonly` and authorize it in the browser. Without a browser in WSL, copy
the COMPLETE `rclone authorize "drive" "<encoded JSON>"` command that WSL `rclone config`
prints (the second argument carries the drive.readonly scope) to a machine with a browser;
a bare `rclone authorize "drive"` drops that scope. Sync a batch with
`scripts/sync_videos.sh <drive_folder_id> <batch_name>` (lands in `~/bx/videos/<batch_name>`),
and list new/changed videos with `uv run bxjudge scan ~/bx/videos/<batch_name>`.

## 校準（必須先有真實場地多邊形）

```bash
uv run bxjudge calibrate ~/bx/videos/20261001/example.mp4 --at 1 --name 20261001 --out calib
```

將 `example.mp4` 換成已同步的一支影片；WSL 需可顯示 OpenCV GUI。依視窗提示點選
stadium、lid、battle_area、over_left、over_right、xtreme、launch_left、launch_right，
Enter 儲存、u 復原、Esc 取消。參考 PNG 與 JSON 放在 `calib/`，換機位／場地須重新校準；
低相似度影片會標記 `recalibrate`，不硬套舊區域。校準原始畫面與來源路徑不進 Git。

## 本機標註：抽格 → SAM 3.1 → 人工修正 → COCO

先在 WSL 執行 `bash scripts/setup_labeling.sh`。主程式保留 NumPy 2 / OpenCV 5；
SAM 官方目前要求 NumPy < 2，因此使用 `scripts/sam-runtime/pyproject.toml` 與獨立
`uv.lock` 中固定的受支援環境（NumPy 1.26 / OpenCV 4.11）。不會在主環境安裝 SAM。
SAM SDK 固定為官方 commit `7f473cfd983d5eb43040ff608f66bff0d7bdcb53`，實際使用
**SAM 3.1 multiplex video**，RTX 3090 上 `use_fa3=False`、`compile=False`，並使用
checkpoint 原生 complex RoPE (`use_rope_real=False`)。每次 prelabel 指令只啟動一個
工作程序、載入一次模型；工作程序結束是模型記憶體與 SDK autocast 狀態的清理邊界。

權重為 gated 模型：先在 [facebook/sam3.1](https://huggingface.co/facebook/sam3.1)
申請／接受存取條款，必要時自行執行 `uv run --project scripts/sam-runtime hf auth login`
（舊文件稱 `huggingface-cli login`）。預設權重位置為
`weights/sam3.1/sam3.1_multiplex.pt`；以 `config/labeling.yaml` 修改路徑。
本機已驗證的檔案 SHA256：`0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`，
來源 HF revision `daa63191845a41281374e725f4c9e51c7a824460`。檔案不進 Git。

所有抽格與標註政策預設值位於 `config/labeling.yaml`；可另存一份並用 `--config` 指定。
預設 10/1 批次，每支 8 張，30% 均勻取樣、其餘優先能量峰值，最小間隔 0.1 秒：

```bash
uv run bxjudge label sample --per-video 8
# 或指定本機影片資料夾與輸出資料夾
uv run bxjudge label sample ~/bx/videos/20261001 --out data/labeling
uv run bxjudge label prelabel --out data/labeling
```

抽格先寫出 `samples.json`（影片絕對路徑、內容 SHA256、fps、原始格號），prelabel
重驗來源內容後產生可匯入的 `tasks.json`、各影片 JSON／`.report.json`，以及穩定的
`images/<video_sha256>/<原始格號>.jpg`。不要在標註或匯出前移動／刪除這個資料夾。
預標採每段最多 120 張**連續原始格**的實際暫存資源；只處理有選定格的區段，區段內
不降幀。每區段各類別重新提示並傳播，區段間物件 ID 重設。只保存選定格的框和分數，
不保存所有全解析遮罩。每區段在 finally 關閉迭代器、reset 並清空 state，再刪除暫存格；
CPU offload 或 async loading 本身不代表有界記憶體。這是靜態圖片標註，不提供跨區段軌跡。

`config/labeling.yaml` 的 `batched_grounding_batch_size: 4`（R47 實測）：固定版本的 SDK
builder 寫死 16，在真實 1080p 的 120 格窗口於第 17/120 格已達約 23.4 GiB 而中止；只把
grounding batch 改為 4（`window_frames` 120 與 SDK `postprocess_batch_size` 16 不變），同一
窗口 126.93 s 完成，全卡峰值 12381 MiB（含桌面約 2.2 GB）。這是資源量測，不代表框的品質：
該次 5 個 `bey_part` 框都錯把整顆陀螺或發射器框入，必須人工修正。

本版繞過官方 `start_session` 的不支援參數錯誤，使用固定版本的 `predictor.model`
直接 API。傳播不傳 optional `max_frame_num_to_track`，避免該版本兩層不同的結尾定義；
傳播範圍就是已限制張數的實際資源。官方組裝模型時會印出中途 tracker-only 的
missing/unexpected keys；最終完整模型會額外核對 checkpoint key 集合，任何缺漏直接失敗。

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

輸出目錄必須為空。匯出只接受每張圖恰好一筆已完成、未取消的人工作業；未標、取消或
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

## 訓練兩種偵測器 → 比較 → selection

先完成人工修正與 COCO 匯出，資料須含互不重疊的整支影片 train / val；一支影片全放 train
的資料不足以比較。SAM 預標不是真值。下載官方 Ultralytics 的 YOLO26 Nano 預訓練權重，
放到 `config/detection.yaml` 實際使用的位置（不會啟動訓練或推論）：

```bash
mkdir -p runs/pretrained
curl --fail --location --output runs/pretrained/yolo26n.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
uv run bxjudge train --kind rfdetr --dataset data/dataset --out runs/rfdetr-real --config config/detection.yaml --thresholds config/thresholds.yaml
uv run bxjudge train --kind yolo26 --dataset data/dataset --out runs/yolo26-real --config config/detection.yaml --thresholds config/thresholds.yaml
uv run bxjudge compare --dataset data/dataset --rfdetr runs/rfdetr-real/fit/checkpoint_best_total.pth --yolo26 runs/yolo26-real/fit/weights/best.pt --output runs/compare-real.json --selection-out weights/selected.json --config config/detection.yaml --thresholds config/thresholds.yaml
```

[官方 YOLO26 說明](https://docs.ultralytics.com/models/yolo26/)；下載位置對應已安裝套件下載器的官方
`ultralytics/assets` v8.4.0 release。RF-DETR Nano 由框架取得其官方預訓練權重。
訓練採 `config/detection.yaml` 的 production policy（100 epochs，解析度取自
`config/thresholds.yaml`）；每次訓練用新的空 out 目錄。compare 在相同 COCO val 上比較兩者
並產生 `weights/selected.json`，run 必須使用此 manifest，不能用任意單一 checkpoint 代替。
`config/detection_smoke.yaml` 僅供明確指定的合成 smoke，不支持真實準確率驗收；
不要以 smoke selection 覆寫上述正式 selection。

## 專案本機 Ollama（固定 0.40.7）

使用[官方 Linux release](https://github.com/ollama/ollama/releases/tag/v0.40.7)，不執行
全域 install.sh、不使用 systemd，也不變動共享的 11434。安裝器只接受新 runtime 目錄，
先驗證官方 amd64 archive 的 1505607169 bytes 與 SHA256
`7f082e625b4c334df84ed1558fe282d6d8091fe3545790aef30b3a9c50d44d41`，再安全解壓。
解壓工具 zstandard 0.25.0 放在 uv 的一次性隔離環境，不改專案或全域 Python 依賴。

```bash
uv run --isolated --no-project --with zstandard==0.25.0 python scripts/setup_ollama.py
# 已下載同一官方 archive 時可重用，仍會檢查大小與 SHA256：
# uv run --isolated --no-project --with zstandard==0.25.0 python scripts/setup_ollama.py --archive /absolute/path/ollama-linux-amd64.tar.zst
bash scripts/serve_ollama.sh
```

此命令以前景執行 `weights/ollama-runtime/bin/ollama serve`，固定
`OLLAMA_HOST=127.0.0.1:11435`、`OLLAMA_NO_CLOUD=1`、`OLLAMA_MODELS=$PWD/weights/ollama`。
終端保留服務，Ctrl+C 停止此服務；若 11435 已被占用，先確認原服務，不要終止他人的程序。
不需更新已存在且版本正確的 runtime；重新安裝須明確選擇新的 `--runtime` 目錄，
並將該路徑作為 `serve_ollama.sh` 第一參數。第二參數可指定另一個本機 model store。

另開終端，先確認服務，再明確下載所選的兩個本地模型（這兩步不會上傳影片）：

```bash
curl --fail --noproxy '*' http://127.0.0.1:11435/api/version
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NO_CLOUD=1 weights/ollama-runtime/bin/ollama pull qwen3.8:27b
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NO_CLOUD=1 weights/ollama-runtime/bin/ollama pull qwen3-vl:8b
```

`config/vlm.yaml` 指向上述端點，主模型 27b、備援 8b。每次 VLM phase 在抽取事件畫面前
只做一次 `/api/version` preflight（最多 5 秒，不載入模型）；端點不可用時只警告一次，
保留核心判定並寫 `vlm_unavailable` 備註。preflight 不保證模型已下載或 GPU 記憶體足夠；
推論失敗時保留既有本地備援政策。只接受 literal loopback，拒絕 cloud tag、HTTP redirect
與環境代理；VLM 永遠只寫 `vlm_note`，不覆寫判定或人工覆核。

## 批次判定 → 選用 VLM 重試 → 人工覆核

```bash
uv run bxjudge run ~/bx/videos/20261001 --calib calib --out results/20261001 --selection weights/selected.json --config config/detection.yaml --thresholds config/thresholds.yaml --rules config/rules.yaml --vlm-config config/vlm.yaml
# after_run: true 預設會在釋放偵測器後執行 VLM；--no-vlm 可先只產出核心結果。
# 服務就緒後可另跑，只處理仍未人工覆核、needs_review 且尚無成功備註的回合：
uv run bxjudge vlm results/20261001 --calib calib --config config/vlm.yaml --thresholds config/thresholds.yaml --rules config/rules.yaml
uv run bxjudge review results/20261001 --calib calib --thresholds config/thresholds.yaml --rules config/rules.yaml
```

瀏覽器開啟 http://127.0.0.1:8501，逐一覆核需覆核回合；完成後停止 UI，再做稽核抽樣。
run 會跳過同一 outdir 中未變的影片；已人工覆核的影片不可重分析。變更偵測器／門檻或加入
影片以重做驗收時，一律使用新的 results outdir。建立 audit_sample.json 後整批原始母體凍結，
不可在原目錄補判或重跑；請保留該目錄及稽核 metadata，不可手改或刪除 metadata 以重抽。
`calib/`、`results/`、`reports/`、`graphify-out/` 預設忽略，另選自訂路徑也須自行避免加入 Git。

## 評估與門檻校正（`bxjudge eval` / `bxjudge tune`）

人工確認需覆核回合後，也必須抽查「原始未標覆核」回合，才能評估自動裁判正確率。
以下 `--n 100 --seed 42` 是抽樣例子，樣本不足則指令拒絕，不能自行挑容易的回合取代；
可改為實際母體大小做全查。抽樣後需逐筆完成人工覆核（UI 的「抽樣稽核」篩選器）：

```bash
uv run bxjudge audit-sample results/20261001 --n 100 --seed 42 --rules config/rules.yaml
uv run bxjudge review results/20261001 --calib calib
uv run bxjudge eval results/20261001            # 寫 reports/eval-YYYYMMDD.md
uv run bxjudge tune results/20261001            # 寫 reports/tune-YYYYMMDD.md，印出建議 diff
# 優化迴圈：新設定輸出到新資料夾，再拿已覆核的舊結果當真值（兩邊都只讀）
uv run bxjudge run ~/bx/videos/20261001 --calib calib --selection weights/selected.json --out results/20261001-try2
uv run bxjudge eval results/20261001-try2 --truth results/20261001
```

- 真值 = 已覆核回合的 effective 值（原判 + `reviewer_fix`）；預測 = 原始欄位（含原始
  `needs_review`）。`recalibrate` / `no_round` / `decode_error` / `analysis_error` 不是回合。
- 報告最上方是摘要表：spec §11 五項（未標覆核回合判定正確率、觸碰召回、空碰召回、
  需覆核比例、逐影片速度要求）與是否達標，以及核心判定延遲、VLM 單次請求延遲
  （中位數 / p95 / 最大，冷 / 熱）、VLM 觸發率、端到端延遲、速度達標率、裁判正確率。
  分母為 0 的指標顯示 unverified，一律不算達標。
- 未標覆核準確率需完整全查，或凍結的隨機 audit 樣本完整覆核後以 Wilson 信賴下界判定；
  樣本未覆核完、來源不完整或真值不足都顯示 unverified。只覆核被標記回合不能驗證自動判定。
  reviewed output agreement 是覆核輸出對覆核真值的一致性，不能當作原始裁判準確率。
- 逐影片速度要求保留「處理時間 ≤ 影片長度」；目前核心 + VLM stage sum 的最差逐片比值
  超標會 FAIL，符合估計值仍為 UNVERIFIED，正式 PASS 需要實測逐影片 wall 證據，目前管線
  尚未記錄。batch wall throughput 與中位數僅是獨立診斷，不能取代逐影片要求。
  核心 stage 不含一次性模型載入與 scan / probe；批次 wall 含初始化與 VLM。
  影片長度用 videos.json 的播放秒數，慢動作不換算。
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