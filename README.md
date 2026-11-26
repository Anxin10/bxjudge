# bxjudge

Beyblade X 比賽影片的本機 AI 裁判（Local AI referee for Beyblade X match videos）。
所有程式都在 WSL2 Ubuntu 內執行；影片不上傳到任何雲端服務。

## 目前狀態（2026-11）

- 本機流程：同步影片 → 場地定位與校準 → 偵測與雙陀螺追蹤 → 回合切分 → 規則判定 → 人工覆核。
- 判定原則：證據不足就交給人工覆核。軟性預測不當成裁判證據，身分不確定時不硬判 X／B。
- 新功能一律以設定開關加入，**預設關閉**；驗證過才考慮開啟。
- 驗證集（holdout-21）目前產出 0 個自動判決，主要瓶頸是發射時間與場地幾何。
  各項實驗與結果見 [`reports/`](reports/)。

## 資料與隱私

- 影片、影格、標註資料、權重與執行輸出（`data/`、`runs/`、`weights/`、`calib/`、`results/`）
  都不進 Git，也不上傳任何雲端服務。
- `reports/` 只收錄整理過、不含畫面的實驗報告；其餘產生的報告留在本機。
- 驗證集與 identity-test 的真值只用於驗收，不用來調整參數。

## 目錄

| 路徑 | 內容 |
|---|---|
| `src/bxjudge/` | 主程式（CLI 入口 `bxjudge.cli:main`） |
| `config/` | 所有門檻與功能開關（YAML） |
| `tests/` | pytest 單元測試與小型軌跡 fixture |
| `scripts/` | 環境安裝、同步、匯出工具；`scripts/e*/`、`scripts/holdout/` 是實驗腳本 |
| `docs/` | 規格、計畫、使用說明與開發日誌 |
| `reports/` | 實驗報告（E0–E9、holdout-21） |

## 文件

- [規格](docs/spec.md)、[實作計畫](docs/plan.md)
- [標註、評估與規則核對](docs/guide/labeling-eval-rules.md)
- [進階功能與設定](docs/guide/advanced-features.md)（TensorRT、軌跡動畫、動態校準等）
- [競品差距分析與實驗路線](docs/COMPETITIVE_GAP_ANALYSIS.md)
- [設計與實作計畫（superpowers）](docs/superpowers/)
- [2026-10 開發日誌](docs/history/dev-log-2026-10.md)

## 環境設定

```bash
cd ~/bxjudge
bash scripts/setup_wsl.sh      # ffmpeg、rclone、uv、Rust 與 Python 依賴
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
uv run pytest                  # 確認環境正常
```

環境需有 Rust 1.83 以上與 C 編譯工具。開新終端機時，先執行上方的 PATH 設定。
Python 套件安裝時會編譯裁判幾何擴充，包含多邊形判定、矩形接觸、
整框入袋與軌跡速度。修改 Rust 原始碼後，執行 uv sync 會重新編譯。

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

## 完整流程（從同步到評估）

以下依序執行即可從全新 checkout 跑完一個批次（以 10/1 批次 `20261001` 為例）。全部在
WSL 的 `~/bxjudge` 下執行；各步驟的細節在後面的章節。產生的 `calib/`、`results/`、
`reports/`、`runs/`、`data/`、`weights/` 都在 `.gitignore` 內，**不要**加進 Git 或上傳
（內含比賽畫面）。指令有錯（路徑不存在、YAML 格式錯、設定值不合法）時只印一行
`error: …` 並以結束碼 2 結束；要看完整 traceback 時在子指令前加 `--debug`，例如
`uv run bxjudge --debug run …`。

1. **環境與同步**（見上方「環境設定」「同步影片」）：

   ```bash
   cd ~/bxjudge
   bash scripts/setup_wsl.sh
   scripts/sync_videos.sh 1ExampleDriveFolderId0123456789ab 20261001
   uv run bxjudge scan ~/bx/videos/20261001
   ```

2. **校準場地**（spec §6.2；需要 WSLg 視窗，依畫面提示點選各區域）。每種機位／場地
   擺放各做一次，名稱自訂；輸出 `calib/<name>.json` 與參考畫面 `calib/<name>.png`：

   ```bash
   uv run bxjudge calibrate ~/bx/videos/20261001/<任一支影片>.mp4 --name 20261001 --at 1.0
   ```

3. **標註**（見下方「本機標註」）：抽格 → LocateAnything 預標 → Label Studio 人工修正 → 匯出 COCO：

   ```bash
   bash scripts/setup_labeling.sh
   uv run bxjudge label sample ~/bx/videos/20261001 --out data/labeling --per-video 8
   uv run bxjudge label prelabel --out data/labeling
   # 在 Label Studio 匯入 data/labeling/tasks.json，逐張修正並 Submit，匯出 JSON
   uv run bxjudge label export human-export.json --out data/dataset
   ```

4. **訓練兩種偵測器並比較**（RTX 3090；兩個輸出資料夾都必須是新的空資料夾）。`train`
   會印出最佳 checkpoint 的路徑；`compare` 用同一份 COCO 驗證集比較，並把勝出者寫進
   `weights/selected.json`（`bxjudge run` 預設讀這個檔）：

   ```bash
   uv run bxjudge train --kind rfdetr --dataset data/dataset --out runs/train-rfdetr-20261001
   uv run bxjudge train --kind yolo26 --dataset data/dataset --out runs/train-yolo26-20261001
   uv run bxjudge compare --dataset data/dataset \
     --rfdetr runs/train-rfdetr-20261001/fit/checkpoint_best_total.pth \
     --yolo26 runs/train-yolo26-20261001/fit/weights/best.pt
   # → weights/selected.json 與 runs/compare.json
   ```

   沒有 `weights/selected.json` 時 `bxjudge run` 會直接報錯並提示先做這一步。

5. **啟動本機 VLM**（spec §8、R26）：只用專案內獨立的 Ollama 0.40.7，綁 loopback
   `127.0.0.1:11435`、`OLLAMA_NO_CLOUD=1`，模型存在 `weights/ollama`；不影響系統上其他
   Ollama（例如 11434）。第一次先安裝並下載模型（壓縮檔 SHA256 必須一致）：

   ```bash
   # 安裝（只下載官方壓縮檔，驗證大小與 SHA256 後才解開到 weights/ollama-runtime；
   # 已存在的 runtime 不會被覆蓋，除非加 --force；不會下載模型）
   uv run --no-project --with zstandard python scripts/setup_ollama.py
   ```

   每次開機後在**另一個終端機**前景啟動（Ctrl+C 停止；綁 `127.0.0.1:11435`、
   `OLLAMA_NO_CLOUD=1`、模型目錄 `weights/ollama`；不會管理或停止其他 Ollama），
   再確認有回應：

   ```bash
   cd ~/bxjudge
   scripts/serve_ollama.sh
   # 另一個終端機：
   curl -s --max-time 5 http://127.0.0.1:11435/api/version
   ```

   第一次啟動後下載兩個模型（`config/vlm.yaml` 的主模型與備用模型，只下載、不上傳）：

   ```bash
   OLLAMA_HOST=127.0.0.1:11435 weights/ollama-runtime/bin/ollama pull qwen3.8:27b
   OLLAMA_HOST=127.0.0.1:11435 weights/ollama-runtime/bin/ollama pull qwen3-vl:8b
   ```

   VLM 階段開始前會先以 `GET /api/version` 檢查伺服器（回應必須含非空的 `version`，否則視為連不上）（逾時 `preflight_timeout_s`，
   在 `config/vlm.yaml`）。連不上時**不解碼任何畫面**，直接跳過 VLM 階段：需 VLM 的回合
   寫入 `vlm_unavailable: server unreachable …` 註記並印一行警告；之後啟動伺服器再執行
   第 8 步即可補問。VLM 只寫 `vlm_note`，永不改判定。

6. **自動校準驗收與預覽**（spec §6、§6.6；唯讀，不寫 `calib/`）。鏡頭只移動一點時，
   `bxjudge run` 會以最接近的手動校準自動對齊，不必每支影片都重新點選。正式判定前先用
   `autocalib` 看每支影片會得到什麼；快照與預覽圖寫到 `--out` 指定的暫存資料夾（格式同
   結果資料夾的 `calibrations/`）：

   ```bash
   uv run bxjudge autocalib ~/bx/videos/20261014/*.mp4 --calib calib --out runs/autocalib-check
   ```

   每支影片印一行 `影片: manual|auto|recalibrate name=<校準名> score=<分數> id=<快照 id>
   preview=<預覽圖路徑>`（`recalibrate` 沒有快照，只印 `name=- score=-`）。`--calib` 必須是
   存在的資料夾，否則印 `error:` 並以結束碼 2 結束；`--calib` 只讀，快照與預覽圖寫到 `--out`
   指定的暫存資料夾（格式同結果資料夾的 `calibrations/`）。

   - `manual`：影片第一秒與某個手動校準相似度 ≥ `calibration_min_similarity`（0.8），
     直接用該手動校準，不微調；`score` 是相似度，預覽圖是該手動校準自己的參考畫面。
   - `auto`：以最接近的手動校準對齊到影片第一秒的畫面，對位分數 ≥ `config/autocalib.yaml`
     的 `min_score`（0.6）。**同一機位的影片共用同一組快照與同一張預覽圖**（`id` 與 `preview`
     相同）：預覽圖以該組**第一支**被處理的影片的第一秒畫面為底，`score` 是該組建立時的
     對位分數；其他影片只知道與該組畫面的相似度 ≥ 0.8，並沒有各自對位。預覽圖疊上全部 8 個
     區域的輪廓（左上角標示 kind、template、score）。**請開 `preview=` 指的那張圖，目視確認
     每個輪廓都貼著場地的邊（外框、過界口、極限區）**，歪掉就不要採用（見下方否決）。分數
     接近 0.6 的（例如 0.6–0.65）尤其要仔細看，常見的是某一側的外框或蓋子輪廓偏掉。
     **覆核區間**：自動校準分數 ≥ `min_score`（0.6）才會被採用；分數低於
     `config/autocalib.yaml` 的 `auto_review_score`（0.75，須介於 `min_score` 與 1 之間）時，
     `bxjudge run` 仍以它判定，但該影片的每個回合都標成需覆核（`needs_review`，原因
     `calibration_low_score`），會出現在 `bxjudge review` 的需覆核清單並計入 review_ratio，
     不會不經覆核就計分。手動校準不受此區間影響。
   - `recalibrate`：沒有任何手動校準對得上、對位也不夠可靠，判定時此影片會被標成
     `recalibrate`，不猜區域；為該鏡頭新增一個手動校準（第 2 步）後重跑。

   **用手動真值驗收（`--truth`）**：在同一個機位另外手動點一次當真值，**存到 `calib/` 以外的
   資料夾**；放進 `calib/` 它就變成範本，該機位會直接以 `manual`、誤差 0 通過，完全沒測到
   自動對齊。`--truth` 只有一個檔，會套用到命令列上的**每一支**影片，所以只放該真值那個機位
   的影片：

   ```bash
   uv run bxjudge calibrate ~/bx/videos/20261014/<任一支影片>.mp4 --name truth-20261014 --out truth
   uv run bxjudge autocalib ~/bx/videos/20261014/*.mp4 --calib calib --out runs/autocalib-check \
     --truth truth/truth-20261014.json
   ```

   有 `--truth` 時，每行再加 `boundary_error_px=<8 區域平均邊界誤差，推論像素> <狀態>`，
   最後印 `PASS n / FAIL n / REJECTED n`：

   - `PASS`：誤差 ≤ `max_boundary_error_px`（8 px）。
   - `FAIL`：誤差超過門檻。
   - `REJECTED`：結果是 `recalibrate`（`boundary_error_px=-`），沒有產出校準，不能算通過。

   任何 FAIL 或 REJECTED 時結束碼為 1。白色、黑色下盤有真值時應該要對得上；只有驗收透明
   下盤時（允許對不準、但必須被標成 `recalibrate`，spec §6.6）才加 `--allow-recalibrate`，
   此時 REJECTED 仍會印出並計入摘要，但不影響結束碼（FAIL 仍然是 1）。

   使用上要注意：

   - **每一種拍攝位置／下盤顏色都要有一個手動校準**。自動對齊只能吸收範本附近的小幅
     移動（位移約 ≤ 15% 畫面寬、旋轉 ≤ 8°、縮放 0.85–1.15、輕微透視）；換了盤、換了
     下盤顏色或鏡頭角度差很多，就要另外手動點一次。只有一個範本時，其他機位多半會是
     `recalibrate`，這是刻意的保守行為，不是錯誤，也不要為此調低 `min_score`。
   - 影片第一秒若有手或記錄板遮住場地，對位會失敗，維持 `recalibrate`（不會硬套）。
   - **範本組合改變就重新對位**：自動快照記錄建立時 `calib/` 內全部可讀範本的識別碼
     （`template_set`，屬於快照內容的一部分）。新增、刪除或修改任何一個範本後，下一次
     `bxjudge run` 會把自動校準的影片重新對**所有**範本對位、取分數最高者；得到的快照不同
     就重新分析一次（已覆核的回合保留）。範本組合不變時沿用原快照，同機位影片照常共用。
     例外：被否決的組不會因為別處新增範本，就由同一個未改動的範本換個識別碼重建。
   - 黑色與透明下盤目前沒有確認的影片與手動真值，**尚未驗證**；有影片後各自手動校準一次、
     用 `--truth` 驗收。透明下盤允許對不準，但必須被標成 `recalibrate`，不得出現分數過關
     而區域明顯歪掉。
   - **否決**：`bxjudge review` 的每個回合會顯示該回合使用的校準與「校準組預覽圖」。
     若預覽顯示歪掉，按「不採用此組校準」（只有 `auto` 校準有此按鈕；按「恢復採用此組校準」
     可還原）。該組尚未覆核的回合不計入比分，但仍留在評估母體（`bxjudge eval` 會列出被否決
     的組數與影片數）；原判定不被改寫，已覆核的回合照常比對。VLM 不會拿被否決組的區域去問，
     這些回合的註記是 `vlm_unavailable: calibration vetoed`。被否決的機位會維持
     `recalibrate`，直到新增手動校準或範本改變。

7. **判定**：只處理新的或變更過的影片，結果寫到 `results/20261001/`（`--out` 可改）。
   `config/vlm.yaml` 的 `after_run: true` 會在判定後（偵測器釋放後）接著問 VLM；
   加 `--no-vlm` 跳過：

   ```bash
   uv run bxjudge run ~/bx/videos/20261001 --calib calib
   ```

   `--calib` 與 `--out`（預設 `results/<批次>`）必須是**各自獨立、互不包含**的資料夾（含符號連結
   解析後）：不可把結果寫進 `calib/` 底下，也不可把 `calib/` 放進結果資料夾，否則印
   `error:` 並以結束碼 2 結束，避免自動校準的快照被誤當成手動範本。

8. **（選用）單獨補問 VLM**：對尚未覆核、需覆核且還沒有成功 VLM 註記的回合再問一次：

   ```bash
   uv run bxjudge vlm results/20261001
   ```

9. **人工覆核**：開本機網頁（只綁 127.0.0.1），Windows 瀏覽器開
   <http://127.0.0.1:8501>，逐一確認或修正需覆核的回合：

   ```bash
   uv run bxjudge review results/20261001
   ```

   `bxjudge run` 若用了自訂 `--autocalib`，第 8、9 步也要給同一個 `--autocalib`（預設
   專案根目錄的 `config/autocalib.yaml`，不論在哪個目錄執行；明確給的路徑照原樣使用）；
   設定檔讀不到時整個指令報錯，不會默默不疊校準區域。

10. **隨機抽樣稽核**（R62/R64）：從原判「不需覆核」的回合隨機抽樣，回到第 9 步的覆核
   畫面用「抽樣稽核」篩選，把抽到的回合**全部**覆核完：

   ```bash
   uv run bxjudge audit-sample results/20261001 --n 80 --seed 1
   uv run bxjudge review results/20261001   # 篩選「抽樣稽核」，全部覆核
   ```

   只有抽樣全部覆核、回合數 ≥ `audit_min_n`，而且 Wilson 下界 ≥ 目標（95%）時，
   「未標覆核回合判定正確率」才會 PASS；只達到點估計時顯示 `INCONCLUSIVE (樣本不足)`。
   目標 0.95、信心水準 0.95 時，下界本身就需要至少 73 個全部正確的抽樣回合，所以建議
   `--n 80`（不能超過可抽的回合數）。若之後重新判定（rerun）同一個結果資料夾，請重新抽樣。

11. **評估與門檻校正**（見下方「評估與門檻校正」）：

    ```bash
    uv run bxjudge eval results/20261001     # reports/eval-YYYYMMDD.md
    uv run bxjudge tune results/20261001     # reports/tune-YYYYMMDD.md（只印建議 diff，不改設定）
    ```

注意：同一個結果資料夾一次只跑一個 `bxjudge run` / `bxjudge vlm`；試不同設定時每次用新的
`--out`，再以 `bxjudge eval <新資料夾> --truth results/20261001` 比較。

## 更多說明

標註、評估門檻校正與規則核對見 [docs/guide/labeling-eval-rules.md](docs/guide/labeling-eval-rules.md)；
選用功能見 [docs/guide/advanced-features.md](docs/guide/advanced-features.md)。

已有輸出要套用新設定或重新計時時，對新的結果資料夾執行 `bxjudge run --rejudge` 重新分析。
