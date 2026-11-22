# 8504 工作台 HUD 重設計

狀態：已實作（feat/local-judge）。

## 為什麼
8504 工作台原本是偏文字的研究報告集合，看不出「裁判這回合判了什麼、為什麼退回人工」。task7/core 的 56 回合全部 needs_review，覆核才是常態，所以畫面以「覆核原因落在哪個流程節點」為主軸，並套用參考影片的 HUD 視覺語言。

## 使用者決定
- 範圍：新增逐回合裁判視圖 `/judge/`，並重寫可存檔核對頁（reid-review / reid-challenge）。
- Live 畫面：原影片播放＋即時疊框。
- 版本並排比較：這次不做。
- 動效：適度且可關閉（開關＋prefers-reduced-motion）。

## 反目標
存檔 payload、localStorage key、`/…/save`、`/…/saved` 被改；原片、人工真值、`calib/`、run 輸出被寫入；動效干擾閱讀；開頁觸發重運算。

## 組成
- `src/bxjudge/web/bx_theme.css`、`bx_fx.js`：共用色彩 token（青＝採用/焦點、橘＝待覆核、紫＝核心/鎖定、金＝累積證據、珊瑚＝軌跡/fallback、灰＝非作用）、面板角標、疊框語法（框＋跟隨標籤＋輪廓＋填色/斜線＋主目標外光＋四角 L 標）、流動虛線與光點、數值收斂、掃描帶、降級狀態短暫 glitch。`body.fx-on` 控制所有動效。
- `src/bxjudge/judge_view.py`：只讀 run dir，產生每回合 payload：判定摘要、review reason → 流程節點（偵測／兩顆追蹤／場地幾何／身分／終結判定／其他）、軌跡（先保留實體 ID 與連續觀測區段，再按 side 呈現）、two_bey 主框與未指派候選框、場地多邊形（只取發射來源格的已驗證 alignment 作固定參照；缺失或 bridged 就隱藏，不借遠格；僅在參照來源格顯示，明示非逐格對位）、影片 stream start_time 偏移（瀏覽器以 PTS 呈現，pipeline 以 frame/fps 計時；實測 0.939s）。

### 2026-10-30 證據正確性修正（待再審及部署）

- 框的 X/B 標記依保存的實體映射、初始側別、該格 identity_valid 與未知區間決定，不用框包含其他軌跡中心猜側別。
- 軌跡點在降採樣前保留 source frame 與連續區段標記；前端按 entity ID／side／segment 分線，未知實體使用未知樣式，不跨陀螺或缺口連線。
- 固定發射參照與當格幾何分開；尚未實作逐格校準線播放時，不以「已對位」暗示持續有效。
- 幾何使用每回合 geometry_evidence 的確切 path/SHA256/calibration_id，驗證 source 與載入不可變校準快照，不依檔名字典序或目前 video 設定猜版本。
- `src/bxjudge/web/judge_view.js`：回合清單、Live（影片＋SVG 疊框、模型/偵測細節、Output 與 Fallback 卡）、Background（流程節點＋回饋迴圈＋證據庫）、底部狀態列。`data-state=judged|review` 一次切換整組視覺。
- `src/bxjudge/workbench_theme.py`：頁面外殼、總覽頁（由已驗證的 `_workbench/snapshot.json`＋judge index 產生）、reid 核對頁、對舊頁面注入頂部導覽（不動其 JS/表單）。
- `src/bxjudge/workbench_restyle.py`：安裝到站點；先驗證 reid 頁內嵌 sha 與 manifest 一致才寫入；每個被取代頁面留一次性 `.before-v2` 備份；`/judge/` 只取代自己建立（有標記檔）的目錄。
- `workbench_server.py`：`--video-run` 讀 run 的 videos.json，`GET /video/<name>` 只供白名單原片，支援 Range；靜態檔加 `Cache-Control: no-cache`，避免重產頁面後讀到舊快取。

## 我的決定（可再議）
- identity-audit 由一次性 sdd 腳本產生、內嵌存檔 JS，只注入新外殼，不重寫表單。
- 總覽頁不重跑 `scripts/build_workbench.py`：其上游來源 SHA 已變（appearance source SHA mismatch），重跑會被自身驗證拒絕；改由 snapshot 產生，舊總覽保留為 `index.before-v2.html`。若之後重跑 build_workbench，需再執行 restyle。
- 原計畫的獨立 `scripts/build_judge_view.py` 併入 `python -m bxjudge.workbench_restyle`。

## 部署
```
.venv/bin/python -m bxjudge.workbench_restyle --site runs/dynamic-referee-20261025 \
  --judge-run runs/next-four-tasks-20261027/task7/core \
  --store identity-audit .superpowers/sdd/identity-launch-20261025/submissions \
  --store reid-review runs/reid-review-20261026/submissions \
  --store reid-challenge runs/reid-challenge-20261026/submissions
.venv/bin/python -u -m bxjudge.workbench_server --site runs/dynamic-referee-20261025 \
  --identity-submissions .superpowers/sdd/identity-launch-20261025/submissions \
  --reid-set reid-review runs/reid-review-20261026/submissions \
  --reid-set reid-challenge runs/reid-challenge-20261026/submissions \
  --port 8504 --video-run runs/next-four-tasks-20261027/task7/core
```
