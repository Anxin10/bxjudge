# bxjudge

Beyblade X 比賽影片的本機 AI 裁判（Local AI referee for Beyblade X match videos）。
所有程式都在 WSL2 Ubuntu 內執行；影片不上傳到任何雲端服務。

## 碰撞合併後的身分失效

兩顆近期實測陀螺的預測中心同落在一個合併框時，X／B 身分從該格失效，不能因原生
ByteTrack ID或位置接得上就恢復。時間與框邊距在 `two_bey_tracking.yaml`；沒有足夠的
獨立速度／外觀證據就維持 unknown。完整原框與匿名空間軌跡仍保留，判定入口不再消費
失效後的側別觀測，CSV亦清空該段side。新紀錄用 `physical_initial_sides` 加
`physical_identity_unknown_since` 表達初始歸屬與有效期限。

真實210958 f2796–2840 fixture含71筆原觀測：兩側從f2808失效，分開後26筆全unknown，
不再沿用已知換反的玩家身分。這是防止錯判的修正，沒有把unknown硬改成真值X。

## 第一批人工勝負真值（W4）

從 <http://127.0.0.1:8504/batch-outcomes/truth-slice/> 開始，沿用既有覆核頁
<http://127.0.0.1:8505/>。已剔除5段準備／收拾／極短候選；20段中先核15段覆蓋率
≥80%的候選，另5段74–79%是補充，不假稱全部達標或已是乾淨真值。
在右側修改「結束方式」與「獲勝方」並儲存；不確定或非對戰先留待覆核，不把未知當重賽。
答案只寫獨立的 `runs/truth-slice-20261027/`，原40支封存預測與既有人工答案保持原樣。
這是按覆蓋率選取的開發樣本，不是隨機稽核，不能外推全體準確率。

## 每格對位拒絕原因（D0）

新產生的動態幾何樣本保存 `rejection_reason` 與 `diagnostics`：包括特徵／內點不足、
變換越界、ECC未收斂、全域／區域邊緣支持不足，以及恢復／重新定位確認尚未完成。
`tracking` 與實際有嘗試的 `relocation` 分開保存量測；拒絕或確認前綴仍無有效幾何。
這些是實際失敗閘門，不能只因分數低就聲稱已確認遮擋或範本不符。舊資料缺細因仍為未知。

202834全片10,147格重播保留原決策；候選f505–9379的8,875格中，6,403格內點不足、
2,201格越界。診斷見 <http://127.0.0.1:8504/batch-outcomes/geometry-coverage/202834-d0/>。
相鄰影格的三個匹配對照通過，但尚未串成有漂移驗證的絕對幾何，不能當作D3已完成。

## 幾何覆蓋率診斷（W1）

入口：<http://127.0.0.1:8504/batch-outcomes/geometry-coverage/>。40 支原片的保存候選
範圍中，52 段雙顆候選共 105748／161095 格有有效幾何（65.64%），11 段達到 90%。
範本已經有 8 組，包含 214534、222226、230754、10/14，不需要因舊交接資訊重做這四組。
覆蓋率不是位置正確率；候選可能包含準備或收拾，尚非人工確認的發射到終結範圍。

```bash
PYTHONPATH=src:. .venv/bin/python scripts/report_geometry_coverage.py \
  runs/batch-outcomes-20261027/core --out runs/geometry-coverage-new
```

此診斷只讀封存結果，輸出須在原結果之外的新資料夾。它列出逐片、逐候選、逐範本場景
與最差 10 支影片的缺失來源格；重疊候選用聯集計數，缺格與尚未完成確認的格都算缺失。
報表目標門檻在 `config/geometry_coverage.yaml`，不會更動對位或裁判門檻。
歷史 `lost` 沒有保存拒絕細項，不能據此捏造遮擋／越界／範本不符；另存的失敗格重播
僅為選定影格的診斷，不會回填原幾何或外推全人口原因。W5 出現可比對的核心勝負前，
後續先處理幾何與人工真值，不再迭代 VLM 提示、做 ReID 研究或新增播放器功能。

## 非得分區離場的使用者規則（2026-10-27）

使用者已確認**對戰全程適用**：確認有陀螺先經非得分區離場，該回合重賽，後續極限／
擊飛事件不計分。設定明列 `battle_check.non_scoring_exit: redo`；規則版本含使用者覆寫
後綴，不宣稱這項自訂規則是官方規則。它不增加發射失誤次數。

`non_scoring_exit_reason` 按來源影格區間區分：確定在得分前的離場、得分後的離場、
先後不明。已成立的得分之後才發生的取走／離場，不倒推取消；先後區間重疊仍待覆核。
只有已確認的實際離場才可使用 `non_scoring_exit@X/B/unknown` 標記；單純跨到蓋面、
掉偵或校準失效不能當作已離場。新的事件提取仍待整合，這項變更先完成規則與時序接口，
沒有把原始模型列或已保存人工答案改寫成重賽。

## 本地事件交叉核查

`scripts/judge_actor_events.py` 讓模型只讀中立物件 ID 與原片事件，再核對兩端實際觀測、
有效校準及區域轉換。X／B 來自本地發射上下文；只有同格實測兩顆分開的盤內陀螺，
且模型也看到完整自由陀螺時，才可提出另一方的條件式身分。原框、失敗回應與未定位回合皆保留。

```bash
PYTHONPATH=src:. .venv/bin/python scripts/judge_actor_events.py \
  --results runs/batch-outcomes-20261027/core --out runs/actor-events-new
```

取樣與模型上下文設定在 `config/vlm_events.yaml`。輸出須為新資料夾，不改原結果；
只使用設定的本地模型，不在失敗時換模型。較早非得分離場的假設會先於後續得分，
先後重疊不推勝方；投影到盤外仍可能只是騰空，**不是已確認物理離場**。
中心進口亦未證明全身入袋、持續與未返回；所有事件與條件式勝負仍待覆核，不自動計分。

## 本地 VLM 的勝負候選（2026-10-27）

入口：<http://127.0.0.1:8504/batch-outcomes/local-vlm/>。這是本機模型直接讀原片抽格的
新輸出，與核心規則結果、Codex 視覺判讀及既有人工答案分開保存。批次目標為原40支
來源中的52段雙顆候選；實際進度與回答以該頁及 `summary.json` 為準，不表示52場有效對戰。

專案內 Ollama 服務啟動後，可以單獨執行，不重跑偵測器、不改原結果：

```bash
PYTHONPATH=src .venv/bin/python scripts/judge_local_outcomes.py \
  --results runs/batch-outcomes-20261027/core \
  --out runs/local-vlm-new \
  --config config/vlm.yaml --policy config/vlm_judgment.yaml
```

輸出必須是新目錄。可重複 `--case 20261001_201811.mp4:1` 明確選取開發案例；省略則
處理全部已保存的雙顆候選。新路徑只用設定中的主模型，不在失敗時換模型。現有
`bxjudge vlm` 備註功能仍是另一個用途，並不將備註偷偷轉成勝方。

每段先看全段與發射候選窗口，再保留全部初始上下文、加入中間格及模型指出區間的
較密取樣。候選窗口不是已確認物理分離。原片由f0順序解碼，圖片保留實際來源格編號；
原核心勝方、Codex結論與人工標記不送入提示。來源、代碼／設定與模型digest存manifest。
每次原始模型回應先存檔，再檢查決策、側別、事件端點與引用影格；格式錯誤、矛盾或
原片結束未入鏡卻宣稱incomplete，均另記失敗，不映射成重賽或已完成判決。

所有回答仍待覆核；候選分數來自專案規則，不寫成已採納比分。模型的not_bout亦非
已確認非對戰，不能刪除原回合或評估人口。兩次判讀不同會標示；模型一致也不是準確率。
目前已看到真實對戰誤分類與終結漏判，即使流程測試1428 passed／4既有xfailed，也不代表
視覺裁判準確率合格。此功能只讓本地模型實際參與並留下可查證候選，沒有解除核心閘門。

## 40 支原片試跑與身分修正（2026-10-27）

結果入口：<http://127.0.0.1:8504/batch-outcomes/>。版本 `83aabc3` 已完整處理凍結的
40 支原片：102 個候選列中，52 段確認雙顆觀測、50 段尚未確認雙顆，勝方預測仍為 0。
雙框共現也可能是準備或收拾，不能稱作 52 場有效比賽；本批尚無獨立裁判準確率。
影片串流合計 8976.92 秒、整批牆鐘 7698.20 秒，仍須逐片檢查 R66a，不能僅憑總和宣告速度合格。
原始結果封存於 `runs/batch-outcomes-20261027/core/`，來源與輸出雜湊列於
`baseline-completion-audit.json`。Codex 原片判讀另存，沒有寫成使用者真值或核心結果。

身分錨點現在以原始影格的陀螺與發射器框判斷接近關係；側別依有效的校準區域，或
同格兩個互不重疊的發射器左右順序取得。相互矛盾、單支無校準位置的發射器、重複重疊
框仍保留未知。必須連續實測支持；已觀測到自由移動的開頭不能改用後來拿起的畫面。
這只提供 X／B 身分，不填補失敗幾何，也不把框分離當作物理發射時刻。

同一批保存觀測的身分稽核：已知實體側別由 15 增為 32，原有已知側別無改錯或遺失；
66/66 既有人工核對點吻合。另以新碼完整重跑 4 支原片，66/66 核對點仍吻合，仍無勝方。
這是重複使用的開發回歸，不能外推準確率；測試並行時的重跑速度也不是正式速度驗收。
25 個針對性測試與全套 1407 passed／4 xfailed 通過。新碼、基準碼及四片重跑來源雜湊分開記錄於
`identity-context-real-audit.json` 與 `identity-context-pilot/manifest.json`、`audit.json`。
下方 `/two-bey/` 播放內容仍是先前兩支開發片的版本。

未指派框另記錄 `adjudication_eligible`，與實測軌跡使用相同發射階段條件；已排除的
掛載期框完整保留，不再單獨清除終結候選。仍符合裁判條件的身分不明觀測繼續否決勝方，
兩實體模式仍強制覆核。40 片保存觀測中，9 段含 189 筆被排除的未指派框；每段仍有其他
有效但未解釋的觀測，沒有因這項修正取得新勝方。這項比對只計算證據資格，沒有重新判事件。
修正後再完整重跑 201622、201712 兩片，3 個候選仍無勝方，66/66 既有身分點吻合；
所有原始觀測、40 片封存輸出及原片／政策／人工資料 SHA 核對通過。全套 1409 passed／4 xfailed。
證據另存 `phase-eligibility-pilot/`，沒有覆寫原四片或40片結果。

## 兩顆陀螺的連續軌跡

最新播放入口：<http://127.0.0.1:8504/two-bey/>，也在統一工作台導覽。這次影片直接來自
正式分析保存的兩實體軌跡，不另重算配對。兩個主要回合各有兩個持續身分；同一顆換
native ID或漏偵後接回仍沿用物件身分。發射器另以綠框顯示，零件保留作burst證據，不當第三顆。

實線是原片實際觀測，虛線／十字是兩個實測端點間的顯示插值；插值不寫入裁判CSV、
不支持落點／碰撞／停止事件，也不是遮擋期間的真實位置保證。多餘框、重複框與被拒
配對的原始觀測都完整保留在geometry證據，不因上限兩顆而刪除或算成已正確辨識。

正式模式明確啟用，輸出到新資料夾：

```bash
PYTHONPATH=src .venv/bin/python -m bxjudge.cli run \
  runs/fragment-refinement-20261026/inputs --calib calib \
  --out runs/two-bey-tracking-20261026/new-core \
  --trt-manifest weights/rfdetr-trt/manifest.json \
  --dynamic-calibration config/dynamic_calibration.yaml \
  --two-bey-tracking config/two_bey_tracking.yaml --no-vlm --rejudge
```

`two_bey_tracking.yaml`管理最多兩個身分、關聯、重複框及Kalman參數。連續兩個來源影格
都實際觀測到兩顆後，才發布該回合的正式物理身分；單顆或未連續共現的片段只保留
暫定觀測，不發布新的正式身分，也不產生發射／勝負分析。每個來源回合重設，
Hungarian一對一配對不會讓同一框同時屬兩顆；固定native歸屬矛盾時保留未指派，不悄悄
交換身份。首次建立身分仍可能誤認，因此新模式繼續要求覆核。沒有足夠身分／幾何或
仍有未解釋原框時，不輸出確定勝方。無新回合也保存原始偵測與動態幾何。

本次正式重跑：7279筆來源陀螺觀測中，7121筆進入兩實體軌跡、158筆另存未指派；
兩個主要回合分別只有2個物理ID。66/66已人工核對的X/B原框匹配，10組已確認同物件
案例的身分歸屬一致（重複原框可由同實體代表框表示，不當額外採納的裁判觀測）。
顯示補點516筆，全部標為非裁判證據。這些是兩支重複使用的開發片，不是獨立準確率。
完整批次wall time 242.48秒／253.43秒影片，正式R66a speed_ratio 0.957 PASS，關閉VLM；
測量時另有測試程序運行，不能外推其他片、硬體或含VLM的速度。

另一個原分段只有0.37秒、僅一顆觀測；12筆原始觀測全部保留於未確認片段，不新增
第三個正式身分，也不假造第二顆。該列仍在覆核及評估人口中；未修改舊人工判決，
也未擅自認定它是賽後收拾。兩實體整合與連續播放已接入，仍未具備全片逐框真值
或獨立裁判準確率驗收；三列新結果都要求人工覆核。

從正式新結果匯出可播放證據（兩個out都必須是新資料夾）：

```bash
PYTHONPATH=src:. .venv/bin/python scripts/export_two_bey_results.py \
  --results runs/two-bey-tracking-20261026/core-paired \
  --reference-pointer .superpowers/sdd/launch-state-20261026/current-confirmed-reference.json \
  --out runs/two-bey-tracking-20261026/new-export
PYTHONPATH=src:. .venv/bin/python scripts/render_two_beys.py \
  --reconstruction runs/two-bey-tracking-20261026/new-export/report.json \
  --config config/trajectory_video.yaml --out runs/two-bey-tracking-20261026/new-preview
```

原覆核頁的軌跡影片亦識別保存的`two_bey_tracking`證據；只在該模式畫虛線橋接，原CSV
不變。X/B未知時用兩種穩定顏色與物件編號區分，不把顏色直接當作側別真值。

## 本機統一工作台（目前開發資料）

固定入口：<http://127.0.0.1:8504/>。首頁集中最新試跑、先前人工覆核、X／B 身分對照
與中心接合原尺寸裁切；`launch-motion/` 等實驗頁保留為研究紀錄。此網址只在本機服務
運行時可用。頁面是建置時的快照，新的判定或人工修訂後須重新建置。

目前展示兩支開發原片，核心速度評估關閉 VLM；不能代表 109 支整批、獨立準確率或
完整 VLM 處理速度。舊資料的三回合已覆核，與新 AI 試跑的待覆核狀態分開呈現。
f1618 的深色 X／淺色 B 修訂另存版本，原提交、預測與瀏覽器草稿不改寫；修正標註
不代表模型已改善，軌跡碎片與中心／勾爪接合模型仍待驗證。

在 repo 根目錄建置既有資料的唯讀首頁（不啟動模型）：

```bash
PYTHONPATH=src .venv/bin/python scripts/build_workbench.py \
  --site runs/dynamic-referee-20261025 \
  --results runs/fragment-refinement-20261026/pilot \
  --human-results runs/trt-final-20261025/pilot \
  --evidence runs/fragment-refinement-20261026/evidence \
  --core-commit 6e68fcc250f73dd181803a90cef2e7d8a2ee3041 \
  --review-set runs/reid-review-20261026/site-ready \
  --appearance-set runs/reid-appearance-20261026/diagnostics-final \
  --contrast-set runs/reid-contrast-20261026/diagnostics-final \
  --challenge-set runs/reid-challenge-20261026/site \
  --two-bey-set runs/two-bey-tracking-20261026/paired-preview
```

`--core-commit` 必須記錄輸入結果實際量測的版本；上例的核心量測版本為 `6e68fcc`。
文件或核對工具的新提交不能冒充已重新量測裁判核心。首頁來源與 SHA-256
列於 `/_workbench/snapshot.json`。建置保留舊首頁與子頁備份，僅增加工作台導覽；
不會重設原逐格核對頁的儲存端點或草稿。

### 新增的斷點局部核對

固定入口：<http://127.0.0.1:8504/reid-review/>。只核對保留的五組疑點，不重做原本三回合。
播放來源影格計數的前後短片，比較①／②候選框，選同一顆／不同顆／看不清及同回合
狀態，再勾「這組已核對」。X／B、可見性與是否碰撞是獨立欄位，可以留空或選不知道。
同回合的局部答案不會自動建立全場已驗證 scope，也不自動建立外觀參考庫或改判決。

按「保存到專案」後，收到檔名和 SHA-256 才是已存版本；瀏覽器草稿與匯出 JSON 都
不等於專案保存。每次修訂另存 `runs/reid-review-20261026/submissions/reid-<sha>.json`，
可用「載入選定版本」跨瀏覽器手動取回，不自動覆蓋目前草稿。原 identity-audit 頁與
它的提交仍使用原本獨立資料夾和既有格式，沒有轉換舊提交或改寫原人工標記。

建立新核對版本（輸出必須是新目錄，不覆寫已有核對集；沒有模型推論）：

```bash
PYTHONPATH=src .venv/bin/python scripts/build_reid_review.py \
  --diagnostics runs/fragment-refinement-20261026/evidence/fragment-refinement/diagnostics.json \
  --tracks-dir .superpowers/sdd/identity-continuation-20261026 \
  --out runs/reid-review-20261026/site-ready \
  --coordinate-size 1280 720 --context-s 1.0
```

來源影片、原始 native boxes、FPS、裁切影格範圍、影片／端點圖 SHA 和產生器版本存於
`manifest.json`；目前輸入 cache 是之前保存的原始觀測，未包含新的人工作證。
上方工作台建置會驗證媒體並安裝這份核對集；新 manifest 不會靜默取代已安裝的不同版本。

同一個服務維護逐格核對與兩套斷點核對的保存端點，僅綁 localhost；若8504已有服務勿重複啟動。

```bash
PYTHONPATH=src .venv/bin/python -m bxjudge.workbench_server \
  --site runs/dynamic-referee-20261025 \
  --identity-submissions .superpowers/sdd/identity-launch-20261025/submissions \
  --reid-set reid-review runs/reid-review-20261026/submissions \
  --reid-set reid-challenge runs/reid-challenge-20261026/submissions --port 8504
```

此層僅收集人工疑點資料；全場範圍確認、凍結參考庫、外觀提案與重新辨識匹配仍未實作。

### 新增六組困難候選核對

入口：<http://127.0.0.1:8504/reid-challenge/>。這是新一批，原五組答案保留，不用重做。
第1–2組是重疊前後、第3–4組是同格外觀相近、第5–6組是低清晰度測量候選。只選
同一顆／不同顆／看不清、同回合狀態，再勾已核對並保存。X/B與可見性可以不知道；
若框到發射頭或其他物件，選無法判定並在補充註記說明。原圖／短片只作人工核對，
同nativeID、IoU、低Laplacian或色彩相近都不當成真值。

挑片參數在`config/reid_challenge.yaml`：每6來源格取樣，每類最多2組；排除已核對的
區間與鄰近重複片段。740個取樣來源格、1215個原生框量測選出6組，來自相同兩支開發片，
不是同色陀螺或碰撞的獨立驗收。候選範圍及缺額明列，沒有預填人工答案或改判決。
`selection.json`保存挑選原因／參數／來源SHA；媒體manifest與新瀏覽器草稿各自綁SHA。

```bash
PYTHONPATH=src:. .venv/bin/python scripts/select_reid_challenges.py \
  --tracks-dir .superpowers/sdd/identity-continuation-20261026 \
  --prior-manifest runs/reid-review-20261026/site-ready/manifest.json \
  --reference-pointer .superpowers/sdd/launch-state-20261026/current-confirmed-reference.json \
  --appearance-policy config/reid_appearance.yaml \
  --selection-policy config/reid_challenge.yaml \
  --out runs/reid-challenge-20261026/new-selection
```

再用`build_reid_review.py`對`new-selection/selection.json`產生**新**核對集，並將同一份
selection.json複製入該新資料夾；工作台建置時驗證它與manifest及全部來源一致。
已安裝版本不同時拒絕覆寫，保留現有草稿。服務以明確`--reid-set NAME SUBMISSIONS`配置
兩個隔離資料集，不掃描任意資料夾自動接收新manifest。

### 多格外觀離線診斷

入口：<http://127.0.0.1:8504/reid-appearance/>，亦已加入工作台。五組新核對全部已保存：
四組同一顆、一組不確定。修訂 `25d81dd4` 記錄第5組藍框①來源f1713是發射頭端；
僅排除該端點作為陀螺外觀來源，不將整條原生軌跡改類別，也不把不確定改成不同顆。

新工具用原始未畫線BGR影格，依原生觀測取端點前後最多8張裁切；不補缺格、不重新跑
偵測器。色彩直方圖與灰階梯度距離分開記錄，並保留模糊測量、重疊物件及品質拒絕。
第4組已確認同顆但色彩距離較大，不能只看顏色接線。裁切可見性、可信X/B參考及全回合
scope仍未驗證；這五組沒有不同顆對照，不設定匹配閾值或宣稱準確率。全部`applied=false`。
參數在 `config/reid_appearance.yaml`，只用於離線取樣與描述子，不是已校準的裁判設定。

重播目前資料（輸出必須是新的目錄）：

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_reid_appearance.py \
  --manifest runs/reid-review-20261026/site-ready/manifest.json \
  --labels runs/reid-review-20261026/submissions/reid-25d81dd400d7521a2d172db7bfbd9c5df5ddbda8016e47372d332700873b705a.json \
  --endpoint-annotations runs/reid-appearance-20261026/diagnostics-final/endpoint-annotations.json \
  --config config/reid_appearance.yaml \
  --out runs/reid-appearance-20261026/next-diagnostics
```

JSON凍結來源／設定／人工修訂／程式與crop雜湊；CLI輸出完整離線工具wall time，頁面另標
取樣特徵時間的範圍。這不是整套裁判時間，不沿用舊核心速度PASS到新模組。

### 沿用人工標記的不同顆對照

<http://127.0.0.1:8504/reid-contrast/> 已加入統一工作台，不需重做核對。工具只取人工確認、
同一影格中唯一X與B皆為陀螺且已分離的原框；掛載／未知／壞偵測不作負對照，不沿native
ID外推其他格。最新f1618修訂包含在來源中。原38個標註影格保留人口統計，6個符合條件，
其餘10個缺乏唯一雙側、22個含未分離狀態；6對來自2支影片、3個相關區間，不是6場獨立比賽。

這批同顆案例的色彩距離中位數範圍0.174–0.398，不同顆同格距離0.671–0.826；梯度紋理
範圍有重疊。兩種取樣條件不等價、同片資料相關，也尚未包含足夠同色陀螺，因此不選
匹配閾值、不計新準確率或啟用自動接續。所有`applied=false`。來源原片、人工參考修訂、
原始觀測manifest與裁切皆綁SHA，獨立保存到新資料夾；工作台發布前重新驗證。

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_reid_contrast.py \
  --reference-pointer .superpowers/sdd/launch-state-20261026/current-confirmed-reference.json \
  --identity-manifest runs/dynamic-referee-20261025/identity-audit/manifest.json \
  --appearance-set runs/reid-appearance-20261026/diagnostics-final \
  --config config/reid_appearance.yaml \
  --out runs/reid-contrast-20261026/new-diagnostics
```

### 軌跡接續現況

[軌跡與身分通用規則](docs/superpowers/specs/2026-10-26-tracking-general-rules.md)
已區分現有實作與待補能力；五組未接續案例保留作證據，不自動當成不同物件的真值。
[遮擋與部分碰撞後重新辨識規格](docs/superpowers/specs/2026-10-26-occlusion-collision-reidentification.md)
已制定；第一版規劃為只產生候選與證據的 shadow 模式，尚未實作。

動態裁判以發射前的陀螺與發射器區域連續觀測建立 X／B 身分，再沿已通過接續檢查的
成員繼承。中途跑到另一側，不會按當時畫面左右重新命名；沒有足夠證據就保留未知。
這項區域證據不代表機械接合，也不決定物理分離時點。

建立身分需 `held_s` 對應的連續原始影格，陀螺在發射區聯集、近身發射器只有同一個
區域身分且沒有矛盾。真漏偵可重新累積；已有觀測但對位失敗時，不能跨過後定錨。
身分與證據寫在 geometry 的 `identity`；有效但未知身分的陀螺會要求覆核並禁止自動勝方。
舊的靜態模式仍是明確的獨立流程，不能把其半邊判定當成動態身分錨點。

初始兩支影片有30條陀螺候選軌跡，前輪為27條；本輪相同原始偵測重播與完整原片重跑皆為26條，保留
原有三次接續、新增一組重複追蹤的交接。正式速度 gate 為0.897 PASS（兩支開發原片、
關閉VLM）；三回合仍待覆核，沒有整體準確率通過的依據。
66 筆重複使用的開發影格觀測中，目前38筆指派與人工一致、28筆未知、0筆指派不一致；
未知不計入判對，也不是獨立身分或勝負準確率驗收。

本輪接續使用最近 `link_velocity_frames` 個來源影格內實際位移的中位數；有缺格時
除以實際來源影格差，不補造座標。重疊交接使用真實共格位置，非重疊的短缺口仍預測。
方向、距離、時間等門檻數值保持；估計方法的改變仍會改變接受集合，須以原片驗證。
`phase.joins[].motion` 保存取樣影格、速度、位置依據與殘差。工作台的
`/fragment-refinement/` 列出原片對照與五組仍被拒絕的代表案例；這些拒絕原因不是人工
身分真值，不能據此認定兩段必定不同。長遮擋不靠相近位置或顏色硬接。

接續沿用互相唯一、速度、方向、尺度與位置條件。短暫重疊必須每格有雙方實際框且
IoU 達 `config/launch_phase.yaml` 門檻；相反身分錨點拒絕接續。共用影格保留較早
軌跡的觀測，被取代原始框及分數另存於 geometry 的 `phase.joins`，成員 ID 在
`phase.members`。不插補漏格、不中斷追溯，也不以此推定物理分離。接續結果仍加上
`identity_join_requires_review`，保持人工覆核。其餘軌跡碎片仍待解決。

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

3. **標註**（見下方「本機標註」）：抽格 → SAM 3.1 預標 → Label Studio 人工修正 → 匯出 COCO：

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
### RF-DETR TensorRT 後端（本機選用，無自動 fallback）

已驗證環境：WSL Ubuntu、RTX 3090、CUDA 13.2 PyTorch、RF-DETR 1.11.2、TensorRT-cu13 10.16.1.11。TensorRT 引擎只用於同 GPU 型號／compute capability 與相同套件版本；更換環境或 checkpoint 後重新匯出。

```bash
cd /home/<user>/bxjudge
uv sync --frozen --extra tensorrt
# out 必須是不存在的新目錄；匯出完成才會發布 manifest.json。
uv run --no-sync python scripts/export_rfdetr_trt.py \
  runs/train-rfdetr-20261023b/fit/checkpoint_best_total.pth \
  --out weights/rfdetr-trt
uv run --no-sync bxjudge run ~/bx/videos/20261001 --calib calib \
  --trt-manifest weights/rfdetr-trt/manifest.json \
  --out results/20261001-trt --rejudge
```

`config/tensorrt.yaml` 管理匯出設定；解析度仍唯一來自 `config/thresholds.yaml`。這是 FP16-enabled mixed precision（輸入 FP32），不是所有層都強制 FP16。原 RF-DETR RGB 前處理、座標／類別後處理、信心門檻及裁判規則不變。訓練與 `compare` 繼續使用原生後端，不會改寫原選模檔的量測。

`--trt-manifest` 明確啟用 TensorRT；沒有這個參數則明確使用原生 PyTorch。指定後若引擎／權重雜湊、解析度、版本、GPU 或 tensor contract 不符，啟動會失敗，不切回其他後端。每回合 `detector_provenance.runtime.detector` 記錄實際引擎與雜湊。切換後端請使用新結果目錄或 `--rejudge`，因為普通重跑會保留未變影片的既有判定；已人工覆核的原始判定仍受保護。

96 格實驗量測只包含 detector adapter，排除解碼、追蹤、事件、VLM。正式速度以 `bxjudge eval` 的完整牆鐘時間判斷；資料未經人工真值／稽核之前，不宣稱裁判準確率達標。校準未匹配影片繼續標示需重校。

### 軌跡動畫

覆核頁「該回合軌跡圖」會從已存軌跡 CSV 及判定時的校準快照產生 H.264 MP4，可播放、暫停及拖曳。背景使用原始影片，能看見真實陀螺；觀測線條按保存的原始影格時間以1倍速累積；圖下 `Video t` 是原影片時刻。X 橘、B 藍、未確認灰色；首末標記指每段軌跡的實際觀測，缺格不插點、不連線。原 PNG 放在「靜態軌跡總覽」。

軌跡輸出會在同名 JSON 記錄座標影格尺寸及 CSV 雜湊；缺少紀錄或雜湊不符時不猜尺寸，需重新產生軌跡輸出。首次檢視需要產生動畫，之後讀取 `trajectories/playback/` 快取。原影片內容（SHA256）、CSV、校準幾何／參考圖、回合時間、繪圖程式或 `config/trajectory_video.yaml` 改變時會產生新的快取。輸出為 H.264/yuv420p、faststart，使用專案既有 ffmpeg/libx264。編碼失敗會明確顯示訊息，不發布半成品。影片由原始第0格開始按保存的影格編號對齊，避免時間跳轉造成錯位；若來源影片無法讀取會明確告知，不切回固定背景。此功能只解碼並建立覆核檢視媒體，不重跑模型、不修改判定／人工覆核／原軌跡／校準／計時紀錄。

盤面跟隨預覽以判定時的校準參考圖為固定基準，逐格用盤緣 ORB 特徵與 RANSAC 初始化，再以 ECC 仿射對位估計小幅平移、旋轉及角度壓縮，移動全部八區；不累積前一格的位移。只有最終對位的內點比例、雙軸尺度、位移範圍及盤緣整體／多區邊緣相關分數都通過 `config/trajectory_video.yaml` 的 `stadium_follow` 設定時才畫綠線。圖下 `Follow preview` 數字是邊緣相關分數，不是裁判信心或準確率；失敗時顯示 `Alignment lost: zones hidden` 並隱藏區域，下一格重新對原基準圖估計。這一階段僅改變播放預覽，保存的影格座標軌跡、靜態圖、原始晃動證據及裁判結果均不重算；歷史軌跡不是盤面穩定化後的座標。畫面真正失焦仍無法恢復。

### 追蹤候選與裁判證據

`config/detection.yaml` 的 `candidate_confidence` 僅供 `run` 的 ByteTrack 關聯；原 `confidence`（0.3）仍是裁判證據門檻。正式預設兩者皆為0.3：0.1候選門檻在兩支實片未改善追蹤，未採用為預設。實驗可透過完整的 `--config` 設定明確降低候選門檻，以下安全邊界一律適用。弱框不寫入交給裁判的軌跡、CSV 或旋轉影像，不因低分配對取得內部 ID 就提升成可信軌跡。只有高分種子再獲得高分配對才會輸出。長時間沒有高分觀測時，依 ByteTrack 原有失聯秒數上限切斷關聯，低分更新不能無限續命。

`run` 的 provenance 記錄兩種門檻。偵測器獨立預設及 `compare` 的 `evaluation_confidence` 不變。設定缺欄位或讓弱框門檻／啟動條件繞過證據門檻會明確報錯。這個修正處理短漏偵的接線，不代表碰撞後的 X/B 身分已驗收；duplicate、missing、gap、id-swap 覆核仍保留。已人工覆核結果不受覆寫，修正比較應使用新的結果資料夾。

### 動態校準參與裁判（明確啟用）

`bxjudge run <影片資料夾> --calib calib --out <新的結果資料夾> --dynamic-calibration config/dynamic_calibration.yaml --trt-manifest weights/rfdetr-trt/manifest.json --no-vlm`

小幅對位門檻仍取 `config/trajectory_video.yaml` 的 `stadium_follow`。持續拒絕才依 `config/dynamic_calibration.yaml` 定時對原始快照做較大範圍重新定位；候選須高分且多次一致，才建立新段。失敗影格保持未知，不延用上一格或回填。短暫遮擋後也需連續有效觀測才恢復顯示，避免接受／拒絕交替閃爍。

動態模式的裁判觀測逆變換至基準盤面，原始影片裁切與CSV的x/y保持來源影格座標；CSV的zone依裁判座標計算，無幾何為unknown。原始盤面晃動仍相對本影片first-second frame量測，沒有先穩定畫面再量晃動。`geometry/` 使用內容雜湊檔名保存逐格矩陣、有效狀態、重新定位段、設定、來源影片雜湊及發射／碎片處理紀錄；回合以path及SHA256引用。動態回放、事件截圖與VLM影格讀取該次保存的幾何，不能重新套用當前校準。證據缺損則明確提示，不回退成固定區域。

`config/launch_phase.yaml` 僅在動態模式啟用：以實際發射器身體觀測確認掛載及分離，排除已證明的前置掛載觀測；發射器消失不算分離，不以落地或終結倒推發射，也不延長既有3秒期限。短缺口碎片僅在位置／速度相容且前後候選唯一時接續，沒有插點；任何接續仍標記identity_join_requires_review，不能將ID數量下降當成X/B真值驗收。未觀察到分離者標記launch_release_unobserved。動態幾何缺失會阻止自動出界結論，重新定位段須覆核。

新模式使用新out做比較最清楚；相同out在模式或設定改變時會重新分析，已覆核回合仍依原有保護規則保留。需重新測量含逐格對位的完整wall time；舊TensorRT速度PASS不適用本模式。原先靜態模式保留作實驗基準，動態失敗絕不自動切回靜態。

2026-10-25效能驗證：`stadium_follow.ecc_iterations` 最大迭代由50改為10，接受分數、位移／旋轉／雙軸尺度、內點比例及多區門檻不變。18支影片376格的接受／拒絕與原50次皆相同（221／155），相對原對位的平均邊界差最大0.35px；232控制例及7/8恢復控制結果維持。兩段各8秒衍生片在共用已載入正式TensorRT模型下，完整處理44.77→32.80秒，仍超過16秒影片總長，不能外推整場即時驗收；完整253秒的886秒舊測量沒有重寫成優化後結果。原判決及人工覆核不變。

2026-10-26串流預取：CPU 解碼、原始盤面晃動及動態對位，與主執行緒的單一偵測器重疊執行；最多排隊兩格，維持來源影格順序。取消會終止當次解碼子程序並等待回收，不把取消當成少格完成；來源／關閉錯誤仍會回報。兩段相同8秒衍生片、共用已暖機正式TensorRT的單次成對測量，完整wall由33.37降至25.94秒（減少22.3%）；判決欄位、軌跡與480格幾何接受狀態一致，幾何浮點數非逐位元相同，探測點差最大0.000173px。仍超過16秒影片總長，亦未包含8.54秒模型初始化，不能宣稱整場即時或準確率達標。既有out須以 `--rejudge` 真正重算才有新計時；詳見 [目前優化狀態](docs/optimization-status-2026-10-26.md)。

發射時序診斷會在新的 `geometry` 記錄 `phase.diagnostics`：原始首見格、固定發射器ID、造成發射判定中止的原因／影格，以及後續真正連續的 `box_clear_runs`。這些只表示兩個偵測框分開；框分開時點不能當成精確物理發射，也不會補入 `releases`、裁掉原軌跡或延後三秒期限。舊快照不補寫，已有分析需明確 `--rejudge` 才會產生新紀錄；本輪實片對照只重播已保存偵測，不重跑整片。


2026-10-26後續原片驗證：`config/detection.yaml` 的 `opencv_threads: 4` 在模型初始化後、解碼前套用，正式判決記錄requested／actual。完整warmup後正反序的短片試跑（每模式2次），預取1執行緒平均33.06→4執行緒26.52秒；仍不是整片速度證據。另以兩支原始整片、新CLI程序實測核心動態流程（`--no-vlm`）：單片209.49／153.85秒與120.39／99.57秒，batch343.64秒／253.43秒，R66a speed_ratio1.362，仍FAIL；3/3回合需覆核。最終1282測試通過、4既有xfail；詳見[原始資料優化狀態](docs/optimization-status-2026-10-26.md)，不把舊886秒改寫為優化後結果，亦不宣稱含VLM即時或裁判準確率已達標。


2026-10-26最新驗證：將既有raw-motion和動態對位分成兩個有界CPU階段，並與單GPU偵測重疊；原演算法、迭代及門檻皆不變。兩支開發原片重新以fresh CLI、`--no-vlm`跑完整流程：129.97／153.85秒、78.85／99.57秒，batch含模型載入213.19秒／253.43秒，正式speed_ratio=0.845，**僅本次核心速度PASS**。三回合仍全部需覆核，裁判準確率和含VLM時間未驗證。與直接上版判決、軌跡、raw-motion證據及7611格接受狀態相同；浮點矩陣非逐位元相同。1288測試通過、4既有xfail，仍沿用原分支；[完整範圍與未完成項目](docs/optimization-status-2026-10-26.md)。既有out請以 `--rejudge` 重新分析才有新計時。


### 初始發射方證據與短暫漏偵

`config/identity.yaml` 定義來源時間的 `context_wait_s` 與重複發射器框的
`body_duplicate_overlap_min`。只在初始位置尚未知、沒有已觀測自由運動證據時，
等待短暫缺失的發射器；逾時、已自由運動或互相矛盾的訊號不會在後來拾取時取得身分。
重複本體框須逐對重疊，跨越兩位玩家的寬框保留未知。
初始錨定只代表發射方；碰撞合框後的 `identity_unknown_since` 仍會切斷側別證據。


### 時序盤面幾何證據

動態模式使用 `config/temporal_geometry.yaml` 的來源秒數及來源像素界限。
盤緣光流須通過前後追蹤一致性、分區支持與影像橢圓檢查，並在期限內與固定範本
重新錨定閉環核對；每格仍檢查當下可用的固定範本證據，週期僅限制光流錨點重設，不跳過模板驗證。`temporal_pending` 在閉環完成前沒有有效幾何，
只有核對一致的觀測才成為 `temporal_verified`。任何不一致、來源缺格或逾期都丟棄待確認鏈。
前景遮罩使用該格的實際偵測框及相機運動補償後的差分；原始碰盤晃動訊號仍保留。

短缺口可另存 `bridged` 估計及兩端來源格，但不計入有效覆蓋率、不進 X/B 裁判分析，
也不在軌跡影片顯示有效綠線；包含橋接的回合不得自動產生終結或勝方。
資料同時保存原始拒絕原因，橋接不會改寫成已觀測成功。
新設定與實作雜湊寫入分析政策。已人工覆核的舊輸出保持原狀，請在新的結果資料夾
明確使用 `--rejudge` 比較；舊速度測量不能代替加入此流程後的完整計時。
