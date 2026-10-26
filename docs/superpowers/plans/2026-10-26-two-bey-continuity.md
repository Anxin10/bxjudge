# 每回合兩顆陀螺的穩定身分與連續軌跡

使用者新目標取代先前純外觀門檻工作：一場只有兩顆，軌跡需整合連續。已確認兩顆指陀螺，發射器另追蹤。沿用現有分支，不再開分支或擴充核對工具。

設計：每回合固定兩個實體槽位，cv2.KalmanFilter預測位置、SciPy Hungarian一對一配對；位置、有界更新的外觀與nativeID只作關聯線索，不當可信X/B參考庫。高IoU重複框不建立第三顆；所有原框及未配對框另存，不刪原始證據。僅從實際偵測寫observed點，實測點間的缺格用明確interpolated虛線呈現，無已觀測終點不外推補成真值。

先用保存native偵測離線重播兩片3回合並產生可播放預覽，驗證固定上限／不跨回合／一對一／來源完整／人工X/B與同顆案例。原裁判及原人工答案保持不變；這是新的兩實體重建，不藉新軌跡宣稱既有判決已正確。

- [x] RED/GREEN：同顆換nativeID不新增實體、漏偵後接回、重複框不生成第二個假物件、兩顆交叉不交換、超過兩框保留unassigned、回合獨立reset。
- [x] 缺格連續輸出：每一格最多兩實體，觀測／插值區分，沒有量測不能成裁判證據；不修改傳入原偵測。
- [x] 實片重播與正式CLI接線：保留兩類視覺層（陀螺、發射器），原bey_part留作burst原證據、不併入完整陀螺身分；不丟未知／誤偵分母。
- [x] 自動測試、獨立審查發現親驗修正、正式原片重跑、統一工作台預覽。
- [x] 未確認片段不發布額外正式身分：連續兩格實測共現才發布兩實體；單顆原觀測完整保留、留在覆核與評估人口，不產生發射／勝負分析。無須先替使用者認定它是賽後收拾，才能防止第三個正式身分。

全片逐框身分真值及獨立裁判驗收仍是後續限制；本次完成的是正式兩實體整合與連續顯示，不是保證遮擋中的物理路徑或全部勝負正確。

## 驗證與裁決紀錄

- BASE d61041e，沿用feat/local-judge，無新分支／worktree／push。先前外觀可行性草稿移至本工作scratch的deferred-feasibility，不讓缺module的草稿污染正式tests。
- 核心、pipeline與官方viewer各先出RED；獨立審查4個Important均親自RED→GREEN：invalid descriptor先污染狀態；未接受的native owner污染alias；未知原框被容量移除後仍可輸出勝方；raw_source_tracks在phase之後才保存。no_round缺geometry與held prefix被裁掉另以RED修正。
- 獨立審查員檢查過核心、第一版顯示、第一版正式接線；最後一次追加覆審因工具額度中止，未聲稱最後匯出／官方viewer已獨立覆審。最後修改以作者自審＋合成與實際媒體檢查驗證。
- 全套1397 passed、4既有xfail、1既有FutureWarning；新增正式匯出與顯示修正後64項相關測試再通過。各新Python檔及修改檔ruff通過；原測試129、動態整合15及最後完整結果在scratch logs，不把不同版本的數字累加。
- 正式最終CLI：core-final，動態幾何＋TensorRT＋two-bey＋no-vlm，兩支完整原片，250.307秒／253.428秒影片，正式speed_ratio0.993842 PASS；同時有測試程序，不能外推整批或含VLM。
- 保存7279筆來源bey框：7133採納、146保留未指派；逐筆皆能對回phase前native框，沒有丟棄／插造原始來源。兩主要回合CSV各只2個物理ID；第三候選段只1個，保留未知。
- 66/66已確認X/B原框都配對到單一實體且雙方不共享；10個same案例的所有native歸屬一致，部分端點為保留的重複／同實體重疊框而非額外裁判觀測。這是相同開發片回歸，不是獨立accuracy。
- 516個缺格顯示點全部tagged interpolated/non-evidence；原CSV只實測。每實體首末實測間逐來源格連續，沒有向前／後虛構整段；缺格路徑不是物理真值。
- 正式export只讀geometry，不重跑配對；published-preview及原覆核generate_trajectory_video都實跑出可編碼MP4，包含source frame索引、SHA與輸出幀數驗證。兩個原片正常review renderer亦已實際生成；未使用瀏覽器視覺自動化。
- 最終測量的source SHA在final-runtime-source-sha.json，formal-audit.json逐條對照需求。67個受保護檔案（原影片／結果／人工／既有非本次.gitignore等）雜湊不變。原三回合已覆核、五組修訂與新六組答案均保留。
- 上列core-final／published-preview為2bf3878的歷史結果，保留原檔；下列為修正後的新結果，不混用版本。

## 單顆候選段的正式身分修正

- RED→GREEN釘住單顆段發布新物理ID、產生發射分析、覆核頁誤顯示正式軌跡三處問題。另測試依序出現兩個物件及不連續雙框不能確認共現。
- 設定`pair_confirmation_frames: 2`：連續兩個來源影格都有兩個實測物件，才確認該回合兩實體；否則保存hypothesis_observations及全部原框，不發布正式ID。未確認列仍在評估／覆核人口，不以排除困難樣本換取通過。
- 原片core-paired正式CLI重跑：兩主要回合CSV分別只有41/42及30/31；單顆候選段沒有正式ID、發射／落點／勝負分析，12筆原始觀測全部保留，沒有跨回合硬接。
- 來源7279筆＝正式實測7121＋未指派158；66/66人工X/B原框及10/10同物件案例歸屬一致。516筆顯示插值不作裁判證據；每實體首末實測間來源格連續。
- 批次wall 242.4769秒／253.4276秒影片，正式R66a speed_ratio 0.95679 PASS（關閉VLM、測試並行，限本兩支開發片）。所有新列仍待覆核，不宣稱自動裁判準確率。
- 全套1402 passed、4既有xfail；最後相關77項通過、ruff與diff檢查通過。本輪為作者自審，未聲稱最後修改已獨立覆審。紀錄在`.superpowers/sdd/trajectory-integration-20261026/`。
- paired-runtime-source-sha.json、paired-audit.json逐筆核對正式保存來源；67個受保護原影片／答案／結果等雜湊不變。/two-bey/已發布paired-preview，正式ID數為[2,2,0]，未知段置於可展開的原片證據區。
- 使用者尚未回答該短段是否賽後收拾；程式保留不確定性。兩實體與連續顯示需求已完成，後續全片真值與獨立驗收不冒充已完成。
