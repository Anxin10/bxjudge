# 每回合兩顆陀螺的穩定身分與連續軌跡

使用者新目標取代先前純外觀門檻工作：一場只有兩顆，軌跡需整合連續。已確認兩顆指陀螺，發射器另追蹤。沿用現有分支，不再開分支或擴充核對工具。

設計：每回合固定兩個實體槽位，cv2.KalmanFilter預測位置、SciPy Hungarian一對一配對；位置、有界更新的外觀與nativeID只作關聯線索，不當可信X/B參考庫。高IoU重複框不建立第三顆；所有原框及未配對框另存，不刪原始證據。僅從實際偵測寫observed點，實測點間的缺格用明確interpolated虛線呈現，無已觀測終點不外推補成真值。

先用保存native偵測離線重播兩片3回合並產生可播放預覽，驗證固定上限／不跨回合／一對一／來源完整／人工X/B與同顆案例。原裁判及原人工答案保持不變；這是新的兩實體重建，不藉新軌跡宣稱既有判決已正確。

- [x] RED/GREEN：同顆換nativeID不新增實體、漏偵後接回、重複框不生成第二個假物件、兩顆交叉不交換、超過兩框保留unassigned、回合獨立reset。
- [x] 缺格連續輸出：每一格最多兩實體，觀測／插值區分，沒有量測不能成裁判證據；不修改傳入原偵測。
- [x] 實片重播與正式CLI接線：保留兩類視覺層（陀螺、發射器），原bey_part留作burst原證據、不併入完整陀螺身分；不丟未知／誤偵分母。
- [x] 自動測試、獨立審查發現親驗修正、正式原片重跑、統一工作台預覽。
- [ ] 使用者確認0.37秒單顆分段是否賽後收拾；全片身分與獨立驗收仍未具備真值，不以局部測試宣稱全目標完成。

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
- /two-bey/已裝上統一工作台。核心已接線，但「全片每框皆正確」及單顆分段仍非已證實；goal保持未完成直到剩餘需求有足夠證據。
