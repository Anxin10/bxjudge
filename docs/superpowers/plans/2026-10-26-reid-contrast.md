# 已確認不同顆外觀對照

目標：沿用既有人工修訂，只擷取同一來源格中、人工確認X/B且都separated的陀螺框，建立可追溯負對照；不要求重做原核對。

介面：scripts/analyze_reid_contrast.py讀reference pointer／原identity manifest／現有appearance evidence／原YAML，驗來源與修訂SHA；原片frame0起CPU串流只解碼一次，取已確認框，使用既有describe_crop與distances。新輸出目錄原子發布report/index/crop與人工依據副本。

邊界：不同顆GT僅屬同一影格X/B，不跨格外推身分。掛載、未知、重複側別、壞框排除並計數；原始全人口保留。鄰格按影片與window_s區間分組，報相關性，不當獨立比賽數。正例仍是4個局部開發案例，無自動scope／可信模板／新準確率／接續閾值。原裁判不改、所有applied=false。

- [x] RED：新工具不存在時測試收集失敗；只用人工確認且都分離的唯一X/B，排除掛載／未知／壞偵測。
- [x] GREEN：合成影片原始框縮放與凍結證據、修改原始來源拒絕、輸出不覆寫、未知／品質失敗保留分母、解碼關閉。
- [x] 原片對照：凍結五組修訂與逐框reference、核對所有SHA，展示負對照與正例距離範圍，明列場景不等價與資料相關性。
- [x] 整合統一工作台、獨立子頁來源驗證、測試與自審、原檔保護驗證。

## 完成證據

- 沿用feat/local-judge，BASE db288bb，未開分支／worktree，未跑新模型或改裁判核心。
- 12項新合成測試；與既有appearance/review/media/server共50 passed。新與修改Python檔ruff通過，git diff --check通過。本次未重跑全套GPU／訓練測試，不冒稱新全套結果。
- 自審確認局部X/B不外推；負例只同格、兩方都separated；0負例／品質失敗仍保留人口，無閾值與accuracy。新增發布驗證helper的三種竄改測例先RED後GREEN，另測缺失frozen hash拒絕。沒有本輪獨立審查員；不將作者自審稱作獨立覆審。
- 38個原人工影格：6符合負對照、10不具唯一雙側、22含未分離；6對共12張原始crop、2支影片、3個時間相關區間。這些區間不是已驗證回合，也不是獨立盲測樣本。
- 色彩：4同顆案例中位數範圍0.174459–0.397510；6不同顆同格距離0.671256–0.825675，範圍未重疊。梯度：0.066696–0.181803對0.141977–0.227524，範圍重疊。取樣條件不同，不設定通用門檻。
- 原人工reference修訂53eceea945597025、五組核對補充25d81dd4均保留；66個受保護檔案SHA未變，含原影片、判決／軌跡／人工提交。f1618使用最新深色X／淺色B，未修改原始逐框答案。
- 最終artifact：runs/reid-contrast-20261026/diagnostics-final；report SHA c98ac4217ed4b949f9967c2ac9487c0013e0a88f5d13496a27f1a2687e4c5ae4。原產物保留；來源／code／policy／frozen ref／12crop檔與pixels SHA驗證。
- 工作台發布驗正例報告版本與所有來源、凍結副本及crop；頁面HTTP、report及12crop與磁碟一致。8504前程序已停止，以相同產品服務重啟；舊儲存資料夾不改。未做瀏覽器視覺驗證，未繞過先前政策限制。
- 入口 /reid-contrast/ 已加入統一工作台；原reid-review/appearance頁與core實測6e68fcc保留。診斷全部applied=false，沒有開啟自動接續或聲稱新裁判速度／準確率通過。

審查重點：未標記框不能因nativeID變可靠；裁切未畫線、FPS／尺寸一致；f1618依最新修訂；多個相鄰格不可灌成獨立驗收；不同顆也可能相近不得自選決策值。新實作維持既有分支，不開worktree。
