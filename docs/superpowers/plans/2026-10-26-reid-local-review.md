# 重新辨識第一層：局部人工核對 Implementation Plan

> Inline execution using executing-plans；實作者本身完成，最後一次獨立唯讀審查。沿用現有分支。

**Goal:** 五個保存疑點可播放前後短片，獨立記錄同一顆／不同顆／看不清與X/B、可見性、碰撞、同回合狀態，直接不可覆寫儲存到本機。
**Architecture:** 新reid_review domain驗證新manifest與payload；review_store處理atomic內容雜湊快照。workbench_server提供兩個現行核對功能各自獨立端點；既有identity-launch validator移入維護中的產品module，不改schema/舊提交與草稿。builder從保存diagnostics與原始native觀測抽有限短片和端點原圖，不啟偵測器。頁面localStorage以manifest雜湊隔離，手動儲存後顯示server檔名與SHA，不把草稿當保存成功。
**Tech Stack:** Python stdlib http.server/JSON/hashlib、既有ffmpeg、靜態HTML/JS，無新依賴或模型。
**Spec:** ../specs/2026-10-26-occlusion-collision-reidentification.md，僅第9節第一層可用資料核對，不宣稱scope全程驗證/參考庫/匹配已完成。

## Global Constraints

- 原始觀測/人工/判決不可改，runtime裁判及YAML不改。新提案applied=false、scope未確認，不產生eligible或同色跨場規則。
- 同一顆/不同顆/看不清與side分開；未審與不確定分開，無預填人工作證。跨回合可標不同/未知，不從候選自動猜回合。
- 新服務仍localhost8504，兩既有/新增核對功能皆維護，無舊格式fallback/翻譯。
- 舊identity頁、localStorage與submissions原樣；移入產品的validator保留現行嚴格schema，新測試證明舊儲存有效。
- 保存時驗manifestSHA、caseID、欄位型別/選項，空白拒絕，revised另存；host/origin與資料長度受限。

## Review Focus

跨瀏覽器舊草稿/新版manifest混用；samecase被編輯source evidence；空白預設當GT；儲存失敗卻清草稿；停止/重啟既有服務傷到舊頁或覆核；影片來源影格/原座標疊框對不上。

## Tasks

- [x] RED：schema/manifest/case tamper、三態unknown與未審、舊格式identity API、atomic nooverwrite、Origin與wrongmanifest拒絕。
- [x] 實作domain/store/server，synthetic temp資料HTTP測試。
- [x] builder抽五段短片及端點原圖，SHA/fps/frame provenance，scope保持未確認；UI手動保存、server回執、manifest隔離草稿、匯出JSON，不寫舊result。
- [x] 原片/2原提交/人工修訂/舊manifest保護，測試新服務及原POST功能再替換8504。
- [x] 自動測試、JS語法/純狀態測試、唯讀審查與核實、改錯再驗證，工作台入口與五組核對URL。

## 驗證與審查結果

- 全套 Python：1336 passed、4 既有 xfailed、1 warning；最終後端與媒體測試重跑23 passed。
- Node實際產品module＋記憶體DOM與純狀態：7 passed；另通過JS語法、新增Python程式ruff、git diff檢查。
- 審查發現親自以RED測例確認並修復：無效原草稿被覆寫、註記仍顯示已保存、持久化雙失敗卻聲稱草稿保留，以及保存中的change事件允許平行提交。最後唯讀覆審無尚存阻斷或Important。
- 原identity validator的AST一致；兩份原人工提交均仍可驗證。64個受保護檔案雜湊不變，含原片、原提交、人工修訂、舊manifest、結果／軌跡及非本次的.gitignore。原三回合已覆核狀態保留。
- 五組來源／原生框／生成器／UI／媒體SHA一致；ffprobe確認短片影格數與原解析度，端點框按1920×1080／1280×720換算。未驗證實際瀏覽器視覺操作；工具政策拒絕後未繞過。
- 最終核對集：`runs/reid-review-20261026/site-ready`，manifest SHA `a7ea3040a3a292fad7f0d0cdbab8afe1dd746fa0bc16db54ffabd2e7242db887`。前面未部署的建置保留作開發紀錄。
- 主入口與新頁／JS／CSS／manifest、舊manifest、已存版本API均HTTP200；正式新標記清單為空，合成標記只在測試臨時資料夾。8504舊程序已停止，沒有占用者；由新package服務提供兩個現行核對端點。
- 詳細測試、保護雜湊與部署檢查：`.superpowers/sdd/reid-audit-20261026/`。核心實測版本保持`6e68fcc`；新工具不代表重新量測或提升模型準確率。

## 完成界線

只交付新資料核對工具，五組尚未人工填答，不冒充已有新GT；不執行外觀匹配或重跑GPU。shadow候選、自動scope/refs凍結仍後續實作。頁面視覺驗證若工具政策不可用如實記錄，不繞過。
