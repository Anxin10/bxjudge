# 重疊／模糊／外觀相近候選核對

目標：從保存原生觀測與原片挑最多6個候選，以既有三態核對工具供必要的新人工核對。
不啟模型、不改接續或裁判；frame0串流BGR量測只供挑片。最多2組／類，已核對影片區間排除、同片相鄰候選抑制，無符合者不湊數。候選不當GT。

Files: config/reid_challenge.yaml；scripts/select_reid_challenges.py；既有build_reid_review.py（重用）；web/reid_review.js（同功能相對端點）；workbench_server.py（兩套現行reid資料集獨立路由與資料夾）；build_workbench.py（來源驗證與入口）；tests。

- [x] RED/GREEN：挑片有界、排除已看區間、類別缺額可少於6、endpoint為實際觀測、sourceSHA一致、無預填人工值。
- [x] 重用媒體builder產生候選核對包，區分三種挑片線索；原片及舊答案雜湊保護。
- [x] 現行服務明確配置兩個reid-set，各別manifest／parent／存檔；原路由保留，新版頁面用相對URL，錯套資料集拒絕。
- [x] 合成HTTP與實際產品JS測試；唯讀自審／來源檢查；安裝統一入口與新頁。

Review: 外觀相近不代表同色／不同顆；低Laplacian不是模糊GT；IoU不是碰撞GT；同nativeID不代表身分未換。候選scope未驗證、不使用已確認側別去猜鄰格，不能因挑片而刪原分母。來源條件固定config，無匹配閾值或新準確率。

## 驗證與完成界線

- BASE 2bde1b7，沿用feat/local-judge；沒有另開分支/worktree或push。原五組的manifest、UI副本、草稿鍵與兩份人工提交不改；新UI資產只用於新核對包。
- 原型挑片函式不存在、服務尚未接受明確多資料集、新頁相對路由尚未接入時均有RED；修正後Python相關59 passed，實際產品JS＋純狀態8 passed。新／修改Python檔ruff與git diff --check通過；沒有聲稱重跑全部GPU／訓練測試。
- 流程測試使用ffmpeg合成原片；包含缺額不湊數、已看區間抑制、原生endpoint、來源SHA錯誤、媒體被改、保存資料夾隔離、跨manifest／parent拒絕、存檔中不可平行提交、無效草稿保護。合成標記只寫臨時目錄。
- 自審確認分類只作挑片：候選未綁定勝方、X/B或relation；IoU高也可能重複偵測，低清晰度不當GT。沒有本輪獨立審查員，不將作者自審稱為獨立覆審。
- 740格、1215個框量測；選出2 overlap、2 similar、2 blur。沒有同色或碰撞GT聲稱；仍是兩支重複使用的開發片。六組選擇與舊核對區間／彼此clip不重疊，中心與前後來源格均在實際短片內。
- 媒體包 runs/reid-challenge-20261026/site；manifest SHA 620ab8c3cc18d6abb081b4892a012fe125ac5155377b79f165e42f064d5f3ab6。原片1920×1080、native框1280×720換算、六部短片來源影格數、PNG/MP4/UI與選擇／來源SHA均驗證。每組scope仍unverified。
- 66個受保護檔案SHA不變。新六組提交數0，原五組API仍有2份保存版本（含25d81dd4發射頭補充）；原3回合覆核及人工修訂未重算。
- 8504服務以明確 --reid-set reid-review 與 reid-challenge 重啟；資料夾不可共用，沒有舊CLI參數fallback。原identity-audit/save與reid-review/save保留現行格式，新challenge獨立端點。檢查URI與資料版本不需瀏覽器UI，未繞過先前瀏覽器政策限制；未作視覺驗證。
- 上線/reid-challenge/已加入工作台。可只核對部分案例並保存；同物件／scope不清楚可選未知，其餘欄位可留空。這一層未改核心或啟用自動接續，實測核心版本仍6e68fcc。
- 詳細量測／SHA／部署檢查在 .superpowers/sdd/reid-challenge-20261026/。
