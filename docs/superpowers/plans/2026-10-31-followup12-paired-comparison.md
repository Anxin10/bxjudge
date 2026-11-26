# 第二批12張：資料核查與固定seed配對比較草案

日期：2026-10-31。狀態：A513／B525各100epochs與3300steps完整完成，三方評估及獨立CPU重算結束；兩組均未過原採用gate，不晉級、不更換正式模型。結果見`runs/paired-hard-cases-20261031/eval/REPORT.md`，以下保留事前設計。

資料階段已完成：project2共12/12提交、30框，task36／43右側launcher縮框已核對；其他10筆及project1的32筆結果未變。新候選`data/dataset-hard-cases-v3`為525張train／911框，原136val共137檔byte-equal。證據：`.superpowers/sdd/followup12-corrected-20261031/REPORT.md`。獨立審查亦核實資料與來源切分，固定seed訓練已開始。

## 先完成資料

1. 使用者在 http://127.0.0.1:8081/projects/2/data 修正全部可見物件並逐張Submit。預標不等於人標，空預標不等於無物件。
2. 取得12份唯一、非取消的提交後，核對task來源SHA／影格／原圖SHA與匯入manifest相同，逐類檢查框範圍與正面積。空框須個別核對，不把Submit自動視為品質合格。
3. 保存獨立的第二批人標匯出；比對project1全部32筆與v2，任何變動先報告，不偷偷混入新候選。
4. 以原481張baseline，加第一批32張v2人標及第二批12張組成新資料集，預期525張train。不可用已含32張的513張dataset再加入44張，避免重複。
5. 原baseline、v1/v2匯出與資料集都保留。固定136張val及其137個檔案逐位元相同，保留原train標註，來源影片SHA不跨train/val。新標註框數未定，不先填數字。
6. 第二批目前4來源均10/1既有train且供過本專案開發，不能宣稱held-out或已補足10/14。

## 之後的兩次配對訓練

目的：比較這12張增量的開發價值，降低先前seed未固定的比較干擾。只有人標核查完成、實作參數經審查且獲得訓練授權後才執行；本草案不是啟動腳本。

| 項目 | A：固定seed基底 | B：固定seed增量 |
|---|---|---|
| train影像 | v2的513張 | 新資料集預期525張 |
| val | 原136張 | 同一份136張 |
| 初始化 | 同一份COCO pretrained RF-DETR Nano | 相同檔案與SHA |
| seed（擬定） | 42 | 42 |
| epochs | 100 | 100 |
| 解析度／batch | 1280／2，累積8 | 相同 |
| precision／workers | BF16／2 | 相同 |
| 增強與EMA | 沿已解析production policy | 相同 |

所有實驗覆寫參數寫入獨立YAML並保存解析結果，不改production config。啟動前核實現版框架的seed實際傳入與初始化、loader／augmentation隨機狀態記錄；已核實現有training-policy YAML可傳fit.seed42，但框架on_fit_start晚於模組建立，故本次train_one.py在呼叫原train()前另以seed_everything42固定初始化，並記錄PYTHONHASHSEED42、RNG與worker種子路徑。兩次用同一套程式、套件、預訓練檔與裝置依序執行，不並跑、不續接舊v2 checkpoint、不裝新依賴或下載模型。

同seed不保證逐位元可重現，也不會讓兩個不同長度資料集具有相同增強順序。此次CPU實際框架對齊測試確認513與525均pad到528，因此每epoch預期33 optimizer steps、100epoch3300steps；padding重複分布不同。兩邊保存實際完整train/val CSV與步數／wall time，曝光數只另列對齊政策推算，將結果解釋為「這個資料與訓練政策的差異」，不宣稱純12張的因果效應。這一對不估計seed方差，不因結果不理想而自動換seed或加跑挑最好。

過去seed=None的v2只保留為歷史參考，不能充當本次固定seed A。

## 評估與晉級

先保存三者的同PyTorch預測：目前production checkpoint、A、B。B對A作配對開發比較；是否晉級還須B對production通過既有門檻，不以較弱的A取代安全基準。

門檻保持：bey／launcher的FN至少一類下降且另一類不增加，兩類FP均不增加，mAP50:95不下降；同分不晉級，part不參與採用計數門檻。統一BGR1280、conf0.3／IoU0.5，mAP用conf0.001／maxDets100。列每類TP/FP/FN、整體及逐影片結果，不刪最差影片、不事後改val或閾值。

val同時用於best-checkpoint選擇及本次開發比較，不能當獨立泛化驗收。即使B過門檻，也只取得進兩片小批的資格；候選TRT需另驗一致性與速度，不改production選模。核心勝方仍須對照真值，核對落地／空中接觸先後、身分與幾何；沒有實片驗證勝方，專案目標仍未達成。

## 回報與停止點

- 資料階段：12筆提交、框數、來源與舊資料不變的核查，具體疑點交人工處理。
- 訓練前：獨立唯讀審查本草案與實際解析YAML，逐條確認後才決定是否啟動。已取得唯讀審查通過，無未解Critical／Important；精確manifest與逐條處置見`.superpowers/sdd/paired-training-20261031/REVIEW.md`。
- 每次訓練：PID/start time、真實exit code、checkpoint SHA、實際參數、步數與時間；失敗就報告，不自動改政策重跑。
- 比較階段：保存原始預測、獨立CPU重算、門檻結果及限制，不把mAP上升等同核心裁判成功。
- 本次無產品碼變更、無commit；訓練政策33＋進度驗證6＝39pass，未重跑產品全套。已開始配對訓練，沒有訓練完成／新模型晉級結論。
