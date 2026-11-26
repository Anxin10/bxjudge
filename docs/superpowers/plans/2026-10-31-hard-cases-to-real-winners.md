# BXJUDGE 難例重訓與實片勝方驗收工作清單

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 本工作清單供 Claude 審查；沿現有分支由 Codex 執行，遵守使用者既定的獨立唯讀審查流程，不新增分支／工作樹。

**Goal:** 使用已完成的32張人工難例改善偵測證據，在20回合人工真值切片先取得可追溯、可比較的核心勝方，再決定是否擴回40支影片。

**Architecture:** 沿用現有RF-DETR Nano、TensorRT、動態盤面幾何、雙陀螺身分及規則引擎。先固定人工匯出與驗證集，重訓單一候選、離線比較，再接小批核心；物件偵測、物理接觸、玩家身分與裁判結果分別驗證，不以其中一項代替另一項。

**Tech Stack:** WSL Ubuntu原生ext4、uv／Python、既有Label Studio、RF-DETR、PyTorch CUDA、TensorRT、pytest；不因本清單新增依賴或模型家族。

**Spec:** `.superpowers/sdd/codex-plan-2026-10-27.md`、`.superpowers/sdd/codex-next-2026-10-27.md`、`docs/spec.md`。下述現況更新舊文件的過時數字，不變更原本實片勝方目標。

## Claude 審查狀態

- **使用者指示：先不管Claude登入。** 不再重試OAuth。下方有效Claude工作階段的審查已交付，Codex回應另記，不覆蓋該審查原文。
- [x] Claude 已讀本清單並留下逐項審查結果（2026-10-31，Claude Code 桌面版有效登入的工作階段，唯讀，未啟 GPU）：見 `.superpowers/sdd/claude-review-worklist-2026-10-31-verdict.md`。重點：
  - **Important：task19 疑似漏標。** 右上袋區約 x1220–1290、y300–330 有疑似陀螺，指尖下的深色物體也待確認。使用者已決定補框，由使用者在 LS 操作，之後另存 v2 匯出與 v2 dataset，重訓改用 v2。
  - **Critical：32 張難例全部來自 20 回合真值切片的 5 支影片**（202324／212413／213053／214534／222152），其中包括小批用的 212413。Task 4 須分組報告，小批以 201734 為主。
  - **Critical：D265／G555 空中卻被當成落地起點**，會讓空中碰撞被當成盤面碰撞，產生假勝方。Task 4 每個勝方都要人工核對 physical_floor 時點；無法核對就 needs_review。
  - **Important：** RF-DETR seed=null，單次訓練的差異含跑間雜訊；`compare` 無法比兩個 RF-DETR；門檻比較須 PyTorch 對 PyTorch，且要事前登記。
- 歷史紀錄：Codex呼叫CLI曾回401、未產生審查；不與上方後來有效Claude工作階段的verdict混同。不再處理登入。
- [ ] 核實並處理每項 Critical／Important，再執行重訓與後續新程式修改。
- 審查回報位置：`.superpowers/sdd/claude-review-worklist-2026-10-31-verdict.md`。
- 建議每條寫：成立／不成立／不確定、Critical／Important／Minor、證據路徑與理由、對應任務。寫入檔案是交付管道，不代表 Claude 已讀。

## Global Constraints

- 分支 `feat/local-judge`，本次查核HEAD **d426130**。不新建分支／工作樹、不push、不merge、不清理既有舊分支。
- 不修改原始影片、原本617張人標、使用者人工答案、`calib/`手工範本；新輸出另存，不覆寫歷史試驗。
- 不保留相容fallback；缺證據維持unknown／needs_review。不能為產生勝方放寬幾何、whole-body入袋、返回盤面、身分或否決規則。
- 所有新閾值放YAML；先查既有依賴和API，不增加未驗證的抽象層。
- 程式變更：重現RED→最小修正GREEN→適當測試→獨立唯讀審查→親自核實每條→修正／再測／複審→只提交自有檔案。資料操作以來源／輸出雜湊和原始紀錄核對，不把合成測試當實片驗收。
- 原6個tracked共享／外來修改以及未提交HUD檔案不動、不stage；不要從整個worktree直接git add。
- 不同時開重負載模型／測試批次；啟動前確認PID、GPU及WSL可用記憶體。觀察timeout不能直接當程序終止或重啟理由。

## 已完成與現況（2026-10-31實查）

- [x] 本機標註專案1及32張任務建立；入口 `http://127.0.0.1:8081/projects/1/data`。
- [x] 32／32任務各有一筆未取消的人標提交，無重複來源影格；原任務data／人標結果前後一致。
- [x] 人工匯出 `data/detector-hard-cases-20261030/human-priority32.json`，SHA256 `65c6821dc1c730a6a17421cf53c9297ecfe69c55b16e80af168d33cb2e41769e`。
- [x] v1人工框73個：bey48、launcher25、bey_part0；task19為已Submit的空答案，**疑似漏標、不得稱乾淨負樣本**。保留歷史原答，等待使用者在LS補框；重訓不得使用尚未修正的v1。
- [x] 逐來源比對預標與Submit：10／32張有框／類別變更，其餘22張已Submit且框相同。task19原預標與人標均為0框、未取消；不等於完全遮擋物件不存在，也不擅自改使用者答案。
- [x] 透過正式 `bxjudge label extend-training` 建立 `data/dataset-hard-cases-v1`：train481→513圖、807→880框。
- [x] 使用者修正task19並Submit，已另存human-priority32-v2.json，SHA `c3e711c3b7c4b59f59d4414d3e045d82b23d07bf5994bad38b34ac106afaae22`。只有task19新增1個bey框，其他31份result／取消狀態不變。dataset-hard-cases-v2為513train／881框、136val，v1保留不覆寫。
- [x] 原dataset **620個檔案**前後SHA一致；val的JSON＋136張圖 **137個檔案**與擴訓副本逐byte一致，舊train labels保留，無重新切分。
- [x] 最新程式修正d426130：VLM建議核對判定／來源／動態幾何／veto後回寫、selector入口防止輸出污染calib。全套1658pass／4既有strict-xfail，獨立複審無阻斷。
- [x] 已有8組人工校準範本；不要照舊交接再要求補原本列出的4組。
- [x] 人工真值20回合已完成reviewer_fix：15有勝方、5無結果。保存的核心勝方預測仍0；None/None配對的25%不是5個成功勝方。
- [x] 使用者明確指示開始重訓後，已啟動**一次**RF-DETR候選100epochs，工具session12415；詳細PID／start_ticks／manifest在 `.superpowers/sdd/training-hard-cases-v2-20261031/`。仍未更換production weights或啟動新影片批次。
- [x] 重訓100epochs完整結束，exit0、4885.76秒，最佳checkpoint SHA b16df07f...可載入、training.json存在，changed_inputs=[]；未改production選模。

數據證據：`.superpowers/sdd/priority32-project-20261031/human-export-verification.json`、`dataset-verification.json`；實片現況：`.superpowers/sdd/acceptance-audit-20261031/STATUS.md`。

## Review Focus

1. **task19空答案：** 是疑似漏標待修，模型預標也是0框不構成物件不存在的證明。使用者已決定在LS補框，之後保留v1、建立v2；不能代填。增量沒有bey_part，不宣稱零件能力因此改善。
2. **評估重疊：** 32張雖未洩漏到固定val，卻全部來自真值切片的5支影片。實片必須分四組報告；訓練例／val／未入偵測資料的影片不混稱held-out。固定136張val，禁止重切分。
3. **模型選擇偏差：** 單一候選沿production訓練設定；不要反覆調門檻／種子挑最佳，也不要讓`compare`預設覆寫`weights/selected.json`。未通過候選保留為失敗試驗。
4. **接觸與投影不同：** C225／D265／G555人標空中，E303／F417接觸盤面；同片5格是開發例，不是泛化評估。D／G會被現行settle選為候選起點，缺陷仍在。不得用框中心／底點／IoU或慢速位移直接當實體接觸。
5. **身分與時間順序：** 長gap／碰撞merge後unknown不能硬接X/B；後續離地不抹除先前已接觸歷史；較早非得分離場依使用者規則全程適用重賽。未知事件先後不能硬選勝方。

## Task 1：確認本輪資料與對照固定

**Files／輸入輸出：** v1 human JSON／dataset只作歷史對照；修正後輸入`data/detector-hard-cases-20261030/human-priority32-v2.json`，新輸出`data/dataset-hard-cases-v2`。原`data/dataset`仍為固定基底。

- [x] 使用已安裝Label Studio export API匯出，逐task比對data、annotation.result及取消狀態，保留來源SHA。
- [x] 執行正式資料擴充命令，結果為513train／136val，新增73框。
- [x] Claude要求task19補標；使用者已補並Submit，Codex核實新增1個bey框，不需重做其他31張。
- [x] 正式export API另存v2；data完全相同、只有task19的人標result改變，其他31份result／取消狀態相同。DB答案在匯出前後不變。證據 `human-export-v2-verification.json`。
- [x] 原data/dataset＋v2建立dataset-hard-cases-v2，513train／881框、136val；原620檔不變、val137檔一致、舊train標註前綴相同，train/val影片SHA無交集。正式命令已實際exit0，勿對同輸出重跑覆寫。
- [ ] 核對現有選模記錄、原權重、engine、production設定及原train metadata的SHA；訓練開始時寫新manifest。若套件、初始化、種子或訓練政策與原模型不同，明列混雜因素，不宣稱只由32張人標造成差異。

已執行命令（不可再次覆寫同輸出）：

```bash
cd ~/bxjudge
uv run bxjudge label extend-training data/dataset data/detector-hard-cases-20261030/human-priority32.json --out data/dataset-hard-cases-v1
```

**完成條件：** task19依使用者實際Submit修正，v2可追溯且其他31份人標不變、baseline未變、val137檔相同。模型預標與人標均空不能代替物件存在性核對。

## Task 2：單次 RF-DETR Nano 重訓候選

**Files：** 讀取`config/detection.yaml`、`config/thresholds.yaml`、v2 dataset；新輸出`runs/train-hard-cases-20261031-v2/`（若已存在非空，停止核查，不覆寫／自動換多輪試驗）。

- [x] 啟動前GPU compute list空、無其他pytest／train，WSL available約14.5GB；預訓練權重已快取，SHA `d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d`。候選manifest記實際版本與來源，無新模型下載。
- [ ] 確認目前 `train_detector` 的RFDETRNano預設初始化，**不是從目前production模型無縫續訓**；以完整513張train重新訓練。若要更改初始化為fine-tune，須先讓Claude審查新的比較設計，不偷偷換。
- [ ] 沿用production設定：100 epochs、1280輸入、batch2／grad_accum8、2workers、BF16等實際解析參數；保存training.json、framework logs、checkpoint及CUDA peak紀錄。
- [ ] 事前登記RF-DETR新舊seed均None／null；本輪維持單次候選，不宣稱差異可完全歸因32張增量。若要固定seed作因果對照，須先另定同seed的481張基底與513張增量雙跑設計；目前不自行增加這兩次訓練。pretrain檔SHA與套件版本由執行時重新核對，不能用存在日期推定舊訓練SHA。
- [x] 已啟動以下唯一候選（PID1210962為uv子程序，實際Python PID1210965；再觀察須核process.json的PID/start_ticks及同一工具session）：

```bash
cd ~/bxjudge
uv run bxjudge train --kind rfdetr --dataset data/dataset-hard-cases-v2 --out runs/train-hard-cases-20261031-v2
```

- [x] 真實exit0、100epochs完成、最佳checkpoint存在並在獨立比較程序實際載入；來源／val／舊模型與選模SHA不變。

**完成條件：** 單次production候選完整結束、有checkpoint與完整來源紀錄，原production選模和權重完全未改。

## Task 3：固定驗證集比較，決定是否進小批

**已執行結果：** 同PyTorch舊／新、136val、conf0.3/IoU0.5、mAP conf0.001/maxDets100。mAP50:95 0.4926046→0.5180534，但bey FN13→17、launcher FP19→29，未過事前門檻，**本候選不晉級**。沒有為測到PASS調門檻、沒有改正式權重或建新engine。獨立CPU從predictions重算一致；完整 `runs/hard-cases-20261031-v2-eval/REPORT.md`、兩種混淆矩陣PNG/PDF/CSV及原始predictions已保留。Task4本輪不自動啟動。

### 使用者同意的後續：CPU診斷 → 小批補標（不重訓）

- [x] 由保存predictions按COCO逐類重算：bey少5／多1；launcher FP19→29拆為16位置近似延續、13new-only、3removed。new-only是未與舊FP配對，不是13個新物件。
- [x] 13個new-only FP分為6個重複框、5個同類IoU不足、2個無同類GT重疊；逐影片表與32個case裁切存runs/error-audit-20261031-v2，CPU獨立核查一致。
- [x] 圖像範圍／類別疑點分開保存：可能手包含在launcher GT、可見本體缺對應GT、或launcher-like物體對上bey類別。未經人工確認不改val、不刪圖、不調IoU，也不把全部FP當背景負例。
- [x] 共同FN12個，其中a0f62263實為6個而非轉述8；同COCO算法及獨立核對相同，保留差異不改原Claude verdict。
- [x] 檢查剩餘77個train候選及人標時間距離後，選7張原候選，CPU新抽5格，完成12張／4來源的初包。唯一、train-only、無已標同影格、距離既有人標至少35來源格；這不證明視覺獨立。所有4來源仍是開發切片中的10/1影片，沒有涵蓋10/14相機，不假稱held-out。
- [x] 匯入獨立project2：http://127.0.0.1:8081/projects/2/data ，7份封存預標＋5份無預標；12圖應用路由200且bytes一致。原project1的32份人標／task data不變，不Sync。
- [x] 第二批12/12張已完成，task36／43僅右側launcher縮框；其他10筆及第一批32筆未變。匯出第二批v2＋44筆合併v4，建立`data/dataset-hard-cases-v3`：525train／911框；136val共137檔byte-equal、42／11影片SHA互斥。原1934保護檔未變。尚未訓練；固定seed方案尚待獨立審查。證據：`.superpowers/sdd/followup12-corrected-20261031/REPORT.md`。
- [x] 2026-10-31讀庫核查：第二批0/12提交、0草稿；第一批32筆task data／人標與v2相同。已寫[固定seed配對比較草案](2026-10-31-followup12-paired-comparison.md)，尚未審查／啟動，不把文件完成當訓練完成。
- [x] 範圍修正與資料核查完成；固定seed42的513／525兩組方案與runner經獨立唯讀審查、39項測試通過。已啟動A→B各100epochs，任一失敗停止，未換正式模型。證據：`.superpowers/sdd/paired-training-20261031/STATUS.md`。10/14 train側難例尚未擴增；維持原採用gate，不以修改val或threshold讓v2回頭通過。
- [x] A/B各實際100epochs／3300steps完成（各約83分）；同136val三方評估及獨立CPU重算完成。正式／A／B的bey FN13／14／16，FP13／20／24；launcher FN17／14／10，FP19／18／19；mAP50:95 49.26%／51.07%／49.95%。兩組都不晉級，正式模型保留，不建新TRT／不擴批／不追加訓練。完整`runs/paired-hard-cases-20261031/eval/REPORT.md`；新增6項gate測試pass，review無阻斷。尚無新增核心勝方。

### 2026-11-01：重新分組8:1:1

- [x] 使用者授權重分現有661圖／1096框，按整支來源影片SHA分組，新建`data/dataset-811-20261101`：Train528／855框／43片，Val65／109框／5片，Test68／132框／5片，影像比79.88%／9.83%／10.29%；三組各具全部三類，舊資料664檔未變。
- [x] 凍結recipe／assignment／Test SHA；661圖bytes及1096框字段保留，重編annotation IDs並保存舊split+舊ID映射。7項CPU tests pass、獨立核查無阻斷；原664＋新669檔hash通過。現有train/val載入／prepared593圖可用且排除test68。證據`.superpowers/sdd/resplit-811-20261101/REPORT.md`。
- [x] 使用者授權重新訓練，已啟動新811基準：Train528／Val65，RF-DETR Nano從同COCO pretrain重新建立，seed42／100epochs，採原crop增強，3項輸入測試與精確manifest唯讀審查通過。紀錄`.superpowers/sdd/train-811-20261101/STATUS.md`。沒有resume舊模型、沒有Test推論／更換正式選模。
- [x] 新811完整100train/val epochs／3300steps完成（約79分），checkpoint61dbab...20669／770凍結檔／Test隔離均獨立核實。Val65標準化mAP50:95 50.49%，bey P85.53%／R94.20%。固定方案與Val-selected checkpoint後另做一次Test68：mAP32.10%，bey87/20/19 P81.31%／R82.08%、launcher10/1/11 R47.62%，part0/15/5（小分母）。Test數字與775檔獨立重算通過，無程序阻斷但品質不足以支持替換，正式模型未變。證據`runs/train-811-20261101/test/REPORT.md`。未換seed／閾值／切分，原訓練Test false歷史保留。
- [ ] 下一步定位Test誤差集中來源214330（bey FN17/19）與211127（launcher FN8/11），不先假定原因、刪難片或追加重訓。若依這些案例調整模型，應視為開發回饋並另留新來源驗收；現有Test本來也是歷史開發資料重分。
- [ ] 若要再追加標註，先讓增量流程支援凍結Test；現有`label extend-training`是兩切分設計，不直接對本新三切分資料使用、不加fallback。這是後續能力，不阻擋本次建集／新train載入。

### 文獻後的單設定增強對照

- [x] 已定位5篇primary方法／結果，建立`.research/literature_matrix.md`與來源核查；保留DETR crop約+1AP的相反證據，不先宣稱關crop較好。
- [x] Native增強稽核86張train／344draw每政策；current bey340曝光裁出35、4次框面積保留<10%，no_crop保留340。合法裁出不是bug，特選樣本不當人口估計。原人標／val未改。
- [x] 實作C獨立policy：相對B只將scale_jitter true→false（整個resize-crop分支消融）。同525／seed42／100epochs，5項測試pass，獨立審查無未解阻斷；已啟動C完整訓練→固定production/B/C評估，無新分支／commit、不替換正式模型。紀錄`.superpowers/sdd/augmentation-research-20261031/STATUS.md`。
- [x] 2026-11-01核對C完整100epochs／3300steps（約79分）與三方評估完成，獨立CPU重算無阻斷。C相對B：bey FN16持平／FP24→23，launcher FN10→18／FP19→17，mAP49.95%→48.16%；不晉級、不追加訓練，正式模型保留。證據`runs/augmentation-no-crop-20261031/eval/REPORT.md`。這次分支消融未改善，不推廣成所有crop均必要；尚無新核心勝方。

補標與研究範圍詳見 `runs/error-audit-20261031-v2/REPORT.md` 與 `data/detector-hard-cases-20261030/followup12/README.md`。發射前漏偵仍會影響X/B初始錨定，不能只因尚未發射便視為與裁判無關。

**Files：** 參考現有 `.superpowers/sdd/detector-hard-cases-20261030/validate_current_detector.py` 的COCO計分與production BGR路徑；候選獨立輸出 `runs/hard-cases-20261031-v2-eval/`。該參考腳本寫死production TRT且只計IoU0.5，必須另建可指定checkpoint、可算mAP的比較入口，不改歷史腳本、不加入新框架。

- [ ] 門檻比較一律**舊PyTorch checkpoint對新PyTorch checkpoint**：同136張val、同輸入／精度／conf／後處理。不得拿舊TRT的89.6%當本輪PyTorch基準。通過後的TRT另做一致性及速度檢查。
- [ ] 明確不使用`bxjudge compare`比較這兩個RF-DETR；它要求rfdetr＋yolo26各一且預設可能寫production selection。新比較結果與候選manifest一律分開保存。
- [ ] 報告mAP@50及@50:95、每類TP/FP/FN/precision/recall，並對照實際conf0.3、IoU0.5的基準。既有本引擎基準bey P/R89.6%、launcher P67.8/R71.4%、part P40/R50；part真值僅4個，不外推。
- [ ] 32張train難例只報開發改善，不計為驗證accuracy；原目標的40片held-out bey recall≥0.9仍需要獨立人標，現有val不能偷換成那個族群。
- [ ] **事前登記的候選小批門檻**：以計數比較，固定val上bey／launcher的FN至少一類下降、另一類不增加，兩類FP均不增加，mAP@50:95不低於舊模型；part不參與此門檻，同分不晉級。單次seed未固定的雜訊與val兼作best-checkpoint選擇集的限制必須隨報告列出。這是開發晉級條件，不是正式獨立驗收；不得測完改指標挑有利結果。另核查兩顆重疊、入袋、發射器中心接合與source像素一致性。
- [ ] 通過後依既有TRT匯出流程建立**候選獨立engine與manifest**，核對新checkpoint／GPU／套件版本與輸入shape；不得把新權重配舊engine。

**完成條件：** 公平可重現的對照報告、未處理誤框與速度風險明列。是否採用由實測決定；不預定「一定換成新模型」。

## Task 4：兩支小批 → 20回合真值切片

**Files：** 參考已驗證`.superpowers/sdd/w5-ownframe-20261030/run_full_two.py`的來源／輸出隔離模式，建立新run manifest與輸出，勿覆寫原full-two／truth-slice／task7 core。候選selection manifest單獨路徑，不改`weights/selected.json`。

- [ ] 先以201734、212413兩支原片同條件跑舊／新候選；啟用現有dynamic-calibration、two-bey-tracking、`--rejudge --no-vlm`，保持規則／人工範本一致，避免加入VLM混淆偵測改善。
- [ ] **資料分組在跑之前凍結**，以來源影片SHA核對，19支分成四組：新增難例train＝202324/212413/213053/214534/222152（5支）；其他舊train＝201712（1支）；baseline val＝210934（1支）；未入偵測資料＝201642/201734/202233/202423/210958/211141/212340/213415/214505/214640/222645/230754（12支）。逐組列每個結果；「未入偵測資料」也不等於從未供本專案開發檢視。
- [ ] 小批以未入偵測資料的201734觀察轉移表現，212413只作seen/development案例，不能用它的改善宣稱泛化或單獨歸因32張人標。
- [ ] 比較偵測漏框、native track斷點、初始與全程身分unknown區間、有效幾何、事件與核心winner。不能把「初始雙側身分7／20」等同「全程身份全對」。
- [ ] 檢查每個新勝方的來源框／身分／幾何／終結與先後；若碰到落地候選問題，不硬編碼五格答案、不延長窗口冒充接觸證據。只在此主線顯示的具體阻礙上做RED→GREEN修正。
- [ ] **採納任何勝方前的必要檢查：** 人工核對雙方physical_floor候選時點是否真的接觸表面，以及最早碰撞相對落地的先後。不能核對或仍有空碰可能時，該結果維持needs_review、不計入已驗證勝方。另存採納核對紀錄，不改寫AI原判或既有人工真值；此人工gate不等於已修通用接觸演算法。
- [ ] 兩支結果可靠才擴至已review的20回合／19片；逐回合列AI原判vs人工結果及所有miss。15個有勝方真值是主要覆蓋分母，另列5個無結果，不能以None/None湊完成數。
- [ ] 報告預測覆蓋率、裁判正確率（含分母）、人工覆核率、accuracy_lower_bound；未完整真值的族群不外推overall_accuracy。保持原audit freeze與人口變更檢查。
- [ ] 分開記 `raw_winner_coverage=原核心有勝方數/20`、`verified_winner_coverage=經來源核對且正確勝方數/15`、`wrong_winner=錯誤勝方絕對數`。5個無結果真值上任何AI勝方亦列錯誤，不能排除；needs_review不把原始錯誤預測從報告刪掉。
- [ ] 兩支→20回合的晉級條件：wrong_winner=0、每個採納勝方都有上述框／身分／幾何／終結／physical_floor與碰撞先後核對。20→40片另需wrong_winner=0且verified_winner_coverage高於舊模型，附四組分層與15分母；零勝方不能晉級。

**完成條件：** 核心確實有實片勝方、能與真值逐項比對，並明列仍unknown或錯誤的回合。原計畫目標是先至少15回合的可靠小切片；不能把得到一個候選就宣告整體達成。

## Task 5：落地／接觸證據的未解項

**Files／已知負例：** `runs/region-occupancy-20261031/cross-video-pending/five-contact-checkpoints.json`、`AIRBORNE-FINDING.md`；現行`src/bxjudge/landing.py`、`launch_check.py`。本清單不假設已建立接觸classifier。

- [x] Claude已核查並指出假勝方風險，Codex確認物理落地候選會進入空碰截止點／impacts。上方Task4的人工檢查為任何勝方採納的前置條件，不能等「出現錯判才檢查」。這不要求先完成通用接觸模型才能做既定偵測器訓練。
- [ ] 通用接觸判定仍須跨片正反例與可驗證證據設計，再制定最小實作，不先引入未證實模型或3D系統。
- [ ] 人標C225/D265/G555空中、E303/F417盤面不得改寫。D265前19格缺的是有效幾何，native陀螺觀測完整；不重做已排除的漏偵歸因。
- [ ] 修正需要新增真實正反例與跨場景驗證，並測試當格空中不抹除落地歷史。5個單格不能推定首接觸時間或完整區間，不算通用接觸模型驗收。
- [ ] 舊區域VLM的104次呼叫只是重複profile試驗，未通過採用條件；不再追加提示詞挑最佳答案、不把Jev/Gemma/Qwen結果直接接核心。

**完成條件：** 通用演算法仍未完成；Task4即使產生核心勝方，也必須先經已明訂的物理狀態人工gate才可計入已驗證成果。未解的空碰／落地先後維持待覆核；不得把本項移出scope來宣告實片裁判成功。

## Task 6：擴回40片與速度驗收（W5成功後）

- [ ] 只有Task4取得可靠小批結果後才擴回40片，完整保存來源／版本／逐片與batch wall。
- [ ] 同報正確率、覆核率與完整處理時間；每片與整批gate皆須核對。core-only、含VLM及first-decode到final-note延遲的口徑分開，不重複加計。
- [ ] 正式門檻沿`config/evaluation.yaml`：未標覆核的finish＋winner正確率≥0.95、touch/air recall各≥0.90、review_ratio≤0.30、speed_ratio≤1.0。sample估計須完整覆核且n≥30、95% Wilson lower bound達標；20回合開發切片不足以代替隨機稽核（0.95門檻在全對情況仍約需73例）。
- [ ] 舊6fce1e0的19片core-only speed0.947 PASS不當成本版／新模型速度。新增SHA讀取、接觸判定或新backend後須重新量測。
- [ ] 先以既有時間policy執行，正式限定族群與超長片分開報告；若要改標準先讓Claude審查，不能用平均數掩蓋慢片。

**完成條件：** 對同一版本、模型、族群提供有實際數字的accuracy／review／speed報告；未過就保留未完成。

## 暫不做與保存事項

- 不新增ReID研究、區域VLM提示詞輪次、播放器／HUD功能或額外模型家族，除非主線的具體實測證明需要。
- 3個舊Codex分支已有bundle＋dirty/index／ignored證據備份；沒有刪除授權、不因備份完成就清理。不讓W7清理取代W5勝方目標。
- 原8個manual templates不用重做；獨立空間真值與4個已知strict-xfail仍未解，相關分數0.894不代表對位正確率。

## Claude 審查後的回報格式

每項完成用中文記：commit（無則寫無）、實際測試數、獨立審查結果、量測數字與分母、下一步／阻礙。不將本檔勾選等同產品驗收，不把檔案寫好說成Claude已讀。

## 本次自查

- [x] 依既有計畫保留「真實可驗證核心勝方」目標與W1–W7範圍，沒有縮成只做資料匯出。
- [x] 可直接執行的資料／訓練命令已對現版CLI `--help`及source核對；Task1的v1資料已保存，v2待task19補標後建立，Task2後未執行。
- [x] 不捏造不存在的接觸介面／訓練續跑參數／新測試結果；需後續設計的接觸方案明列給Claude決策。
- [x] 保存review focus、空框、val洩漏、模型選擇／backend差異、身份與時間順序的風險；未聲稱取得新勝方或重訓成功。
