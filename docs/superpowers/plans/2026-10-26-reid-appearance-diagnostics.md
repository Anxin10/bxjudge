# 多格外觀診斷 Implementation Plan

> 依既有規格 inline 執行，使用 executing-plans；沿用 feat/local-judge，不另開分支或工作樹。

**Goal:** 將五組已保存人工核對、發射頭端誤偵註記與原始多格裁切凍結，提供可重播的外觀距離診斷頁。
**Architecture:** 獨立 reid_appearance 模組只計算 OpenCV 色彩直方圖與梯度描述子；離線 builder 讀來源 SHA／native observations／指定人工修訂，串流解碼一次並輸出不可覆寫的證據目錄。沒有匹配門檻或自動接續，無對裁判核心的接線。
**Tech Stack:** 現有 OpenCV、NumPy、YAML、ffmpeg 與 video.iter_frames(cpu)，無新依賴或模型。
**Spec:** ../specs/2026-10-26-occlusion-collision-reidentification.md §4、7、8、9 第二層的離線校準子集。

## Global Constraints

- 原片、原人工提交、先前判決／軌跡／geometry 不改；所有輸出 applied=false。
- 四組 same 是局部人工作證，不把整個 native ID 或前後每張裁切當可靠身分參考。
- 第五組只確認藍框①f1713為發射頭端，不能推論全 track15 或將 unknown 改成不同顆。
- 人工未知保留總人口；沒有不同顆對照，不產生匹配閾值、準確率、eligible_proposal 或 X/B。
- 原始 BGR 裁切，來源格／原框／推論尺寸／雜湊可追溯；不使用燒字短片作描述子。
- 離線設定放 config/reid_appearance.yaml；嚴格欄位驗證，採樣與裁切設定屬未校準的診斷參數。

## Review Focus

- 側別或人工作證外推：pair same 不等於每格清楚可見；不得建立可信模板。
- 修訂與端點誤偵：註記必須綁已保存labels SHA、case與before/after，不按文字猜類別或刪整條軌跡。
- 僅正例／相鄰格：不把距離或格數當準確率／獨立樣本；無負例須明示未校準。
- 媒體對位：原始frame0串流，縮放框一致；缺格與類別不符拒絕，不插點。
- 中止與失敗：新out才可發布；來源變更或解碼錯誤不發布半成品，串流需close回收。

## Task 1: 可用的離線外觀證據頁

**Files:** config/reid_appearance.yaml；src/bxjudge/reid_appearance.py；scripts/analyze_reid_appearance.py；scripts/build_workbench.py；tests/test_reid_appearance.py；README與兩份規格狀態。
**Interfaces:** describe_crop(BGR, box, policy) 回傳品質測量／描述子；sample_observations(track, endpoint, direction, fps, policy) 只回原觀測；build(manifest_path, labels_path, annotations_path, config_path, output) 原子發布report.json/index.html/crops。

- [x] RED：相同／異色描述子，無梯度與不合法框、截斷、非有限值、嚴格設定、實際source格取樣不補缺格。
- [x] RED：合成短片完整builder；hash不符／錯誤annotation版本拒絕；未知保留、endpoint排除不外推；重跑不覆寫；失敗無output。
- [x] GREEN：最小描述子與離線builder，合成測試通過。
- [x] 凍結最新人工修訂與顯式endpoint註記，對原片產出五組診斷；保護SHA、無模型、記完整離線wall time。
- [x] 獨立唯讀審查，逐條親驗修正；測試、上線獨立子頁、更新狀態。

## 驗證結果與審查裁決

- 初版全套1347 passed、4既有xfail、1既有warning；審查修正後15項新模組測試及23項核對／媒體／HTTP測試，合計38 passed。新增與修改Python檔ruff通過。
- 審查P2①成立：無可用裁切誤稱人工非陀螺。兩個RED測例確認，改逐端點explicit annotation；品質拒絕顯示small_crop等原因，不外推另一端；GREEN。
- 整合補查P2②成立：上線副本未驗證至原人工／端點註記SHA。兩個副本竄改RED測例確認，report新增frozen_files_sha256與annotations_path，生成與發布皆驗證；GREEN。無未修復阻斷項。
- 最終產物`runs/reid-appearance-20261026/diagnostics-final`；report SHA `03320303b3c2d1befb9002d341b7040a499a87738d283bfe53b8ca539861bcb5`；完整離線CLI時間8.98秒，非裁判速度。前版未刪除。
- 五案例共62個crop observation（包含同來源重用）、181個影格兩兩距離；只有4個人工same區間，不把181當獨立成功樣本。第5before排除、after保留作上下文，原uncertain保留。
- 66個保護檔案SHA不變；未修改模型、裁判軌跡、原結果或人工標記。原片／native cache／代碼／設定／labels、端點註記與無損crop均驗證SHA。
- 工作台新增`--appearance-set`，驗證所有來源／凍結副本／crop再發布；沿用8504服務及原兩核對端點。HTTP與靜態來源檢查通過；未做瀏覽器視覺驗證，未繞過先前工具政策限制。
- 詳細記錄在`.superpowers/sdd/reid-appearance-20261026/`。核心速度與結果仍引用原實測`6e68fcc`，本層沒有重新量測核心或增加任何自動接續。

## 交付界線

此層是校準用的描述子與品質測量，不是完整shadow聯合指派。scope全程、可信X/B參考、多方唯一性、可達範圍與門檻均未驗收。不得聲稱解決全部碎片或新裁判速度通過。
