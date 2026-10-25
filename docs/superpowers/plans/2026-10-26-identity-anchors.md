# 發射前身分錨點實作計畫

**Goal:** 動態裁判的中途碎片不得因為重新出現於左/右半邊就被當作新的X/B；以發射器與發射區共同觀測建立可追溯身分，沒有足夠證據就未知。
**Architecture:** 新identity.py只處理身分證據，消費同一參考座標中的真實陀螺/發射器軌跡；觀測前綴需有held_s要求的連續source frames、位於發射區聯集且近身發射器的區域身分一致。重疊發射區由唯一發射器區域解歧義，不推論物理接合。缺格重新計連續觀測；前綴已不支持或不同side就拒絕，不在賽後拾取時重建。繼承僅透過既有已檢查members；相反錨點永不選一方。
**Tech stack:** 現有Python/dataclasses與geometry函式；不新增模型/依賴。

## 約束

既有feat/local-judge；不改原始影片、先前結果、人工修訂。只在dynamic裁判使用新錨點，static既有路徑是獨立明確模式。設定直接來自launch_phase.yaml已有held_s/body_margin_px，不增加外觀模型或訓練。接續仍覆核，未定身分的有效觀測不得構成自動勝方。

## 步驟

- [x] 保存原始偵測的ByteTrack/BoTSORT對照與OCSORT介面不相容限制，不因片段數少就換追蹤器。
- [x] 先寫RED：中途半邊不分side、前綴發射器證據才anchor、跨中線不重設、重疊區/同side重複框、相反body/缺frame/賽後拾取拒絕、members繼承、dynamic不明身分強制無勝方。
- [x] 實作identity.py與pipeline接線，記錄geometry.identity的anchors與canonical sides，移除dynamic以first-point半邊猜side的用法。
- [x] 原片相同cache重播，與66人工物件分開檢查「能追溯」和「正確side已指派」，不把unknown算正確，記錄覆蓋率與錯側。
- [x] 相關測試/新完整原片freshCLI，無VLM；唯讀審查、核實發現、修正/驗證後commit，更新工作台。

## 邊界

X/B兩body同時近身為未知；新anchor後中線變化不能翻側；geometry未知不補舊變換；同一side的重複launcher框只提供一個區域side，不聲稱機械接合。

## 審查裁決

Important成立：僅檢查原始起始影格不能識別前綴中段geometry失敗。已以raw觀測0..5而geometry f1未知、2..5誤定錨重現RED；新介面強制傳完整source_frames，若任何已有raw觀測被geometry刪除，未完成anchor的前綴在該處截止。真漏偵raw無此frame仍可重數。19針對測試GREEN，唯讀複審解除此項。首輪全套為避免混用版本中止，不算通過；只採suite-final完整輸出。

## 最終驗證

suite-final.log：1307passed，4既有strict xfail，1既有FutureWarning，232.08秒，exit0。新完整原片fresh CLI225.562246秒exit0，formal速度.882352PASS(VLM關閉)，review3/3FAIL，truth0/3尚無整體準確率。27軌跡、3接續比前輪多2；66人工影格觀測38side正確、0side錯誤、28unknown（不算correct）。6854輸出觀測皆可回原native觀測；原片與人工submissions SHA保持。唯讀審查1Important成立已RED/GREEN修正，複審無阻斷。HTTP15資源200，但本輪CUA視覺預覽被工具URL政策阻擋，沒有繞過或假稱視覺檢查成功。
