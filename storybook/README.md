# 工作台 Storybook

用來單獨檢視本機工作台頁面的各種狀態，不需要跑 pipeline 或開 `bxjudge` 伺服器。

| 分類 | 頁面 | 狀態 |
|---|---|---|
| 工作台/總覽 | 首頁（`render_hub`） | 有判定 run、尚無判定 run |
| 工作台/回合裁判 | `/judge/`（`judge_view.js`） | 待覆核、已判定、已人工覆核、場地幾何不可用、沒有回合、讀不到資料 |
| 工作台/斷點核對 | `/reid-review/`（`reid_review.js`） | 空白草稿、有已存版本、瀏覽器草稿進行中、草稿無法驗證、保存失敗 |
| 主題/HUD 元件 | `bx_theme.css` | 色彩與標籤、按鈕與表單、面板與狀態 |

## 執行

在 WSL 的專案根目錄：

```bash
npm install
npm run storybook          # http://127.0.0.1:6006
npm run build-storybook    # 輸出靜態檔到 storybook-static/
```

兩個指令都會先執行 `npm run fixtures`，也就是 `python3 storybook/build_fixtures.py`。這一步用**真正的** Python 渲染函式（`workbench_theme.py`）產生頁面骨架，放到 `storybook/generated/`。這個資料夾不進 git。只需要系統內建的 Python，不需要專案的 venv。

## 運作方式

- 頁面骨架來自 Python 渲染函式，頁面程式直接載入 `src/bxjudge/web/` 的原始檔，所以畫面和實際工作台一致。
- 頁面程式在載入時就會開始執行，所以每個 story 都重新載入一份新的（blob URL），並移除上一個 story 加在 `window`／`document` 的事件。
- `fetch` 在 story 內被替換成假資料（`storybook/harness.js`），不會連到任何伺服器，也不會寫入專案。

## 資料

**全部都是合成資料**（`storybook/fixtures/`）：

- 影片名稱、軌跡、框和場地區域都是程式產生的；
- 斷點核對的「來源影格」是畫出來的 SVG，不是影片畫面；
- 回合裁判沒有附影片，所以會顯示「找不到原影片」。疊加框畫在 t = 1.0 秒（用負的 `media_offset` 達成）。

不要把 `data/`、`runs/`、`calib/` 或影片畫面放進這裡。這些資料不能上傳到任何雲端。
