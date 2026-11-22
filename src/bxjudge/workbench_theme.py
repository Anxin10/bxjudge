"""HUD shell shared by the local workbench pages (pure HTML rendering, no I/O)."""

from __future__ import annotations

import html
import json
import re

NAV = (
    ("/", "總覽", "home"),
    ("/judge/", "回合裁判", "judge"),
    ("/reid-review/", "斷點核對", "reid-review"),
    ("/reid-challenge/", "挑戰核對", "reid-challenge"),
    ("/identity-audit/", "逐格核對", "identity-audit"),
    ("/two-bey/", "兩顆軌跡", "two-bey"),
)
ASSETS = '<link rel="stylesheet" href="/_theme/bx_theme.css"><script defer src="/_theme/bx_fx.js"></script>'
_SHELL = re.compile(r"<!--bx-shell-->.*?<!--/bx-shell-->", re.DOTALL)


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def json_script(element_id: str, data) -> str:
    """Embed JSON safely inside a <script> element."""
    text = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    return f'<script id="{esc(element_id)}" type="application/json">{text}</script>'


def top_strip(title: str, active: str) -> str:
    links = "".join(
        f'<a href="{href}"{" aria-current=page" if key == active else ""}>{label}</a>'
        for href, label, key in NAV
    )
    return (
        f'<header class="bx-top" data-bx-workbench="1"><div class="brand">BX<b>JUDGE</b></div>'
        f'<h1 id="bx-title">{esc(title)}</h1><nav aria-label="工作台導覽">{links}</nav>'
        '<button type="button" class="fx-toggle" aria-pressed="false">動效</button></header>'
    )


def status_bar(items: list[tuple[str, str]], state: str | None = None, element_id: str = "bx-status") -> str:
    attr = f' data-state="{esc(state)}"' if state else ""
    parts = "".join(f"<span>{esc(k)} <b>{esc(v)}</b></span>" for k, v in items)
    return f'<footer class="bx-status" id="{esc(element_id)}" role="status"{attr}>{parts}</footer>'


def page(title: str, active: str, body: str, *, status: str = "", head: str = "") -> str:
    return (
        '<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{esc(title)} · BXJUDGE</title>{ASSETS}{head}</head>"
        f'<body class="bx">{top_strip(title, active)}<main class="bx-shell">{body}</main>{status}</body></html>'
    )


def inject_shell(document: str, title: str, active: str) -> str:
    """Add the theme top strip to an older generated page without touching its own markup/JS."""
    clean = _SHELL.sub("", document)
    block = f"<!--bx-shell-->{ASSETS}{top_strip(title, active)}<!--/bx-shell-->"
    match = re.search(r"<body\b[^>]*>", clean, flags=re.IGNORECASE)
    if match:
        return clean[: match.end()] + block + clean[match.end():]
    if "</style>" in clean:
        return clean.replace("</style>", "</style>" + block, 1)
    match = re.search(r"<meta[^>]*viewport[^>]*>", clean, flags=re.IGNORECASE)
    if match is None:
        raise ValueError("no safe insertion point for workbench shell")
    return clean[: match.end()] + block + clean[match.end():]


REID_IDS = ("saved", "restore", "cases", "progress", "save", "export", "status", "error", "data")


def render_reid_page(manifest: dict, digest: str) -> str:
    """Review page for a reid manifest; element ids and data contract match reid_review.js."""
    count = len(manifest["cases"])
    body = f"""
<section class="bx-block"><header><span class="block-label live">Review · 人工核對</span><h2>只核對新增的 {count} 組斷點</h2></header>
<p class="notice">不用重做原本三回合覆核。這裡沒有預填人工作證，也不改勝負。看不清可選「看不清／不知道」。X／B 與可見性可另外標記，沒有把握可以留空。</p>
<p class="muted">先播放前後短片，再比較青框①／珊瑚框②（①②只是片段編號，與 X／B 無關）。確認本組後按底部「保存到專案」；瀏覽器草稿不等於已保存。</p>
<div class="panel load controls-row"><span class="tag c-gold">已存版本</span><select id="saved" aria-label="已存專案版本"><option>讀取已存版本</option></select><button id="restore" disabled>載入選定版本</button></div>
</section>
<div id="cases"></div>
<section class="panel toolbar focus"><strong id="progress"></strong><div class="buttons controls-row"><button id="save" class="primary">保存到專案</button><button id="export">匯出 JSON 備份</button></div><p id="status" role="status">尚未保存；草稿僅在此瀏覽器。</p><p id="error" role="alert" class="c-orange"></p></section>
<p><small class="mono">manifest SHA-256 {esc(digest)} · 新核對獨立保存；原提交與判決保留。</small></p>
{json_script("data", {"manifest": manifest, "sha": digest})}
<script type="module" src="reid_review.js"></script>"""
    status = status_bar([("核對", f"0／{count}"), ("manifest", digest[:12])], "review", "reid-status")
    return page("斷點局部核對", "reid-review", body,
                status=status, head='<link rel="stylesheet" href="reid_review.css">')


def render_judge_page() -> str:
    body = """
<section class="bx-block"><div class="round-list" id="round-list" role="tablist" aria-label="回合清單"></div></section>
<section class="bx-block" id="live" data-state="review">
 <header><span class="block-label live">Live · 本回合判定</span><h2 id="round-title">載入中…</h2><span class="tag state-text" id="state-tag">—</span></header>
 <div class="live-grid">
  <div class="panel video-panel focus">
   <div class="stage" id="stage"><video id="video" preload="metadata" playsinline muted></video><div class="scan" aria-hidden="true"></div>
    <svg class="overlay" id="overlay" aria-hidden="true"></svg><div class="empty" id="video-empty" hidden></div></div>
   <div class="controls-row"><button id="play" type="button">播放回合</button><button id="to-launch" type="button">跳到發射</button><button id="to-end" type="button">跳到終結</button>
    <span class="mono" id="clock">--</span>
    <label><input type="checkbox" id="show-candidates" checked>候選框</label><label><input type="checkbox" id="show-zones" checked>發射參照區域（非逐格對位）</label><label><input type="checkbox" id="show-trails" checked>軌跡</label></div>
   <div class="legend"><span class="c-cyan">主目標（X／B）</span><span class="c-grey">未採用候選</span><span class="c-coral">軌跡</span><span class="c-purple">場地幾何（鎖定）</span><span class="c-orange">待覆核</span></div>
  </div>
  <div class="panel" id="details"><h3>模型 / 偵測</h3><dl class="kv" id="detail-kv"></dl><h3 style="margin-top:14px">逐項檢查</h3><ul class="checklist" id="checks"></ul></div>
  <div class="decision">
   <div class="panel out-card"><h3>Output · 判定</h3><div class="big" id="out-main">—</div><div id="out-sub" class="muted"></div>
    <div class="meter" id="meter"><i></i><b></b></div><div class="mono"><span id="conf">0.00</span> <span class="threshold" id="th-sign">≥</span> 門檻 <span id="th">—</span></div></div>
   <div class="panel fb-card"><h3>Fallback · 人工覆核</h3><div class="state-text" id="fb-state">—</div><ul class="checklist" id="reasons"></ul></div>
  </div>
 </div>
</section>
<section class="bx-block"><header><span class="block-label">Background · 流程與證據</span></header>
 <div class="bg-grid"><div class="panel"><svg class="flow" id="flow" viewBox="0 0 760 210" role="img" aria-label="判定流程節點"></svg></div>
 <div class="panel"><h3>證據庫</h3><ul class="store-list" id="stores"></ul><p class="muted" id="top-reasons"></p></div></div>
</section>
<noscript><p class="notice">本頁需要 JavaScript 讀取 run 輸出。</p></noscript>
<script type="module" src="judge_view.js"></script>"""
    status = status_bar([("回合", "—"), ("信心", "—"), ("Fallback", "—"), ("證據庫", "—")], "review", "bx-status")
    return page("回合裁判", "judge", body, status=status)


def _stat(label, value, note, colour="c-cyan"):
    return (f'<article class="panel"><h3>{esc(label)}</h3><div class="big {colour}">{esc(value)}</div>'
            f'<small>{esc(note)}</small></article>')


def render_hub(snapshot: dict, judge: dict | None, *, previous_hub: str | None = None) -> str:
    """Overview page from the validated hub snapshot plus the current judge index."""
    rows = snapshot.get("rounds") or []
    side = snapshot.get("side_assignment") or {}
    cont = snapshot.get("continuation") or {}
    two = (snapshot.get("two_bey_preview") or {}).get("previews") or []
    speed_ok = snapshot.get("speed_gate") == "PASS"
    cards = [
        _stat("核心速度（兩支開發原片）", f"{snapshot.get('speed_gate')} · {snapshot.get('speed_ratio', 0):.3f}×",
              "關閉 VLM；不代表整批或準確率", "c-cyan" if speed_ok else "c-orange"),
        _stat("前輪人工紀錄", f"{snapshot.get('human_reviewed_rows_in_previous_results', 0)} 回合",
              "已覆核；沒有被新試跑覆寫", "c-gold"),
        _stat("身分指派（開發樣本）", f"{side.get('correct', 0)}✓ {side.get('wrong', 0)}✗ {side.get('unknown', 0)}?",
              "未知沒有算成判對", "c-purple"),
    ]
    if judge:
        review, total = judge["review_count"], judge["round_count"]
        cards.insert(0, _stat("目前判定 run", f"{review}／{total} 待覆核",
                              "點進回合裁判看每回合原因", "c-orange" if review else "c-cyan"))
        stage_names = {"detect": "偵測", "track": "兩顆追蹤", "geometry": "場地幾何",
                       "identity": "身分", "finish": "終結判定", "other": "其他"}
        reasons = "".join(
            f'<li><span>{esc(code)} <small class="c-grey">{esc(stage_names.get(stage, stage))}</small></span><b>{n}</b></li>'
            for code, n, stage in judge["top_reasons"])
        stores = "".join(f"<li><span>{esc(k)}</span><b>{v}</b></li>" for k, v in judge["stores"].items())
        judge_block = f"""<section class="bx-block"><header><span class="block-label live">Live · 目前判定</span><h2><a href="/judge/">回合裁判台 →</a></h2>
<span class="tag {'c-orange' if review else 'c-cyan'}">{'REVIEW' if review else 'EMIT'}</span></header>
<div class="bg-grid"><div class="panel focus"><h3>主要覆核原因</h3><ul class="store-list">{reasons or '<li>無</li>'}</ul>
<p class="muted mono">run：{esc(judge['run'])}</p></div><div class="panel"><h3>證據庫</h3><ul class="store-list">{stores or '<li>無</li>'}</ul></div></div></section>"""
    else:
        review = total = 0
        judge_block = '<p class="notice">尚未產生回合裁判資料（/judge/）。</p>'
    result_rows = "".join(
        f'<tr><td>{esc(r["video"])}<br><small>第 {esc(r["round"])} 回合</small></td>'
        f'<td>{esc(r["ai_outcome"])}<br><span class="tag {"c-orange" if r["ai_needs_review"] else "c-cyan"}">'
        f'{"待覆核" if r["ai_needs_review"] else "AI 未標覆核"}</span></td>'
        f'<td>{esc(r["human_outcome_recorded"])}<br><span class="tag {"c-gold" if r["human_reviewed_in_previous_results"] else "c-grey"}">'
        f'{"已覆核（原紀錄）" if r["human_reviewed_in_previous_results"] else "無已完成覆核紀錄"}</span></td></tr>'
        for r in rows)
    previews = "".join(
        f'<li><span>{esc(p["video"])} · 第 {esc(p["round"])} 段</span><b>{esc(p.get("entity_count"))} 實體</b></li>'
        for p in two)
    history = [("/identity-audit/", "逐格核對與儲存（保留草稿）"), ("/reid-appearance/", "外觀診斷"),
               ("/reid-contrast/", "不同顆對照"), ("/fragment-refinement/", "碎片接回對照"),
               ("/batch-outcomes/", "批次結果"), ("/launch-motion/", "發射動作"), ("/launch-state/", "發射狀態"),
               ("/launch-diagnostics/", "發射診斷"), ("/optimization/", "速度與驗收記錄"),
               ("/index-before-unified.html", "統一前舊首頁")]
    if previous_hub:
        history.append(("/" + previous_hub, "上一版工作台總覽"))
    links = "".join(f'<li><a href="{href}">{esc(label)}</a></li>' for href, label in history)
    body = f"""
<section class="bx-block"><div class="cards">{''.join(cards)}</div></section>
{judge_block}
<section class="bx-block" id="review"><header><span class="block-label">Review · 待你核對</span></header><div class="cards">
<article class="panel"><h3>斷點核對</h3><p><a href="/reid-review/">{esc((snapshot.get('local_reid_review') or {}).get('case_count', 0))} 組：同一顆／不同顆／看不清</a></p><small>獨立保存，不改判決</small></article>
<article class="panel"><h3>挑戰核對</h3><p><a href="/reid-challenge/">重疊／相似／模糊 挑戰組</a></p><small>{esc(', '.join(f'{k} {v}' for k, v in ((snapshot.get('challenge_review') or {}).get('selected_by_category') or {}).items()))}</small></article>
<article class="panel"><h3>兩顆連續軌跡</h3><ul class="store-list">{previews or '<li>無</li>'}</ul><p><a href="/two-bey/">播放兩實體影片</a></p></article>
</div></section>
<section class="bx-block" id="results"><header><span class="block-label">Background · 前輪裁判紀錄</span></header><div class="panel">
<p class="notice">「原紀錄已覆核」與「新 AI 試跑待覆核」是兩組不同資料；未填的終結欄位不會擅自當作平手或重賽。</p>
<div style="overflow-x:auto"><table><thead><tr><th>來源回合</th><th>AI 試跑</th><th>先前人工紀錄</th></tr></thead><tbody>{result_rows}</tbody></table></div>
<p class="muted">軌跡 {esc(cont.get('before_tracks'))} → {esc(cont.get('after_tracks'))} 條，接回 {esc(cont.get('applied_joins'))} 次；人工標記衝突 {esc(snapshot.get('identity_label_conflicts'))} 筆。<a href="/_workbench/snapshot.json">資料來源與雜湊</a></p></div></section>
<section class="bx-block" id="history"><header><span class="block-label">Archive · 研究紀錄</span></header><div class="panel"><ul class="store-list">{links}</ul></div></section>"""
    status = status_bar([("待覆核", f"{review}／{total}" if judge else "—"),
                         ("速度", str(snapshot.get("speed_gate"))),
                         ("人工紀錄", str(snapshot.get("human_reviewed_rows_in_previous_results", 0))),
                         ("core", str(snapshot.get("core_commit", ""))[:10])],
                        "review" if review else "judged")
    return page("工作台總覽", "home", body, status=status)
