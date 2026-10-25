"""Build a local, read-only entry point from explicit result and evidence folders.

PYTHONPATH=src .venv/bin/python scripts/build_workbench.py --help
No predictions, manual reviews or annotation submissions are written.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
from pathlib import Path

from bxjudge import evaluate as ev
from bxjudge.calibration_store import vetoed_ids
from bxjudge.output import effective, read_rounds


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(value) -> str:
    return html.escape(str(value), quote=True)


def _outcome(row: dict) -> str:
    names = {"spin": "耐久", "burst": "爆裂", "over": "超出", "xtreme": "極限"}
    finish = row.get("finish")
    if finish is None:
        return "未記錄終結／勝方"
    return f"{names.get(finish, finish)} · {row.get('winner') or '勝方未記錄'} · {row.get('points', 0)} 分"


def build(site: Path, results: Path, human_results: Path, evidence: Path, core_commit: str):
    site, results, human_results, evidence = (
        p.resolve() for p in (site, results, human_results, evidence)
    )
    if site in (results, human_results, evidence):
        raise ValueError("site must be separate from input results and evidence")
    rows = read_rounds(results)
    human_rows = read_rounds(human_results)
    timings = _read(results / "videos.json")
    runs = _read(results / "runs.json")
    identity = _read(evidence / "identity-evidence.json")
    attachment = _read(evidence / "attachment-evidence.json")
    reference = identity["reference"]
    trajectories = {p.name: _sha(p) for p in (results / "trajectories").glob("*.csv")}
    if trajectories != identity["source_csv_sha256"]:
        raise ValueError("identity evidence does not match result trajectories; regenerate evidence")
    if reference != attachment["reference"] or _sha(Path(reference["reference"])) != reference["sha256"]:
        raise ValueError("identity and attachment must use the same intact annotation revision")
    config = ev.load_eval_config()
    metrics = ev.evaluate(rows, timings, config=config, runs=runs,
                          vetoed=vetoed_ids(results), audit=ev.read_audit(results))
    gates = ev.passes(metrics, config)
    real = ev.real_rows(rows)
    human_by_key = {(r["video"], r["round"]): r for r in human_rows}
    result_rows = []
    for row in real:
        manual = human_by_key.get((row["video"], row["round"]))
        result_rows.append({
            "video": row["video"], "round": row["round"],
            "ai_outcome": _outcome(row), "ai_needs_review": row["needs_review"],
            "human_reviewed_in_previous_results": bool(manual and manual.get("reviewed")),
            "human_outcome_recorded": _outcome(effective(manual)) if manual else "無既有人工紀錄",
            "human_record_scope": "previous results; not merged into fresh AI predictions",
        })
    source_files = [results / n for n in ("rounds.json", "videos.json", "runs.json")]
    source_files += sorted((results / "trajectories").glob("*.csv"))
    source_files.append(Path("config/evaluation.yaml").resolve())
    source_files += [human_results / "rounds.json", evidence / "identity-evidence.json",
                     evidence / "attachment-evidence.json", Path(reference["reference"])]
    note_path = human_results / "human-review-notes.md"
    note = note_path.read_text() if note_path.is_file() else ""
    if note:
        source_files.append(note_path)
    snapshot = {
        "core_commit": core_commit, "results": str(results),
        "human_results": str(human_results), "annotation_reference": reference,
        "rounds": result_rows, "speed_gate": gates["speed_ratio"],
        "speed_ratio": metrics.speed_ratio,
        "review_ratio": metrics.review_ratio,
        "human_reviewed_rows_in_previous_results": sum(bool(r.get("reviewed")) for r in human_rows),
        "identity_exact_matched_objects": identity["uniquely_exact_matched"],
        "identity_label_conflicts": len(identity["tracks_with_confirmed_side_conflict"]),
        "trajectory_count": identity["track_count"],
        "source_sha256": {str(p): _sha(p) for p in source_files},
        "scope": "local read-only hub; speed applies only to supplied two development originals with VLM disabled; no autonomous referee accuracy claim",
    }
    site.mkdir(parents=True, exist_ok=True)
    assets = site / "_workbench"
    assets.mkdir(exist_ok=True)
    for name in ("identity-three-frames.jpg", "identity-evidence.json", "attachment-evidence.json"):
        shutil.copyfile(evidence / name, assets / name)
    cards = []
    for item in attachment["records"]:
        source = Path(item["crop_file"])
        if _sha(source) != item["image_sha256"]:
            raise ValueError(f"attachment crop changed: {source}")
        shutil.copyfile(source, assets / source.name)
        state = {"mounted": "仍掛載", "separated": "已分離", "unknown": "不確定"}[item["human_state"]]
        cards.append(f'<figure><a href="/_workbench/{_text(source.name)}"><img src="/_workbench/{_text(source.name)}" alt="來源 f{item["frame"]} 接合區原始裁切"></a><figcaption>f{item["frame"]} · 人工：{state}<br><small>原始解析度裁切；未推測中心／勾爪座標</small></figcaption></figure>')
    body_rows = "".join(
        f'<tr><td>{_text(x["video"])}<br>第 {x["round"]} 回合</td><td>{_text(x["ai_outcome"])}<br><span class="tag warn">{"待覆核" if x["ai_needs_review"] else "AI 未標覆核"}</span></td><td>{_text(x["human_outcome_recorded"])}<br><span class="tag">{"已覆核（原紀錄）" if x["human_reviewed_in_previous_results"] else "無已完成覆核紀錄"}</span></td></tr>'
        for x in result_rows
    )
    all_review = sum(x["ai_needs_review"] for x in result_rows)
    manual_count = snapshot["human_reviewed_rows_in_previous_results"]
    speed = _text(gates["speed_ratio"])
    headline = "核心速度已通過" if gates["speed_ratio"] == "PASS" else "核心速度未通過"
    speed_class = "good" if gates["speed_ratio"] == "PASS" else ""
    revision_link = "/identity-audit/revisions/" + reference["revision"][:16] + "/"
    page = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BXJUDGE 工作台</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f4f6fa;color:#192a40;font:16px/1.7 system-ui,sans-serif}}header{{background:#132844;color:#fff}}.shell{{max-width:1120px;margin:auto;padding:24px}}header .shell{{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}}.brand{{font-weight:800;letter-spacing:.08em}}nav{{display:flex;gap:20px;flex-wrap:wrap}}header a{{color:#dce9fb;text-decoration:none}}a{{color:#195e9a}}h1{{font-size:30px;line-height:1.35;margin:8px 0 14px}}h2{{font-size:23px;margin:0 0 12px}}h3{{font-size:18px}}p{{margin:10px 0}}.muted,small{{color:#5a6b7d}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}}.card,section{{background:#fff;border:1px solid #dce3ed;border-radius:14px;padding:23px}}.card b{{display:block;font-size:29px;margin:9px 0}}.good{{color:#237347}}section{{margin:24px 0;scroll-margin-top:20px}}.notice{{border-left:4px solid #d6a03c;padding:12px 16px;background:#fff8e8;border-radius:6px}}.tag{{display:inline-block;padding:3px 9px;border-radius:6px;background:#eaf0f7;font-size:13px;margin-top:8px}}.warn{{background:#fff0d2;color:#885c0b}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{padding:15px 10px;text-align:left;vertical-align:top;border-bottom:1px solid #e1e7ef}}th{{color:#50627a}}.tablewrap{{overflow-x:auto}}.triptych{{width:100%;height:auto;border-radius:8px}}.gallery{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}figure{{margin:0;border:1px solid #e1e7ef;border-radius:9px;overflow:hidden}}figure img{{width:100%;height:245px;object-fit:contain;background:#101820;display:block}}figcaption{{padding:13px}}.links{{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}}.links a{{display:block;background:#f2f5f9;padding:12px;border-radius:7px;text-decoration:none}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 system-ui;background:#f5f7fa;padding:16px}}summary{{cursor:pointer;font-weight:600}}footer{{padding:12px 0 25px;color:#5a6b7d;font-size:14px}}@media(max-width:760px){{.cards,.gallery,.links{{grid-template-columns:1fr}}.shell{{padding:18px}}h1{{font-size:25px}}.card,section{{padding:18px}}figure img{{height:auto}}}}
</style><header data-bx-workbench="1"><div class="shell"><div class="brand">BXJUDGE <span style="font-weight:400">工作台</span></div><nav aria-label="工作台導覽"><a href="#results">目前結果</a><a href="#identity">X／B 身分</a><a href="#attachment">中心接合</a><a href="#history">研究紀錄</a></nav></div></header><main class="shell"><p class="muted">統一入口 · 以原始影片與已保存人工紀錄為準</p><h1>{headline}；勝負仍需人工覆核</h1><p>這裡集中最新結果與證據。原本的 launch-motion、launch-state 等頁面保留作研究紀錄，不再當作最新首頁。</p><div class="cards"><article class="card">兩支開發原片的核心速度<b class="{speed_class}">{speed} · {metrics.speed_ratio:.3f} 倍</b><small>關閉 VLM；未代表整批 109 片或準確率</small></article><article class="card">最新 AI 試跑<b>{all_review}／{len(result_rows)} 待覆核</b><small>尚無可直接採用的自動勝負結果</small></article><article class="card">你先前的人工紀錄<b>{manual_count} 回合已覆核</b><small>原紀錄保留，沒有被新試跑覆寫</small></article></div>
<section id="results"><h2>目前結果</h2><p class="notice">「原紀錄已覆核」與「新 AI 試跑待覆核」是兩組不同資料。人工紀錄沒有消失，也不要求你重做全部核對。未填的終結欄位不會擅自當作平手或重賽。</p><div class="tablewrap"><table><thead><tr><th>來源回合</th><th>最新 AI 試跑</th><th>先前人工紀錄</th></tr></thead><tbody>{body_rows}</tbody></table></div><details style="margin-top:18px"><summary>你對事件先後與重賽的補充</summary><pre>{_text(note or "此資料夾沒有另存的事件補充。")}</pre></details><p><a href="/optimization/">速度、程式驗證與未完成項目</a> · <a href="/_workbench/snapshot.json">本頁資料來源與雜湊</a></p></section>
<section id="identity"><h2>X／B 身分：先校正證據，再修追蹤</h2><p>已將 <b>{identity['uniquely_exact_matched']} 個</b>已標記物件，依相同來源影格、類別、中心與分數精確對上整片軌跡。目前人工標記衝突 <b>{snapshot['identity_label_conflicts']} 筆</b>；整片仍有 <b>{identity['track_count']} 條</b>陀螺軌跡，不能因此宣稱身分追蹤已全面穩定。</p><p>f1618 已依你確認更正為深色 X、淺色 B。ID 3 與 ID 16 在這三格並未因這次修訂而改號；更正的是人工 side 標記，沒有擅改原始預測。</p><img class="triptych" src="/_workbench/identity-three-frames.jpg" alt="f1617、f1618、f1619 的深色與淺色陀螺，人工更正後一致"><p><a href="{revision_link}">目前人工修訂</a> · <a href="/_workbench/identity-evidence.json">逐物件對照證據</a></p></section>
<section id="attachment"><h2>中心接合：原始畫面與確認範圍</h2><p>觀察重點是<b>陀螺中心與發射器勾爪是否仍接合</b>。整個物件框有交集，不足以判定仍掛載；框分開的時間也不等於物理分離時間。</p><div class="gallery">{''.join(cards)}</div><p class="notice">已確認：f1565 仍掛載，f1567、f1568 已分離；f1566 尚未確認，因此首次已分離可能在 f1566 或 f1567，尚不能縮成單一影格。這些畫面尚無人工定位的中心／勾爪關鍵點，也沒有通過驗證的接合模型，不能用框中心冒充機械接合點。</p><p>下一步先處理軌跡碎片與接合點的可見性證據；原有三秒期限與未知狀態保持，不以之後落地倒推發射時點。</p><p><a href="/_workbench/attachment-evidence.json">原尺寸裁切、人工狀態與來源</a></p></section>
<section id="history"><h2>研究紀錄與既有核對頁</h2><p class="muted">以下是不同階段的實驗與核對工具。查目前狀態請回這個工作台。</p><div class="links"><a href="/identity-audit/">既有逐格核對與儲存（保留草稿）</a><a href="/optimization/">最新速度與驗收結果</a><a href="/launch-motion/">相對運動實驗（未採用）</a><a href="/launch-state/">VLM 三狀態實驗（保留不確定）</a><a href="/launch-diagnostics/">掛載／框分離診斷</a><a href="/index-before-unified.html">舊整合報告（歷史測量）</a></div></section><footer>裁判實測版本 {_text(core_commit[:7])} · 人工參考 {_text(reference['revision'][:16])}<br>本頁為本機唯讀工作台。未覆寫來源影片、模型預測、既有人工覆核或瀏覽器草稿。</footer></main></html>'''
    previous = site / "index.html"
    archive = site / "index-before-unified.html"
    if previous.is_file() and 'data-bx-workbench="1"' not in previous.read_text() and not archive.exists():
        shutil.copyfile(previous, archive)
    snapshot_path = assets / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    previous.write_text(page, encoding="utf-8")
    banner = (f'<aside data-workbench-navigation="1" style="padding:14px;background:#fff4d8;'
              f'color:#192a40;font:16px/1.6 system-ui"><a style="color:#195e9a" href="/">← 回統一工作台</a> · '
              f'本頁保留原實驗／核對紀錄；最新人工修訂：'
              f'<a style="color:#195e9a" href="{revision_link}">{_text(reference["revision"][:16])}</a>。'
              f'f1618：深色 X／淺色 B；舊表單與草稿未改寫。</aside>')
    for relative in ("identity-audit/index.html", "launch-motion/index.html",
                     "launch-state/index.html", "launch-diagnostics/index.html",
                     "optimization/index.html", "index-before-unified.html"):
        subpage = site / relative
        if not subpage.is_file():
            continue
        original = subpage.read_text(encoding="utf-8")
        updated = re.sub(r'<aside data-workbench-navigation="1".*?</aside>', '',
                         original, flags=re.DOTALL)
        if re.search(r'<body\b[^>]*>', updated, flags=re.IGNORECASE):
            updated = re.sub(r'(<body\b[^>]*>)', lambda m: m[0] + banner,
                             updated, count=1, flags=re.IGNORECASE)
        else:
            # Existing generated reports use an implicit body after their style block.
            if "</style>" not in updated:
                raise ValueError(f"missing report style/body boundary: {subpage}")
            updated = updated.replace("</style>", "</style>" + banner, 1)
        if updated != original:
            backup = subpage.with_name(subpage.stem + ".before-workbench.html")
            if not backup.exists():
                shutil.copyfile(subpage, backup)
            subpage.write_text(updated, encoding="utf-8")
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("site", "results", "human-results", "evidence"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--core-commit", required=True)
    args = parser.parse_args()
    snapshot = build(args.site, args.results, args.human_results, args.evidence, args.core_commit)
    print(json.dumps({k: snapshot[k] for k in ("core_commit", "speed_gate", "review_ratio", "identity_label_conflicts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
