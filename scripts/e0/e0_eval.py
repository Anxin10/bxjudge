"""E0 read-only evaluation: identity (side IDF1 / track IDF1 / ID switches / HOTA), stop-time error,
Finish Macro-F1, winner accuracy, human-review rate and processing speed for one `bxjudge run` output dir.

Truth: runs/e0-baseline-20261124/truth/e0-truth.json (keyframes every 0.25 s; see its `made_by`).
Predictions: <run>/trajectories/<video>_r<n>.csv (+ .json coordinate_size), <run>/rounds.json, <run>/runs.json.
Never writes into the run directory.

Usage: python e0_eval.py TRUTH RUN_DIR [--json OUT]
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

SIDES = ("X", "B")
ALPHAS = [round(0.05 * i, 2) for i in range(1, 20)]


def load_pred(run: Path, video: str, t_window):
    """bey rows per frame from every trajectory CSV of `video`, in 1920x1080 pixels."""
    stem = video[:-4]
    rows = defaultdict(list)
    for p in sorted((run / "trajectories").glob(f"{stem}_r*.csv")):
        meta = json.loads(p.with_suffix(".json").read_text())
        sx = 1920 / meta["coordinate_size"][0]
        sy = 1080 / meta["coordinate_size"][1]
        for r in csv.DictReader(open(p)):
            if r["cls"] != "bey" or not r["x"] or not r["y"]:  # lost/predicted-only rows carry no position
                continue
            rows[int(r["frame"])].append({
                "xy": (float(r["x"]) * sx, float(r["y"]) * sy),
                "side": r["side"] or None,
                "track": f"{p.stem}:{r['track_id']}",
            })
    return rows


def similarity(gt_box, xy):
    """1 at the box centre, 0 at (and beyond) the box's half-diagonal."""
    x0, y0, x1, y1 = gt_box
    r = math.hypot(x1 - x0, y1 - y0) / 2
    d = math.hypot((x0 + x1) / 2 - xy[0], (y0 + y1) / 2 - xy[1])
    return max(0.0, 1.0 - d / r)


def match(gts, preds, min_sim):
    if not gts or not preds:
        return []
    s = np.array([[similarity(g["box_xyxy"], p["xy"]) for p in preds] for g in gts])
    gi, pi = linear_sum_assignment(-s)
    return [(i, j, s[i, j]) for i, j in zip(gi, pi) if s[i, j] >= min_sim]


def identity_metrics(truth_v, pred):
    """Side-level (what the judge exports) and track-level identity over the truth keyframes."""
    side = Counter()          # idtp, wrong, abstain, miss, fp_side
    seq = {s: [] for s in SIDES}   # predicted side per keyframe for each true side (None = abstain/miss)
    trk = {s: [] for s in SIDES}   # predicted track per keyframe for each true side
    pairs = Counter()               # (true side, pred track) co-occurrence for track IDF1
    n_gt = Counter()
    n_pred_track = Counter()
    hota_frames = []
    for kf in truth_v["keyframes"]:
        objs = [o for o in kf["objects"] if o["side"] in SIDES]
        ignore = [o for o in kf["objects"] if o["side"] == "U"]
        preds = pred.get(kf["frame"], [])
        m = match(objs, preds, 1e-9)
        # predictions that fall on an "unknown" truth object are neither TP nor FP
        rest = [j for j in range(len(preds)) if j not in {b for _, b, _ in m}]
        im = match(ignore, [preds[j] for j in rest], 1e-9)
        ignored = {rest[b] for _, b, _ in im}
        matched_pred = {}
        for i, j, sim in m:
            matched_pred[i] = (j, sim)
        for i, o in enumerate(objs):
            n_gt[o["side"]] += 1
            if i not in matched_pred:
                side["miss"] += 1
                seq[o["side"]].append(None)
                trk[o["side"]].append(None)
                continue
            p = preds[matched_pred[i][0]]
            trk[o["side"]].append(p["track"])
            pairs[(o["side"], p["track"])] += 1
            if p["side"] is None:
                side["abstain"] += 1
                seq[o["side"]].append(None)
            elif p["side"] == o["side"]:
                side["idtp"] += 1
                seq[o["side"]].append(p["side"])
            else:
                side["wrong"] += 1
                seq[o["side"]].append(p["side"])
        for j, p in enumerate(preds):
            if j in ignored:
                continue
            n_pred_track[p["track"]] += 1
            if j not in {b for _, b, _ in m} and p["side"] is not None:
                side["fp_side"] += 1
        hota_frames.append((objs, [p for j, p in enumerate(preds) if j not in ignored]))
    return side, seq, trk, pairs, n_gt, n_pred_track, hota_frames


def switches(seq):
    """Count changes of a non-empty label along the keyframe sequence (None entries are skipped)."""
    last, n = None, 0
    for v in seq:
        if v is None:
            continue
        if last is not None and v != last:
            n += 1
        last = v
    return n


def track_idf1(pairs, n_gt, n_pred_track):
    tracks = sorted(n_pred_track)
    if not tracks:
        return 0.0, 0
    cost = np.array([[pairs.get((s, t), 0) for t in tracks] for s in SIDES])
    gi, ti = linear_sum_assignment(-cost)
    idtp = int(sum(cost[g, t] for g, t in zip(gi, ti)))
    idfn = sum(n_gt.values()) - idtp
    idfp = sum(n_pred_track.values()) - idtp
    return (2 * idtp / (2 * idtp + idfp + idfn) if idtp + idfp + idfn else 0.0), idtp


def hota(frames, key):
    """HOTA (Luiten et al. 2021) on sparse keyframes; `key` picks the prediction id ('track' or 'side')."""
    per_alpha = []
    for a in ALPHAS:
        tp_pairs, fn, fp = [], 0, 0
        gt_count, pr_count = Counter(), Counter()
        for objs, preds in frames:
            ids = [(p[key] if p[key] is not None else f"none:{id(p)}") for p in preds]
            for o in objs:
                gt_count[o["side"]] += 1
            for pid in ids:
                pr_count[pid] += 1
            m = match(objs, preds, a)
            fn += len(objs) - len(m)
            fp += len(preds) - len(m)
            tp_pairs += [(objs[i]["side"], ids[j]) for i, j, _ in m]
        tp = len(tp_pairs)
        if tp == 0:
            per_alpha.append((0.0, 0.0, 0.0))
            continue
        tpa = Counter(tp_pairs)
        ass = sum(tpa[(g, p)] / (gt_count[g] + pr_count[p] - tpa[(g, p)]) for g, p in tp_pairs) / tp
        det = tp / (tp + fn + fp)
        per_alpha.append((math.sqrt(det * ass), det, ass))
    h, d, s = (float(np.mean([x[k] for x in per_alpha])) for k in range(3))
    return {"HOTA": h, "DetA": d, "AssA": s}


def stop_time(round_row, truth_round, fps):
    """Predicted loser stop time in seconds, if the pipeline produced one; else None."""
    ev = round_row.get("motion_evidence") or {}
    for k in ("loser_stop_frame", "stop_frame"):
        if isinstance(ev.get(k), (int, float)):
            return ev[k] / fps
    return None


def interval_error(t, lo_hi):
    lo, hi = lo_hi
    return 0.0 if lo <= t <= hi else min(abs(t - lo), abs(t - hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("truth")
    ap.add_argument("run")
    ap.add_argument("--json")
    a = ap.parse_args()
    truth = json.loads(Path(a.truth).read_text())
    run = Path(a.run)
    rounds = json.loads((run / "rounds.json").read_text())
    runs = json.loads((run / "runs.json").read_text())
    work, duration = {}, {}
    for r in runs:
        work.update(r.get("work", {}))
        duration.update(r.get("videos", {}))

    out = {"run": str(run), "truth_made_by": truth["made_by"], "videos": {}}
    tot = Counter()
    all_frames_track, all_frames_side = [], []
    idf1_parts = Counter()
    finish_true, finish_pred = [], []
    winners = []
    review = []
    stop_err = []
    for video, tv in truth["videos"].items():
        tr = tv["round"]
        pred = load_pred(run, video, None)
        side, seq, trk, pairs, n_gt, n_pred_track, frames = identity_metrics(tv, pred)
        sw_side = sum(switches(seq[s]) for s in SIDES)
        sw_track = sum(switches(trk[s]) for s in SIDES)
        t_idf1, t_idtp = track_idf1(pairs, n_gt, n_pred_track)
        n = sum(n_gt.values())
        idfp = side["wrong"] + side["fp_side"]
        idfn = n - side["idtp"]
        s_idf1 = 2 * side["idtp"] / (2 * side["idtp"] + idfp + idfn) if n else 0.0
        h_track = hota(frames, "track")
        h_side = hota([(o, [dict(p, side=p["side"]) for p in ps]) for o, ps in frames], "side")
        # the truth round = pipeline round(s) overlapping the truth launch..stop window
        lo, hi = tr["launch_s"][0], tr["loser_stop_s"][1]
        cand = [r for r in rounds if r["video"] == video and r["t_launch"] <= hi and r["t_end"] >= lo]
        main_r = max(cand, key=lambda r: min(r["t_end"], hi) - max(r["t_launch"], lo)) if cand else None
        pf = main_r["finish"] if main_r else None
        pw = main_r["winner"] if main_r else None
        st = stop_time(main_r, tr, tv["fps"]) if main_r else None
        if not tr.get("ambiguous"):
            finish_true.append(tr["finish"])
            finish_pred.append(pf)
            winners.append(pw == tr["winner"] if pw else None)
            if st is not None:
                stop_err.append(interval_error(st, tr["loser_stop_s"]))
        review.extend(bool(r["needs_review"]) for r in rounds if r["video"] == video)
        vid = {
            "keyframes": len(tv["keyframes"]), "truth_objects": n,
            "side": dict(side), "side_IDF1": s_idf1, "side_switches": sw_side,
            "side_coverage": (side["idtp"] + side["wrong"]) / n if n else 0.0,
            "track_IDF1": t_idf1, "track_switches": sw_track, "tracks_used": len(n_pred_track),
            "HOTA_track": h_track, "HOTA_side": h_side,
            "round": {"pred_finish": pf, "pred_winner": pw, "true_finish": tr["finish"], "true_winner": tr["winner"],
                      "ambiguous_truth": bool(tr.get("ambiguous")), "needs_review": main_r["needs_review"] if main_r else None,
                      "review_reasons": len(main_r["review_reasons"]) if main_r else None,
                      "pred_stop_s": st, "true_stop_s": tr["loser_stop_s"]},
            "processing_s": work.get(video), "video_s": duration.get(video),
            "rtf": (work[video] / duration[video]) if work.get(video) and duration.get(video) else None,
        }
        out["videos"][video] = vid
        for k, v in side.items():
            tot[k] += v
        tot["n"] += n
        tot["sw_side"] += sw_side
        tot["sw_track"] += sw_track
        idf1_parts["idtp"] += t_idtp
        idf1_parts["gt"] += n
        idf1_parts["pred"] += sum(n_pred_track.values())
        all_frames_track += frames
    n = tot["n"]
    idfp = tot["wrong"] + tot["fp_side"]
    classes = sorted(set(finish_true))
    f1s = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(finish_true, finish_pred))
        fp = sum(t != c and p == c for t, p in zip(finish_true, finish_pred))
        fn = sum(t == c and p != c for t, p in zip(finish_true, finish_pred))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    decided = [w for w in winners if w is not None]
    out["overall"] = {
        "side_IDF1": 2 * tot["idtp"] / (2 * tot["idtp"] + idfp + n - tot["idtp"]) if n else 0.0,
        "side_correct": tot["idtp"], "side_wrong": tot["wrong"], "side_abstain": tot["abstain"], "missed": tot["miss"],
        "side_coverage": (tot["idtp"] + tot["wrong"]) / n if n else 0.0,
        "side_precision_when_assigned": tot["idtp"] / (tot["idtp"] + tot["wrong"]) if tot["idtp"] + tot["wrong"] else None,
        "ID_switches_side": tot["sw_side"], "ID_switches_track": tot["sw_track"],
        "track_IDF1_pooled": 2 * idf1_parts["idtp"] / (idf1_parts["gt"] + idf1_parts["pred"]) if idf1_parts["gt"] else 0.0,
        "HOTA_track_mean": float(np.mean([v["HOTA_track"]["HOTA"] for v in out["videos"].values()])),
        "HOTA_side_mean": float(np.mean([v["HOTA_side"]["HOTA"] for v in out["videos"].values()])),
        "finish_macro_F1": float(np.mean(f1s)) if f1s else None, "finish_classes": classes,
        "winner_decided": len(decided), "winner_correct": sum(decided), "winner_rounds": len(winners),
        "winner_accuracy_over_rounds": sum(decided) / len(winners) if winners else None,
        "stop_time_error_s": {"n": len(stop_err), "median": float(np.median(stop_err)) if stop_err else None},
        "human_review_rate": sum(review) / len(review) if review else None, "pipeline_rounds": len(review),
        "processing_s_total": sum(v["processing_s"] or 0 for v in out["videos"].values()),
        "rtf_mean": float(np.mean([v["rtf"] for v in out["videos"].values() if v["rtf"]])),
    }
    txt = json.dumps(out, indent=1, ensure_ascii=False)
    if a.json:
        Path(a.json).write_text(txt)
    print(json.dumps(out["overall"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
