"""Check local VLM actor-event proposals against recorded source observations.

This is an evidence study, not a replacement for core judgment. Every candidate
remains review-required; neither model text nor projected exits become truth.
"""

import argparse
import html
import json
import math
import os
import subprocess
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from bxjudge.autocalib import load_autocalib_config
from bxjudge.calibration import scale_calibration
from bxjudge.calibration_store import load_snapshot
from bxjudge.dynamic_calibration import (
    normalize_tracks,
    read_geometry,
    recorded_calibration,
)
from bxjudge.geometry import zone_of
from bxjudge.identity import anchor_sides
from bxjudge.launch_phase import prepare_launch_tracks
from bxjudge.rules import finish_points, load_rules
from bxjudge.types import Detection, Track
from bxjudge.video import probe
from bxjudge.vlm import VlmError, _opener, load_vlm_config, preflight
from bxjudge.vlm_events import corroborate_events, parse_events, request_events
from scripts.judge_local_outcomes import atomic, extract, indices, sha


def load_policy(path):
    policy = yaml.safe_load(Path(path).read_text())
    if (
        not isinstance(policy, dict)
        or set(policy)
        != {"overview_frames", "window_frames", "max_windows", "num_ctx", "think"}
        or any(
            type(policy[k]) is not int or policy[k] < 2
            for k in ("overview_frames", "window_frames", "num_ctx")
        )
        or type(policy["max_windows"]) is not int
        or policy["max_windows"] < 1
        or type(policy["think"]) is not bool
    ):
        raise ValueError("invalid actor-event policy")
    return policy


def reconstruct(results, row, video):
    source = Path(video["source_video"])
    source_sha = sha(source)
    record = read_geometry(results, row, source_sha)
    if source.resolve() != Path(record["source"]).resolve():
        raise ValueError("source differs from geometry evidence")
    fps = probe(source).fps
    snapshot = load_snapshot(
        results, row["calibration_id"], autocalib=load_autocalib_config()
    )
    calibration = scale_calibration(
        snapshot.calibration, tuple(record["reference_size"])
    )
    samples = {
        f: SimpleNamespace(
            status=s["status"],
            alignment=SimpleNamespace(**s["alignment"])
            if s["alignment"] is not None
            else None
        )
        for f, s in record["samples"].items()
    }
    native = [
        Track(
            t["track_id"],
            t["cls"],
            [
                Detection(d["frame"], d["cls"], tuple(d["box"]), d["score"])
                for d in t["dets"]
            ],
        )
        for t in record["two_bey_tracking"]["raw_source_tracks"]
    ]
    phase = prepare_launch_tracks(
        native,
        fps,
        record["phase_policy"],
        side_hints={
            int(k): v for k, v in record["identity"]["anchors"]["sides"].items()
        },
    )
    if json.loads(json.dumps(phase.record())) != record["phase"]:
        raise ValueError("saved phase evidence could not be exactly reproduced")
    eligible = {
        (d.frame, d.box, d.score)
        for t in phase.tracks
        if t.cls == "bey"
        for d in t.dets
    }
    anchors = anchor_sides(
        native,
        normalize_tracks(native, samples),
        calibration,
        required_frames=max(2, math.ceil(record["phase_policy"]["held_s"] * fps)),
        body_margin_px=record["phase_policy"]["body_margin_px"],
        fps=fps,
    )
    return {
        "source": source,
        "source_sha256": source_sha,
        "record": record,
        "fps": fps,
        "calibration": calibration,
        "samples": samples,
        "eligible": eligible,
        "anchors": anchors,
    }


def case_evidence(context, physical):
    ids = sorted(map(int, physical["members"]))
    eligible = context["eligible"]
    tracks = [
        Track(
            actor,
            "bey",
            [
                Detection(d["frame"], "bey", tuple(d["box"]), d["score"])
                for d in physical["observations"]
                if d["entity_id"] == i + 1
                and (d["frame"], tuple(d["box"]), d["score"]) in eligible
            ],
        )
        for i, actor in enumerate(ids)
    ]
    reference = normalize_tracks(tracks, context["samples"])
    sides = {}
    for actor in ids:
        known = {
            context["anchors"].sides[n]
            for n in physical["members"][str(actor)]
            if n in context["anchors"].sides
        }
        sides[actor] = next(iter(known)) if len(known) == 1 else None
    locators = []
    for track in reference:
        zones = {
            f: zone_of(point, context["calibration"])
            for f, point in track.centers().items()
        }
        for f in sorted(zones):
            if zones.get(f - 1) != "battle_area" or zones[f] not in (
                "over_left",
                "over_right",
                "xtreme",
                "lid",
                "outside",
                "stadium_other",
            ):
                continue
            if any(
                e["actor_id"] == track.track_id
                and e["to_zone"] == zones[f]
                and abs(f - e["frame_interval"][1]) <= context["fps"]
                for e in locators
            ):
                continue
            locators.append(
                {
                    "actor_id": track.track_id,
                    "to_zone": zones[f],
                    "frame_interval": [f - 1, f],
                }
            )
    return (
        tracks,
        reference,
        sides,
        sorted(locators, key=lambda e: e["frame_interval"][0]),
    )


def wanted_frames(physical, locators, fps, config, policy):
    lo, hi = physical["first_frame"], physical["last_frame"]
    # Cover the earliest and latest measured changes rather than taking only the
    # final scoring opening. Unsampled intermediate events remain a limitation.
    chosen = sorted(
        indices(0, len(locators) - 1, min(len(locators), policy["max_windows"]))
    )
    selected = [locators[i] for i in chosen]
    wanted = indices(lo, hi, policy["overview_frames"]) | {
        physical["pair_confirmed_at_frame"]
    }
    margin = round(config.event_window_s * fps)
    for event in selected:
        a, b = event["frame_interval"]
        wanted |= indices(
            max(lo, a - margin), min(hi, b + margin), policy["window_frames"]
        ) | {a, b}
    return wanted, selected


def annotate(folder, frames, tracks, context):
    images = []
    colors = [(200, 60, 220), (70, 200, 240)]
    for f in sorted(frames):
        path = folder / f"f{f}.png"
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("source image unreadable")
        height, width = image.shape[0] - 32, image.shape[1]
        calibration = recorded_calibration(
            context["calibration"], context["record"], f, (width, height)
        )
        if calibration is not None:
            for name in ("battle_area", "over_left", "over_right", "xtreme", "lid"):
                polygon = np.array(getattr(calibration, name), np.int32)
                polygon[:, 1] += 32
                cv2.polylines(image, [polygon], True, (80, 180, 80), 1)
                cv2.putText(
                    image,
                    name,
                    tuple(polygon[0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (80, 180, 80),
                    1,
                )
        else:
            cv2.putText(
                image,
                "geometry UNKNOWN",
                (width - 190, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (80, 150, 255),
                1,
            )
        sx, sy = (
            width / context["record"]["reference_size"][0],
            height / context["record"]["reference_size"][1],
        )
        for track, color in zip(tracks, colors):
            box = track.box_at(f)
            if box is None:
                continue
            x0, y0, x1, y1 = map(
                round, (box[0] * sx, box[1] * sy + 32, box[2] * sx, box[3] * sy + 32)
            )
            cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
            cv2.putText(
                image,
                f"ID{track.track_id}",
                (x0, max(48, y0 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        if not cv2.imwrite(str(path), image):
            raise ValueError("failed to save annotated source evidence")
        images.append(image)
    return images


def render(out, manifest, results):
    checks = [
        e for r in results for e in (r.get("corroboration") or {}).get("events", [])
    ]
    summary = {
        "planned_cases": len(manifest["cases"]),
        "completed_cases": len(results),
        "source_videos": len({c["video"] for c in manifest["cases"]}),
        "event_checks": dict(Counter(e["status"] for e in checks)),
        "conditional_sequences": dict(
            Counter(
                r["corroboration"]["conditional_sequence"]["if_events_are_confirmed"][
                    "decision"
                ]
                for r in results
                if r.get("corroboration") and r["corroboration"]["conditional_sequence"]
            )
        ),
        "response_failures": sum(r["error"] is not None for r in results),
        "automatic_accepted": 0,
        "referee_accuracy": None,
        "user_confirmed": False,
        "results": results,
    }
    atomic(out / "summary.json", summary)
    text = """<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>本地事件交叉核查</title><style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:1rem;background:#101820;color:#eee}a{color:#7dd3fc}section{border-top:1px solid #567;padding:1rem 0}img{max-width:48%;height:auto}pre{white-space:pre-wrap}</style>
<a href="../">← 批次結果</a><h1>本地事件交叉核查</h1>
<p>模型只回報物件 ID、事件與來源影格。程式核對實際框與當格校準，再用本地身分解讀條件式勝方。所有結果仍須覆核；中心進口與投影離盤不等於確認得分或物理離場。原核心、人工答案與舊模型基準保持獨立。</p>"""
    text += (
        "<p>"
        + html.escape(
            f"已處理 {len(results)}/{len(manifest['cases'])} 段，{summary['source_videos']} 支影片；事件 {summary['event_checks']}；回應失敗 {summary['response_failures']}；自動採納 0"
        )
        + "</p>"
    )
    text += f"<p>原配對人口 {len(manifest['paired_population'])} 段；其中 {len(manifest['unlocated_cases'])} 段沒有連續有效幾何窗口，仍保留原結果，未視為非對戰。</p>"
    for result in results:
        text += "<section><h2>" + html.escape(result["case"]) + "</h2>"
        if result["error"]:
            text += "<p>未取得合法事件回應：" + html.escape(result["error"]) + "</p>"
        else:
            sequence = result["corroboration"]["conditional_sequence"]
            if sequence:
                proposed = sequence["if_events_are_confirmed"]
                label = (
                    "若先離場事件與前置條件確認：重賽（0分）"
                    if proposed["decision"] == "redo"
                    else f"若事件與前置條件確認：{proposed['winner']}／{proposed['finish']}，仍待覆核"
                )
                text += "<p><strong>" + html.escape(label) + "</strong></p>"
            else:
                text += "<p>尚未形成可用的回合先後候選。</p>"
            for event in result["corroboration"]["events"]:
                text += (
                    "<pre>"
                    + html.escape(json.dumps(event, ensure_ascii=False, indent=2))
                    + "</pre>"
                )
            if not result["corroboration"]["events"]:
                text += "<p>模型未提出事件，沒有據此宣稱無事件。</p>"
            text += (
                "<p>模型不確定事項："
                + html.escape(str(result["facts"]["unknowns"]))
                + "</p>"
            )
        text += f'<a href="{result["case"]}/result.json">來源、版本、原始回應與條件</a><details><summary>實際來源影格（ID 框是追蹤假設）</summary>'
        for f in result["frames"]:
            path = f"{result['case']}/frames/f{f}.png"
            text += f'<a href="{path}"><img loading="lazy" src="{path}" alt="來源 f{f}"></a>'
        text += "</details></section>"
    pending = out / "index.html.tmp"
    pending.write_text(text + "</html>", encoding="utf-8")
    os.replace(pending, out / "index.html")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/vlm.yaml"))
    parser.add_argument("--policy", type=Path, default=Path("config/vlm_events.yaml"))
    args = parser.parse_args()
    results_dir = args.results.resolve()
    if args.out.exists() or args.out.resolve().is_relative_to(results_dir):
        raise ValueError("output must be a new folder outside frozen results")
    policy = load_policy(args.policy)
    cfg = replace(
        load_vlm_config(args.config), num_ctx=policy["num_ctx"], think=policy["think"]
    )
    if failure := preflight(cfg):
        raise VlmError(failure)
    with _opener().open(
        cfg.host + "/api/tags", timeout=cfg.preflight_timeout_s
    ) as response:
        [model] = [m for m in json.load(response)["models"] if m["name"] == cfg.model]
    rows = json.loads((results_dir / "rounds.json").read_bytes())
    videos = json.loads((results_dir / "videos.json").read_bytes())
    contexts, evidence, cases, paired, unlocated, protected = {}, {}, [], [], [], {}
    for row in rows:
        if row["geometry_evidence"] is None:
            continue
        path = row["geometry_evidence"]["path"]
        if path not in contexts:
            contexts[path] = reconstruct(results_dir, row, videos[row["video"]])
        context = contexts[path]
        physical = context["record"]["two_bey_tracking"]["rounds"][str(row["round"])]
        if physical["segment_status"] != "paired_observations":
            continue
        key = Path(row["video"]).stem + "-r" + str(row["round"])
        paired.append(key)
        tracks, reference, sides, locators = case_evidence(context, physical)
        if not locators:
            unlocated.append(key)
            continue
        wanted, selected = wanted_frames(
            physical, locators, context["fps"], cfg, policy
        )
        case = {
            "case": key,
            "video": row["video"],
            "round": row["round"],
            "source": str(context["source"]),
            "source_sha256": context["source_sha256"],
            "geometry_evidence": row["geometry_evidence"],
            "calibration_id": row["calibration_id"],
            "frames": sorted(wanted),
            "locators": locators,
            "selected_locators": selected,
            "local_side_evidence": sides,
            "anchor_evidence": context["anchors"].record(),
        }
        cases.append(case)
        evidence[key] = (context, tracks, reference)
    if not cases:
        raise ValueError("no source-supported event windows")
    # Seal every original result file, not just the cases that have locators.
    for path in results_dir.rglob("*"):
        if path.is_file():
            protected[str(path)] = sha(path)
    runtime = [
        "scripts/judge_actor_events.py",
        "scripts/judge_local_outcomes.py",
        "src/bxjudge/vlm_events.py",
        "src/bxjudge/vlm.py",
        "src/bxjudge/video.py",
        "src/bxjudge/identity.py",
        "config/identity.yaml",
        "src/bxjudge/launch_phase.py",
        "src/bxjudge/dynamic_calibration.py",
        "src/bxjudge/calibration_store.py",
        "src/bxjudge/calibration.py",
        "src/bxjudge/autocalib.py",
        "src/bxjudge/geometry.py",
        "src/bxjudge/rules.py",
        "src/bxjudge/config.py",
        "src/bxjudge/types.py",
        str(args.config),
        str(args.policy),
        "config/rules.yaml",
        "config/autocalib.yaml",
    ]
    manifest = {
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "runtime_sha256": {p: sha(p) for p in runtime},
        "original_results_sha256": protected,
        "model": model,
        "config": asdict(cfg),
        "policy": policy,
        "cases": cases,
        "paired_population": paired,
        "unlocated_cases": unlocated,
        "scope": "Development study of actual measured event windows; no independent accuracy estimate",
        "answers_supplied": False,
        "automatic_acceptance": False,
        "no_cloud": True,
    }
    args.out.mkdir(parents=True)
    atomic(args.out / "manifest.json", manifest)
    results = []
    render(args.out, manifest, results)
    began = time.perf_counter()
    rules = load_rules()
    for case in cases:
        context, tracks, reference = evidence[case["case"]]
        folder = args.out / case["case"]
        folder.mkdir()
        wanted = set(case["frames"])
        extract(
            context["source"],
            wanted,
            folder / "frames",
            cfg.image_long_edge,
            case["video"],
        )
        images = annotate(folder / "frames", wanted, tracks, context)
        result = {
            **case,
            "image_sha256": {
                f"f{f}.png": sha(folder / "frames" / f"f{f}.png") for f in wanted
            },
            "facts": None,
            "corroboration": None,
            "error": None,
            "needs_review": True,
            "user_confirmed": False,
        }
        started = time.perf_counter()
        try:
            response = request_events(images, wanted, {t.track_id for t in tracks}, cfg)
            atomic(folder / "response.json", response)
            facts = parse_events(response, wanted, {t.track_id for t in tracks})
            result["facts"] = facts
            checks = corroborate_events(
                facts,
                tracks,
                reference,
                context["calibration"],
                case["local_side_evidence"],
            )
            for event in checks["events"]:
                proposal = event["conditional_outcome"]
                event["conditional_points"] = (
                    finish_points(rules, proposal["finish"])
                    if proposal and proposal["finish"]
                    else 0
                )
                event["conditions"].append(
                    "all_prior_events_and_full_round_geometry_not_established"
                )
            result["corroboration"] = checks
        except (VlmError, ValueError) as error:
            result["error"] = f"{type(error).__name__}: {error}"
        result["request_wall_seconds"] = time.perf_counter() - started
        atomic(folder / "result.json", result)
        results.append(result)
        render(args.out, manifest, results)
        print(
            json.dumps(
                {
                    "completed": len(results),
                    "planned": len(cases),
                    "case": case["case"],
                    "error": result["error"],
                    "events": (result["corroboration"] or {}).get("events", []),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    sources = {str(c["source"]): c["source_sha256"] for c in contexts.values()}
    unchanged = all(
        sha(p) == digest
        for p, digest in (protected | manifest["runtime_sha256"] | sources).items()
    )
    if not unchanged:
        raise ValueError("source, original result or runtime changed during study")
    atomic(
        args.out / "completion.json",
        {
            "completed_cases": len(results),
            "wall_seconds": time.perf_counter() - began,
            "source_runtime_and_original_results_unchanged": True,
            "original_files_verified": len(protected),
            "automatic_accepted": 0,
            "referee_accuracy": None,
        },
    )


if __name__ == "__main__":
    main()
