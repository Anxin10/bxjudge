"""Ask one local VLM for source-linked outcome proposals; keep core/human data intact."""

import argparse
import hashlib
import html
import json
import logging
import os
import subprocess
import time
from collections import Counter
from contextlib import closing
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import yaml

from bxjudge.config import load_thresholds
from bxjudge.video import iter_frames, probe
from bxjudge.vlm import VlmError, _opener, load_vlm_config, preflight
from bxjudge.vlm_judgment import parse_judgment, request_judgment


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path, data):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.replace(pending, path)


def indices(first, last, count):
    return {round(f) for f in np.linspace(first, last, count)}


def load_policy(path):
    policy = yaml.safe_load(path.read_text())
    if (
        not isinstance(policy, dict)
        or set(policy)
        != {
            "overview_frames",
            "launch_frames",
            "refinement_frames",
        }
        or any(type(v) is not int or v < 2 for v in policy.values())
    ):
        raise ValueError("invalid local judgment sampling policy")
    return policy


def render(out, manifest, results):
    counts = Counter(
        r["judgment"]["decision"] if r["judgment"] else "request_failed"
        for r in results
    )
    summary = {
        "planned_candidates": len(manifest["cases"]),
        "completed_candidates": len(results),
        "decisions": dict(counts),
        "automatic_accepted": 0,
        "user_confirmed": False,
        "reviewer_kind": "local_vlm",
        "measured_model": manifest["model"],
        "referee_accuracy": None,
        "results": results,
    }
    atomic(out / "summary.json", summary)
    text = """<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>本地模型勝負候選</title>
<style>body{font:16px system-ui;background:#101820;color:#eee;margin:2rem}a{color:#7dd3fc}section{border-top:1px solid #546575;padding:1rem 0}img{max-width:48%;height:auto}</style>
<a href="../">← 批次結果</a><h1>本地模型勝負候選</h1>
<p>這是本地 VLM 直接看原片抽格後的回答，與 Codex 判讀、原核心結果及使用者真值分開保存。
全部需要覆核，沒有自動採納；抽格之間仍可能漏掉更早事件，不能宣稱裁判準確率。</p>"""
    text += (
        "<p>"
        + html.escape(
            f"模型 {manifest['model']['name']}；已處理 {len(results)}/{len(manifest['cases'])}；{dict(counts)}"
        )
        + "</p>"
    )
    for result in results:
        judgment = result["judgment"]
        text += "<section><h2>" + html.escape(result["case"]) + "</h2>"
        if judgment:
            text += (
                "<p><strong>"
                + html.escape(
                    f"{judgment['decision']} · 勝方 {judgment['winner']} · {judgment['finish']} · 候選分數 {result['proposed_points']}"
                )
                + "</strong></p><p>"
                + html.escape(judgment["reason"])
                + "</p>"
            )
            text += (
                "<p>"
                + html.escape(
                    f"事件區間 {judgment['event_interval']}；發射有效性 {judgment['launch_validity']}；不確定 {judgment['uncertainties']}"
                )
                + "</p>"
            )
            if result["stage_disagreement"]:
                text += (
                    "<p>兩次取樣結論不同，需優先覆核；沒有把模型一致當作正確率。</p>"
                )
        else:
            text += "<p>未取得有效候選：" + html.escape(result["error"] or "") + "</p>"
        text += (
            '<p><a href="'
            + result["case"]
            + '/result.json">完整回答、版本與來源雜湊</a></p>'
        )
        for stage in result["stages"]:
            text += (
                "<details><summary>"
                + html.escape(
                    f"第 {stage['pass']} 次取樣；實際來源格 {stage['frames']}"
                )
                + "</summary>"
            )
            for frame in stage["frames"]:
                path = f"{result['case']}/frames/f{frame}.png"
                text += f'<a href="{path}"><img loading="lazy" src="{path}" alt="來源f{frame}"></a>'
            text += "</details>"
        text += "</section>"
    (out / "index.html.tmp").write_text(text + "</html>", encoding="utf-8")
    os.replace(out / "index.html.tmp", out / "index.html")


def extract(source, wanted, folder, long_edge, video):
    folder.mkdir(exist_ok=True)
    missing = {f for f in wanted if not (folder / f"f{f}.png").exists()}
    if missing:
        with closing(iter_frames(source, long_edge=long_edge, hwaccel="cpu")) as stream:
            for frame, image in stream:
                if frame in missing:
                    labeled = cv2.copyMakeBorder(
                        image, 32, 0, 0, 0, cv2.BORDER_CONSTANT
                    )
                    cv2.putText(
                        labeled,
                        f"{video} source f{frame}",
                        (8, 23),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    if not cv2.imwrite(str(folder / f"f{frame}.png"), labeled):
                        raise ValueError("failed to save source evidence frame")
                if frame >= max(missing):
                    break
    assert all((folder / f"f{f}.png").exists() for f in wanted), (
        "requested frame was not decoded"
    )
    return [cv2.imread(str(folder / f"f{f}.png")) for f in sorted(wanted)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/vlm.yaml"))
    parser.add_argument("--policy", type=Path, default=Path("config/vlm_judgment.yaml"))
    parser.add_argument(
        "--case",
        action="append",
        help="Exact VIDEO:ROUND; omitted means all paired candidates",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.out.exists() or args.out.resolve().is_relative_to(args.results.resolve()):
        raise ValueError("output must be a new folder outside the frozen results")
    cfg = load_vlm_config(args.config)
    policy = load_policy(args.policy)
    if failure := preflight(cfg):
        raise VlmError(failure)
    with _opener().open(
        cfg.host + "/api/tags", timeout=cfg.preflight_timeout_s
    ) as response:
        models = json.load(response)["models"]
    [model] = [m for m in models if m["name"] == cfg.model]
    results_dir = args.results.resolve()
    input_sha = {
        name: sha(results_dir / name) for name in ("rounds.json", "videos.json")
    }
    rows = json.loads((results_dir / "rounds.json").read_bytes())
    videos = json.loads((results_dir / "videos.json").read_bytes())
    cases, geometry_cache = [], {}
    selected = set(args.case) if args.case else None
    for row in rows:
        key = row["video"] + ":" + str(row["round"])
        if selected is not None and key not in selected:
            continue
        ref = row["geometry_evidence"]
        if not ref:
            continue
        path = results_dir / ref["path"]
        assert path.resolve().is_relative_to(results_dir / "geometry")
        assert sha(path) == ref["sha256"]
        if ref["path"] not in geometry_cache:
            geometry_cache[ref["path"]] = json.loads(path.read_bytes())
        geometry = geometry_cache[ref["path"]]
        physical = geometry["two_bey_tracking"]["rounds"].get(str(row["round"]))
        if physical is None or physical["segment_status"] != "paired_observations":
            continue
        source = Path(videos[row["video"]]["source_video"])
        assert (
            source.is_absolute()
            and source.resolve() == Path(geometry["source"]).resolve()
        )
        cases.append(
            {
                "video": row["video"],
                "round": row["round"],
                "key": key,
                "source": str(source),
                "source_sha256": geometry["source_sha256"],
                "geometry_evidence": ref,
                "physical": physical,
                "source_last_frame": max(s["frame"] for s in geometry["samples"]),
                "phase_releases": geometry["phase"]["releases"],
            }
        )
    if selected is not None and {c["key"] for c in cases} != selected:
        raise ValueError("an explicitly selected case is not a paired candidate")
    if not cases:
        raise ValueError("no paired candidates")
    code = [
        "src/bxjudge/vlm_judgment.py",
        "src/bxjudge/vlm.py",
        "src/bxjudge/video.py",
        "scripts/judge_local_outcomes.py",
        str(args.config),
        str(args.policy),
        "config/rules.yaml",
    ]
    manifest = {
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "runtime_sha256": {p: sha(p) for p in code},
        "model": model,
        "config": asdict(cfg),
        "sampling_policy": policy,
        "source_results": str(results_dir),
        "source_results_sha256": input_sha,
        "cases": cases,
        "sampling_population": "all saved paired candidates; no claim each is a real bout",
        "answers_supplied": False,
        "automatic_acceptance": False,
        "no_cloud": True,
    }
    args.out.mkdir(parents=True)
    atomic(args.out / "manifest.json", manifest)
    results = []
    render(args.out, manifest, results)
    th = load_thresholds()
    rules = yaml.safe_load(Path("config/rules.yaml").read_text())
    checked_sources = set()
    batch_began = time.perf_counter()
    for case in cases:
        name = Path(case["video"]).stem + "-r" + str(case["round"])
        folder = args.out / name
        folder.mkdir()
        source = Path(case["source"])
        if case["source_sha256"] not in checked_sources:
            assert sha(source) == case["source_sha256"]
            checked_sources.add(case["source_sha256"])
        info = probe(source)
        physical = case["physical"]
        lo, hi = physical["first_frame"], physical["last_frame"]
        members = {tid for ids in physical["members"].values() for tid in ids}
        releases = [
            f
            for tid, f in case["phase_releases"].items()
            if int(tid) in members and lo <= f <= hi
        ]
        center = min(releases) if releases else physical["pair_confirmed_at_frame"]
        wanted = indices(lo, hi, policy["overview_frames"]) | indices(
            max(lo, center - round(cfg.event_window_s * info.fps)),
            min(hi, center + round(3 * cfg.event_window_s * info.fps)),
            policy["launch_frames"],
        )
        if case["source_last_frame"] - hi <= round(th.round_gap_s * info.fps):
            wanted.add(case["source_last_frame"])
        overview = sorted(indices(lo, hi, policy["overview_frames"]))
        context = set(wanted) | {(a + b) // 2 for a, b in pairwise(overview)}
        result = {
            "case": name,
            "video": case["video"],
            "round": case["round"],
            "source_sha256": case["source_sha256"],
            "model": model,
            "reviewer_kind": "local_vlm",
            "user_confirmed": False,
            "needs_review": True,
            "core_or_human_results_modified": False,
            "stages": [],
            "judgment": None,
            "stage_disagreement": False,
            "proposed_points": 0,
            "error": None,
        }
        for stage in (1, 2):
            images = extract(
                source, wanted, folder / "frames", cfg.image_long_edge, case["video"]
            )
            start = time.perf_counter()
            record = {
                "pass": stage,
                "frames": sorted(wanted),
                "image_sha256": {
                    f"f{f}.png": sha(folder / "frames" / f"f{f}.png")
                    for f in sorted(wanted)
                },
            }
            try:
                response = request_judgment(
                    images, wanted, case["source_last_frame"], cfg
                )
                atomic(folder / f"response-{stage}.json", response)
                judgment = parse_judgment(response, wanted, case["source_last_frame"])
                record["judgment"] = judgment
            except (VlmError, ValueError) as error:
                record["error"] = f"{type(error).__name__}: {error}"
                result["error"] = record["error"]
                result["judgment"] = None
                record["wall_seconds"] = time.perf_counter() - start
                result["stages"].append(record)
                break
            record["wall_seconds"] = time.perf_counter() - start
            result["stages"].append(record)
            result["judgment"] = judgment
            windows = [
                judgment[k]
                for k in ("launch_interval", "event_interval")
                if judgment[k]
            ]
            wanted = context | set().union(
                *(indices(a, b, policy["refinement_frames"]) for a, b in windows)
            )
            if case["source_last_frame"] in record["frames"]:
                wanted.add(case["source_last_frame"])
        if len(result["stages"]) == 2 and all(
            "judgment" in s for s in result["stages"]
        ):
            keys = ("decision", "winner", "finish")
            result["stage_disagreement"] = any(
                result["stages"][0]["judgment"][k] != result["stages"][1]["judgment"][k]
                for k in keys
            )
        if result["judgment"] and result["judgment"]["decision"] == "score":
            result["proposed_points"] = rules["finish_points"][
                result["judgment"]["finish"]
            ]
        atomic(folder / "result.json", result)
        results.append(result)
        render(args.out, manifest, results)
        print(
            json.dumps(
                {
                    "case": name,
                    "completed": len(results),
                    "decision": result["judgment"],
                    "error": result["error"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    for name, digest in manifest["runtime_sha256"].items():
        assert sha(name) == digest, "runtime changed during VLM judging: " + name
    for case in cases:
        assert sha(case["source"]) == case["source_sha256"], (
            "source changed during VLM judging"
        )
    for filename, digest in input_sha.items():
        assert sha(results_dir / filename) == digest
    atomic(
        args.out / "completion.json",
        {
            "candidates": len(results),
            "batch_wall_seconds": time.perf_counter() - batch_began,
            "input_results_unchanged": True,
            "all_attempted": len(results) == len(cases),
            "automatic_accepted": 0,
        },
    )


if __name__ == "__main__":
    main()
