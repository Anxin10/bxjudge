"""Measure immutable dynamic geometry by video, candidate and recorded template.

Does not decode videos, run a model or write core/human/calibration data. Recorded
state reasons are reported literally; historical 'lost' causes are not invented.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from bxjudge.autocalib import load_autocalib_config
from bxjudge.calibration_store import load_snapshot
from bxjudge.dynamic_calibration import read_geometry
from bxjudge.geometry_coverage import measure_window, measure_windows
from scripts.judge_local_outcomes import atomic, sha


def load_policy(path):
    value = yaml.safe_load(path.read_text())
    if (
        not isinstance(value, dict)
        or set(value) != {"target_coverage", "worst_clips", "failure_probes_per_clip"}
        or type(value["target_coverage"]) not in (int, float)
        or not math.isfinite(value["target_coverage"])
        or not 0 < value["target_coverage"] <= 1
        or any(
            type(value[k]) is not int or value[k] < 1
            for k in ("worst_clips", "failure_probes_per_clip")
        )
    ):
        raise ValueError("invalid geometry coverage policy")
    return value


def summarize(records):
    total = sum(r["total_frames"] for r in records)
    valid = sum(r["valid_frames"] for r in records)
    reasons = Counter()
    for r in records:
        reasons.update(r["missing_reasons"])
    return {
        "total_frames": total,
        "valid_frames": valid,
        "coverage": valid / total if total else None,
        "missing_reasons": dict(reasons),
    }


def coverage_report(results, policy):
    rows = json.loads((results / "rounds.json").read_bytes())
    videos = json.loads((results / "videos.json").read_bytes())
    refs = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["video"]].append(row)
    video_records, round_records = [], []
    for video, metadata in sorted(videos.items()):
        selected = grouped[video]
        geometries = {
            row["geometry_evidence"]["path"]
            for row in selected
            if row["geometry_evidence"]
        }
        if len(geometries) != 1 or any(
            not row["geometry_evidence"] for row in selected
        ):
            raise ValueError(
                f"{video}: coverage requires one archived dynamic geometry per video"
            )
        row = selected[0]
        # read_geometry validates exact source and calibration IDs plus the saved matrix structure.
        raw = json.loads((results / row["geometry_evidence"]["path"]).read_bytes())
        if Path(metadata["source_video"]).resolve() != Path(raw["source"]).resolve():
            raise ValueError("video source differs from recorded geometry")
        validated = read_geometry(results, row, raw["source_sha256"])
        for other in selected:
            if (
                other["geometry_evidence"] != row["geometry_evidence"]
                or other["calibration_id"] != row["calibration_id"]
            ):
                raise ValueError("mixed calibration or geometry inside one video")
        refs[str(results / row["geometry_evidence"]["path"])] = row[
            "geometry_evidence"
        ]["sha256"]
        snapshot = load_snapshot(
            results, row["calibration_id"], autocalib=load_autocalib_config()
        )
        parent = (
            load_snapshot(
                results, snapshot.template_id, autocalib=load_autocalib_config()
            )
            if snapshot.kind == "auto"
            else snapshot
        )
        for identifier in {snapshot.id, parent.id}:
            for suffix in (".json", ".png"):
                path = results / "calibrations" / (identifier + suffix)
                refs[str(path)] = sha(path)
        samples = list(validated["samples"].values())
        all_windows, paired_windows = [], []
        for row in selected:
            physical = raw["two_bey_tracking"]["rounds"][str(row["round"])]
            lo, hi = physical["first_frame"], physical["last_frame"]
            result = measure_window(samples, lo, hi)
            all_windows.append((lo, hi))
            if physical["segment_status"] == "paired_observations":
                paired_windows.append((lo, hi))
            round_records.append(
                {
                    "video": video,
                    "round": row["round"],
                    "scene": parent.calibration.name,
                    "calibration_id": snapshot.id,
                    "template_id": parent.id,
                    "first_frame": lo,
                    "last_frame": hi,
                    "segment_status": physical["segment_status"],
                    **result,
                }
            )
        video_records.append(
            {
                "video": video,
                "source": metadata["source_video"],
                "source_sha256": raw["source_sha256"],
                "scene": parent.calibration.name,
                "calibration_id": snapshot.id,
                "template_id": parent.id,
                "template_exists": True,
                "geometry_evidence": row["geometry_evidence"],
                "recorded_span": measure_window(
                    samples,
                    0,
                    max([s["frame"] for s in samples] + [hi for _, hi in all_windows]),
                ),
                "recorded_span_basis": "f0 through last observed geometry/candidate frame; source tail beyond that span is unmeasured",
                "candidate_union": measure_windows(samples, all_windows),
                "paired_union": measure_windows(samples, paired_windows),
            }
        )
    paired = [r for r in round_records if r["segment_status"] == "paired_observations"]
    scene_records = []
    for scene in sorted({v["scene"] for v in video_records}):
        members = [v for v in video_records if v["scene"] == scene]
        scene_records.append(
            {
                "scene": scene,
                "videos": len(members),
                "candidate_union": summarize([v["candidate_union"] for v in members]),
                "paired_union": summarize([v["paired_union"] for v in members]),
            }
        )
    worst = sorted(
        (v for v in video_records if v["paired_union"]["coverage"] is not None),
        key=lambda v: (
            v["paired_union"]["coverage"],
            -v["paired_union"]["total_frames"],
        ),
    )[: policy["worst_clips"]]
    return {
        "schema": "bxjudge.geometry-coverage.v1",
        "policy": policy,
        "scope": "Availability of archived geometry, not spatial correctness or referee accuracy; all windows inclusive source frames",
        "videos": video_records,
        "rounds": round_records,
        "scenes": scene_records,
        "video_count": len(video_records),
        "candidate_count": len(round_records),
        "paired_count": len(paired),
        "paired_above_target": sum(
            r["coverage"] >= policy["target_coverage"] for r in paired
        ),
        "paired_exactly_complete": sum(r["coverage"] == 1 for r in paired),
        "overall_candidate_union": summarize(
            [v["candidate_union"] for v in video_records]
        ),
        "overall_paired_union": summarize([v["paired_union"] for v in video_records]),
        "worst_clips": [
            {
                "video": v["video"],
                "scene": v["scene"],
                "coverage": v["paired_union"]["coverage"],
                "missing_intervals": sorted(
                    v["paired_union"]["missing_intervals"],
                    key=lambda interval: interval["frames"],
                    reverse=True,
                )[: policy["failure_probes_per_clip"]],
            }
            for v in worst
        ],
        "historical_cause_counts": {
            "out_of_bounds": None,
            "occluded_first_second": None,
            "template_mismatch": None,
            "reason": "Historical sample records saved lost/confirming, not rejection gate causes. These causes cannot be inferred from coverage alone.",
        },
        "input_sha256": {
            str(results / "rounds.json"): sha(results / "rounds.json"),
            str(results / "videos.json"): sha(results / "videos.json"),
            **refs,
        },
    }


def render_markdown(report):
    def pct(value):
        return "N/A" if value is None else f"{value:.2%}"

    lines = [
        "# 盤面幾何覆蓋率（封存來源實測）",
        "",
        f"{report['video_count']} 支影片／{report['candidate_count']} 列／{report['paired_count']} 段雙顆候選；雙顆區間聯集覆蓋率 {pct(report['overall_paired_union']['coverage'])}。",
        f"達到 {pct(report['policy']['target_coverage'])} 的雙顆候選 {report['paired_above_target']} 段；完整覆蓋 {report['paired_exactly_complete']} 段。這是可用性，不是幾何正確率或裁判準確率。",
        "",
        "分母使用來源影格的包含兩端區間；逐片合計用區間聯集避免重疊重算。準備／收拾也可能被切成候選，達標不代表是乾淨對戰。",
        "尚無人工確認的發射／終結邊界，這裡量保存的候選範圍，不能冒稱真實對戰起末；記錄來源範圍之後的影片尾端未量測。",
        "",
        "歷史 lost 沒有記錄拒絕的細項，無法據此判定遮擋、越界或範本不符；未知原因保留未知。",
        "",
        "## 逐場景",
        "",
        "| 已使用的人工範本場景 | 影片 | 雙顆區間覆蓋率 |",
        "|---|---:|---:|",
    ]
    lines += [
        f"| {r['scene']} | {r['videos']} | {pct(r['paired_union']['coverage'])} |"
        for r in report["scenes"]
    ]
    lines += [
        "",
        "## 逐影片",
        "",
        "| 影片 | 場景 | 記錄來源範圍 | 所有候選聯集 | 雙顆聯集 |",
        "|---|---|---:|---:|---:|",
    ]
    lines += [
        f"| {r['video']} | {r['scene']} | {pct(r['recorded_span']['coverage'])} | {pct(r['candidate_union']['coverage'])} | {pct(r['paired_union']['coverage'])} |"
        for r in report["videos"]
    ]
    lines += ["", "## 最差影片與缺失範圍", ""]
    for r in report["worst_clips"]:
        spans = ", ".join(
            f"f{i['first_frame']}–{i['last_frame']}（{i['frames']}格）"
            for i in r["missing_intervals"]
        )
        lines += [f"- {r['video']}：{pct(r['coverage'])}；{spans}"]
    lines += [
        "",
        "## 逐候選回合",
        "",
        "| 影片／來源候選 | 類型 | 起末來源格 | 有效／總格 | 覆蓋率 |",
        "|---|---|---|---:|---:|",
    ]
    lines += [
        f"| {r['video']} r{r['round']} | {r['segment_status']} | {r['first_frame']}–{r['last_frame']} | {r['valid_frames']}/{r['total_frames']} | {pct(r['coverage'])} |"
        for r in report["rounds"]
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config/geometry_coverage.yaml",
    )
    args = parser.parse_args()
    if args.out.exists() or args.out.resolve().is_relative_to(args.results.resolve()):
        raise ValueError("coverage output must be new and outside original results")
    report = coverage_report(args.results.resolve(), load_policy(args.policy))
    args.out.mkdir(parents=True)
    atomic(args.out / "coverage.json", report)
    (args.out / "coverage.md").write_text(render_markdown(report), encoding="utf-8")
    assert all(sha(p) == digest for p, digest in report["input_sha256"].items())
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "video_count",
                    "candidate_count",
                    "paired_count",
                    "paired_above_target",
                    "paired_exactly_complete",
                    "overall_paired_union",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
