"""Freeze raw multi-frame crop measurements; no matcher, model or referee writes."""

import argparse
import hashlib
import html
import itertools
import json
import math
import re
import shutil
import tempfile
import time
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np

from bxjudge.manifest import probe
from bxjudge.reid_appearance import (
    describe_crop,
    distances,
    load_policy,
    sample_observations,
)
from bxjudge.reid_review import MANIFEST_SCHEMA, validate_submission
from bxjudge.video import iter_frames


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def overlap(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / union if union > 0 else 0


def compare_samples(before, after):
    values = [
        {"before_frame": a["frame"], "after_frame": b["frame"], **distances(a, b)}
        for a, b in itertools.product(before, after)
        if a["descriptor"] is not None and b["descriptor"] is not None
    ]
    if not values:
        return None
    result = {"pair_count": len(values), "distances": values}
    for field in ("color_hellinger", "gradient_hellinger"):
        finite = [v[field] for v in values if v[field] is not None]
        result[field] = (
            None
            if not finite
            else {
                "min": min(finite),
                "median": float(np.median(finite)),
                "max": max(finite),
                "count": len(finite),
            }
        )
    return result


def page(report):
    esc = lambda value: html.escape(str(value), quote=True)
    cards = []
    for index, case in enumerate(report["cases"], 1):
        groups = []
        for name, label in [("before", "藍框①片段"), ("after", "橘框②片段")]:
            images = []
            for sample in case["samples"][name]:
                if "crop_file" in sample:
                    images.append(
                        f'<figure><a href="{esc(sample["crop_file"])}"><img src="{esc(sample["crop_file"])}" alt="f{sample["frame"]}原始裁切"></a><figcaption>f{sample["frame"]} · 可見性未驗證</figcaption></figure>'
                    )
            if name in case["endpoint_annotations"]:
                content = "此端點有人工作證為非陀螺，未擷取陀螺外觀。"
            elif images:
                content = "".join(images)
            else:
                reasons = sorted(
                    {
                        reason
                        for sample in case["samples"][name]
                        for reason in sample["reason_codes"]
                    }
                )
                content = "沒有可用裁切；品質原因：" + esc("、".join(reasons))
            groups.append(f'<h3>{label}</h3><div class="strip">{content}</div>')
        comparison = case["comparison"]
        metrics = "未計算陀螺配對距離。"
        if comparison:
            parts = []
            for key, label in [
                ("color_hellinger", "色彩"),
                ("gradient_hellinger", "梯度紋理"),
            ]:
                value = comparison[key]
                parts.append(
                    f"{label}距離中位數 {value['median']:.3f}（範圍 {value['min']:.3f}–{value['max']:.3f}）"
                    if value
                    else f"{label}無可用資料"
                )
            metrics = "；".join(parts)
        relation = {
            "same": "同一顆",
            "different": "不同顆",
            "uncertain": "無法判定",
            "unreviewed": "未核對",
        }[case["human_review"]["relation"]]
        cards.append(
            f"<article><h2>第{index}組 · {esc(case['video'])}</h2><p>人工：{relation}。{esc(case['human_review']['note'])}</p>{''.join(groups)}<p>{metrics}</p><p>原因：{esc('、'.join(case['reason_codes']))}</p></article>"
        )
    pop = report["population"]
    return f"""<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>多格外觀診斷</title><style>body{{font:16px/1.7 system-ui;background:#f4f6fa;color:#192a40;max-width:1120px;margin:auto;padding:24px}}article,.notice{{background:white;padding:20px;margin:20px 0;border-radius:12px}}.strip{{display:flex;gap:12px;overflow:auto}}figure{{margin:0;min-width:120px}}img{{max-width:160px;height:120px;object-fit:contain}}a{{color:#195e9a}}small{{overflow-wrap:anywhere}}</style><a href="/#identity">← 回工作台</a><h1>遮擋前後的多格外觀診斷</h1><p class="notice">共{pop["total"]}組：同一顆{pop["same"]}、不同顆{pop["different"]}、無法判定{pop["uncertain"]}、未審{pop["unreviewed"]}。距離小只表示這個描述子較相近，不代表同一顆；本頁不改軌跡、X／B或勝負。{"目前沒有不同顆對照，不能校準匹配門檻或計算準確率。" if not pop["different"] else "此為開發資料，不作獨立準確率；門檻尚未校準。"}</p><p>裁切取自未畫線的原片；周邊樣本尚未逐格人工確認，模糊／混入其他物件的可能性仍在。此為離線診斷，沒有自動接續或已驗證參考庫。</p>{"".join(cards)}<p><a href="report.json">完整證據 JSON</a> · <a href="human-review.json">凍結人工修訂</a> · <a href="endpoint-annotations.json">明確端點註記</a></p><small>人工修訂SHA {report["input_sha256"][report["labels_path"]]}；取樣與特徵wall time {report["extraction_wall_s"]:.2f}秒（不含最後報告序列化與發布，不是裁判核心速度）。</small></html>"""


def build(manifest_path, labels_path, annotations_path, config_path, output):
    started = time.perf_counter()
    manifest_path, labels_path, annotations_path, config_path, output = (
        Path(p).resolve()
        for p in (manifest_path, labels_path, annotations_path, config_path, output)
    )
    if output.exists():
        raise ValueError("output must be a new directory")
    module_root = Path(__file__).resolve().parents[1] / "src/bxjudge"
    inputs = {
        str(p): sha(p)
        for p in (
            manifest_path,
            labels_path,
            annotations_path,
            config_path,
            Path(__file__).resolve(),
            *(
                module_root / name
                for name in (
                    "reid_appearance.py",
                    "reid_review.py",
                    "video.py",
                    "manifest.py",
                )
            ),
        )
    }
    manifest, labels, annotations = (
        read(manifest_path),
        read(labels_path),
        read(annotations_path),
    )
    if manifest["schema"] != MANIFEST_SCHEMA:
        raise ValueError("invalid review manifest")
    validate_submission(labels, manifest, inputs[str(manifest_path)])
    if labels_path.name != f"reid-{inputs[str(labels_path)]}.json":
        raise ValueError("labels SHA filename mismatch")
    if (
        not isinstance(annotations, dict)
        or set(annotations) != {"labels_sha256", "endpoints"}
        or annotations["labels_sha256"] != inputs[str(labels_path)]
        or not isinstance(annotations["endpoints"], list)
    ):
        raise ValueError("endpoint annotation revision mismatch")
    by_case = {c["case_id"]: c for c in manifest["cases"]}
    if len(by_case) != len(manifest["cases"]):
        raise ValueError("duplicate review case")
    if any(
        not isinstance(key, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", key) is None
        for key in by_case
    ):
        raise ValueError("invalid case path identifier")
    excluded = {}
    for entry in annotations["endpoints"]:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"case_id", "endpoint", "object_class", "source"}
            or entry["case_id"] not in by_case
            or entry["endpoint"] not in ("before", "after")
            or entry["object_class"] != "launcher_tip"
            or entry["source"] != "user_explicit_correction"
        ):
            raise ValueError("invalid explicit endpoint annotation")
        key = (entry["case_id"], entry["endpoint"])
        if key in excluded:
            raise ValueError("duplicate endpoint annotation")
        excluded[key] = entry
    policy = load_policy(config_path)
    population = dict.fromkeys(
        ("total", "same", "different", "uncertain", "unreviewed"), 0
    )
    records, requests, sources = [], {}, {}
    for case, human in zip(manifest["cases"], labels["cases"], strict=True):
        population["total"] += 1
        population[human["relation"] if human["reviewed"] else "unreviewed"] += 1
        record = {
            "case_id": case["case_id"],
            "video": case["video"],
            "source_sha256": case["source_sha256"],
            "human_review": human,
            "samples": {"before": [], "after": []},
            "endpoint_annotations": {
                name: excluded[(case["case_id"], name)]
                for name in ("before", "after")
                if (case["case_id"], name) in excluded
            },
            "comparison": None,
            "scope_status": "local_pair_only_full_scope_unverified",
            "side_assignment": "unknown",
            "reference_version": None,
            "applied": False,
            "decision": "diagnostic_only",
            "reason_codes": [
                "scope_unverified",
                "reference_missing",
                "thresholds_uncalibrated",
                "visibility_unverified",
            ],
        }
        records.append(record)
        native_paths = [
            Path(p)
            for p in manifest["native_files_sha256"]
            if Path(p).name == Path(case["video"]).stem + "-tracks.json"
        ]
        if len(native_paths) != 1:
            raise ValueError("native cache not uniquely identified")
        cache_path = native_paths[0]
        if str(cache_path) not in inputs:
            inputs[str(cache_path)] = sha(cache_path)
            if (
                inputs[str(cache_path)]
                != manifest["native_files_sha256"][str(cache_path)]
            ):
                raise ValueError("native cache SHA mismatch")
        native = read(cache_path)
        source = Path(native["source"]).resolve()
        if str(source) not in sources:
            inputs[str(source)] = sha(source)
            if (
                inputs[str(source)] != native["source_sha256"]
                or inputs[str(source)] != manifest["source_files_sha256"][str(source)]
            ):
                raise ValueError("source SHA mismatch")
            info = probe(source)
            sources[str(source)] = info
        info = sources[str(source)]
        if (
            source.name != case["video"]
            or inputs[str(source)] != case["source_sha256"]
            or [info.width, info.height] != case["source_size"]
            or not math.isclose(info.fps, case["fps"], rel_tol=1e-7)
            or not math.isclose(info.fps, native["fps"], rel_tol=1e-7)
        ):
            raise ValueError("source/observation metadata mismatch")
        width, height = case["coordinate_size"]
        if min(width, height) <= 0 or not math.isclose(
            width / height, info.width / info.height, rel_tol=1e-5
        ):
            raise ValueError("coordinate aspect ratio mismatch")
        tracks = {t["track_id"]: t for t in native["tracks"]}
        if len(tracks) != len(native["tracks"]):
            raise ValueError("duplicate native track")
        for endpoint, tid in zip(("before", "after"), case["native_ids"], strict=True):
            chosen = sample_observations(
                tracks[tid], case[endpoint]["frame"], endpoint, info.fps, policy
            )
            exact = next(d for d in chosen if d["frame"] == case[endpoint]["frame"])
            if exact["box"] != case[endpoint]["native_box"]:
                raise ValueError("endpoint no longer matches actual native box")
            if (case["case_id"], endpoint) in excluded:
                record["reason_codes"].append("human_non_bey_endpoint")
                continue
            for d in chosen:
                other = [
                    {
                        "track_id": t["track_id"],
                        "class": o["cls"],
                        "iou": overlap(d["box"], o["box"]),
                    }
                    for t in native["tracks"]
                    if t["track_id"] != tid
                    for o in t["dets"]
                    if o["frame"] == d["frame"] and overlap(d["box"], o["box"]) > 0
                ]
                requests.setdefault(str(source), {}).setdefault(d["frame"], []).append(
                    (
                        record,
                        endpoint,
                        tid,
                        d,
                        other,
                        (info.width / width, info.height / height),
                    )
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".reid-appearance-", dir=output.parent
    ) as tmp:
        stage = Path(tmp) / "evidence"
        stage.mkdir()
        for source, frames in requests.items():
            info = sources[source]
            remaining = set(frames)
            with closing(
                iter_frames(
                    Path(source),
                    start=0,
                    long_edge=max(info.width, info.height),
                    hwaccel="cpu",
                )
            ) as decoded:
                for frame, image in decoded:
                    if frame not in frames:
                        continue
                    if image.shape[:2] != (info.height, info.width):
                        raise ValueError("decoded source coordinate mismatch")
                    for record, endpoint, tid, d, others, scale in frames[frame]:
                        box = [v * scale[i % 2] for i, v in enumerate(d["box"])]
                        sample = describe_crop(image, box, policy)
                        sample.update(
                            frame=frame,
                            native_id=tid,
                            native_box=d["box"],
                            source_box=box,
                            score=d["score"],
                            identity_verified=False,
                            overlapping_observations=others,
                        )
                        if others:
                            sample["reason_codes"].append("overlap_quality_unverified")
                        if sample["descriptor"] is not None:
                            x1, y1, x2, y2 = sample["crop_bounds"]
                            crop = image[y1:y2, x1:x2]
                            sample["pixel_sha256"] = hashlib.sha256(
                                crop.tobytes()
                            ).hexdigest()
                            relative = f"{record['case_id']}/{endpoint}-{frame}.png"
                            target = stage / relative
                            target.parent.mkdir(exist_ok=True)
                            if not cv2.imwrite(str(target), crop):
                                raise OSError("crop write failed")
                            sample.update(crop_file=relative, crop_sha256=sha(target))
                        record["samples"][endpoint].append(sample)
                    remaining.remove(frame)
                    if not remaining:
                        break
            if remaining:
                raise ValueError("missing requested source frames")
        for record in records:
            if "human_non_bey_endpoint" not in record["reason_codes"]:
                record["comparison"] = compare_samples(**record["samples"])
                if any(
                    sum(s["descriptor"] is not None for s in group) < 3
                    for group in record["samples"].values()
                ):
                    record["reason_codes"].append("insufficient_observations")
        for path, digest in inputs.items():
            if sha(path) != digest:
                raise ValueError("input SHA changed during export")
        report = {
            "schema": "bxjudge.reid-appearance-diagnostics.v1",
            "mode": "offline_diagnostic",
            "population": population,
            "cases": records,
            "input_sha256": inputs,
            "labels_path": str(labels_path),
            "annotations_path": str(annotations_path),
            "frozen_files_sha256": {
                "human-review.json": inputs[str(labels_path)],
                "endpoint-annotations.json": inputs[str(annotations_path)],
            },
            "policy": policy,
            "applied": False,
            "timing_basis": "source_frame_indices; nominal FPS sampling window, PTS mapping unverified",
            "limitations": [
                "no_matcher_or_binding",
                "no_verified_reference_bank",
                "visibility_unverified",
                "development_cases_not_independent_accuracy",
            ],
            "extraction_wall_s": 0,
            "runtime": {"opencv": cv2.__version__, "numpy": np.__version__},
        }
        if not population["different"]:
            report["limitations"].append("negative_pairs_missing")
        shutil.copyfile(labels_path, stage / "human-review.json")
        shutil.copyfile(annotations_path, stage / "endpoint-annotations.json")
        for name, digest in report["frozen_files_sha256"].items():
            if sha(stage / name) != digest:
                raise ValueError("frozen input copy SHA mismatch")
        report["extraction_wall_s"] = time.perf_counter() - started
        (stage / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        (stage / "index.html").write_text(page(report))
        stage.rename(output)
    return report


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "labels", "endpoint-annotations", "config", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = build(
        args.manifest, args.labels, args.endpoint_annotations, args.config, args.out
    )
    print(
        json.dumps(
            {
                "population": report["population"],
                "offline_total_wall_s": time.perf_counter() - started,
                "output": str(args.out),
            }
        )
    )


if __name__ == "__main__":
    main()
