"""Different-object appearance controls from already confirmed simultaneous X/B."""

import argparse
import hashlib
import html
import json
import math
import shutil
import tempfile
import time
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np

from bxjudge.manifest import probe
from bxjudge.reid_appearance import describe_crop, distances, load_policy
from bxjudge.video import iter_frames


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def select_pairs(rows):
    counts = dict.fromkeys(
        (
            "reference_frames",
            "eligible_frames",
            "unconfirmed",
            "bad_detection",
            "non_unique_sides",
            "not_separated",
        ),
        0,
    )
    chosen, seen = [], set()
    for row in rows:
        key = (row["video"], row["frame"])
        if type(row["frame"]) is not int or row["frame"] < 0 or key in seen:
            raise ValueError("invalid or duplicate reference source frame")
        seen.add(key)
        counts["reference_frames"] += 1
        if (
            type(row["confirmed"]) is not bool
            or type(row["missed_or_bad_detection"]) is not bool
        ):
            raise ValueError("invalid human confirmation flags")
        if not row["confirmed"]:
            counts["unconfirmed"] += 1
            continue
        if row["missed_or_bad_detection"]:
            counts["bad_detection"] += 1
            continue
        sides = {
            side: [
                o
                for o in row["objects"]
                if o["detector_class"] == "bey" and o["side"] == side
            ]
            for side in ("X", "B")
        }
        if any(len(group) != 1 for group in sides.values()):
            counts["non_unique_sides"] += 1
            continue
        if any(group[0]["state"] != "separated" for group in sides.values()):
            counts["not_separated"] += 1
            continue
        a, b = sides["X"][0], sides["B"][0]
        for obj in (a, b):
            box = obj["box"]
            if (
                len(box) != 4
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in box)
                or box[0] >= box[2]
                or box[1] >= box[3]
            ):
                raise ValueError("invalid confirmed box")
        if a["index"] == b["index"] or a["box"] == b["box"]:
            raise ValueError("two distinct confirmed observations required")
        chosen.append(row)
        counts["eligible_frames"] += 1
    return sorted(chosen, key=lambda row: (row["video"], row["frame"])), counts


def summary(values):
    return (
        None
        if not values
        else {
            "min": min(values),
            "median": float(np.median(values)),
            "max": max(values),
            "count": len(values),
        }
    )


def page(report):
    esc = lambda value: html.escape(str(value), quote=True)
    rows = []
    for case in report["cases"]:
        cells = []
        for side in ("X", "B"):
            sample = case["samples"][side]
            cells.append(
                f'<figure><img src="{esc(sample["crop_file"])}" alt="人工{side}原始裁切"><figcaption>人工{side} · f{case["frame"]}</figcaption></figure>'
                if "crop_file" in sample
                else f"<p>{side}無可用裁切：{esc('、'.join(sample['reason_codes']))}</p>"
            )
        text = "無可用距離"
        if case["distances"]:
            text = "；".join(
                f"{key}={value:.3f}" if value is not None else f"{key}=無紋理"
                for key, value in case["distances"].items()
            )
        rows.append(
            f'<article><h2>{esc(case["video"])} · f{case["frame"]}</h2><p>同格人工X/B：不同顆；相關區間 {case["correlation_group"]}。</p><div class="pair">{"".join(cells)}</div><p>{esc(text)}</p></article>'
        )
    stats = []
    for name, label in [
        ("color_hellinger", "色彩"),
        ("gradient_hellinger", "梯度紋理"),
    ]:
        c = report["comparison"][name]

        def fmt(value):
            return (
                "未驗證" if value is None else f"{value['min']:.3f}–{value['max']:.3f}"
            )

        stats.append(
            f"<tr><td>{label}</td><td>{fmt(c['positive_case_medians'])}</td><td>{fmt(c['negative'])}</td><td>{'有重疊' if c['range_overlap'] else '未觀察到重疊' if c['range_overlap'] is False else '未驗證'}</td></tr>"
        )
    return f"""<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>不同顆外觀對照</title><style>body{{font:16px/1.7 system-ui;max-width:1120px;margin:auto;padding:24px;background:#f4f6fa;color:#192a40}}article,.notice{{background:#fff;padding:20px;border-radius:12px;margin:20px 0}}.pair{{display:flex;gap:24px;flex-wrap:wrap}}figure{{margin:0}}img{{width:180px;height:140px;object-fit:contain}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ccc}}a{{color:#195e9a}}</style><a href="/#identity">← 回工作台</a> · <a href="/reid-appearance/">四組同顆與誤偵案例</a><h1>從既有人工標記整理不同顆對照</h1><p class="notice">本頁沒有新增人工答案。只使用已確認、同一來源格同時為X/B且皆已分離的原框；未知、掛載與壞偵測不當負對照。共{report["negative_frame_pairs"]}個不同顆影格對、{report["correlation_groups"]}個相關區間、{report["source_videos"]}支影片；相鄰格與同片並非獨立驗收樣本。</p><h2>外觀距離能否區分？</h2><table><tr><th>描述子</th><th>同顆案例中位距離範圍（{report["positive_cases"]}組）</th><th>不同顆同格距離範圍</th><th>範圍對照</th></tr>{"".join(stats)}</table><p>同顆多格與不同顆同格是不同取樣條件，這張表只診斷混淆，不能直接挑門檻。範圍未重疊也不代表已可辨識；距離不是機率。沒有把人工答案當模型預測、沒有自動接續或新準確率。</p>{"".join(rows)}<p><a href="report.json">完整來源與證據JSON</a> · <a href="confirmed-reference.json">凍結的原人工修訂</a></p><p>裁切可見性仍未驗證，不把同一ID後續影格當可靠模板；f1618採最新人工修訂。未知及品質失敗保留人口。</p></html>"""


def build(pointer_path, identity_manifest_path, appearance_set, config_path, output):
    started = time.perf_counter()
    pointer_path, identity_manifest_path, appearance_set, config_path, output = (
        Path(p).resolve()
        for p in (
            pointer_path,
            identity_manifest_path,
            appearance_set,
            config_path,
            output,
        )
    )
    if output.exists():
        raise ValueError("output must be a new directory")
    policy = load_policy(config_path)
    module_root = Path(__file__).resolve().parents[1] / "src/bxjudge"
    frozen = {
        str(p): sha(p)
        for p in (
            pointer_path,
            identity_manifest_path,
            config_path,
            appearance_set / "report.json",
            Path(__file__).resolve(),
            *(
                module_root / name
                for name in ("reid_appearance.py", "video.py", "manifest.py")
            ),
        )
    }
    pointer = read(pointer_path)
    reference_path = Path(pointer["reference"]).resolve()
    if sha(reference_path) != pointer["sha256"]:
        raise ValueError("human reference SHA mismatch")
    frozen[str(reference_path)] = pointer["sha256"]
    reference = read(reference_path)
    selected, population = select_pairs(reference)
    manifest = read(identity_manifest_path)
    by_video = {c["video"]: c for c in manifest["cases"]}
    if len(by_video) != len(manifest["cases"]):
        raise ValueError("duplicate source video")
    appearance = read(appearance_set / "report.json")
    if (
        appearance["schema"] != "bxjudge.reid-appearance-diagnostics.v1"
        or appearance["applied"] is not False
        or appearance["policy"] != policy
    ):
        raise ValueError("appearance evidence policy or schema mismatch")
    for p, digest in appearance["input_sha256"].items():
        if sha(p) != digest:
            raise ValueError("positive evidence input SHA mismatch")
        frozen[p] = digest
    for name, digest in appearance["frozen_files_sha256"].items():
        if sha(appearance_set / name) != digest:
            raise ValueError("positive frozen copy SHA mismatch")
    sources, cases, requests = {}, [], {}
    group_id, prior = 0, None
    for r in selected:
        m = by_video[r["video"]]
        source = Path(m["source"]).resolve()
        if (
            source.name != r["video"]
            or r["source_sha256"] != m["source_sha256"]
            or r["coordinate_size"] != m["coordinate_size"]
            or r["fps"] != m["fps"]
        ):
            raise ValueError("human observation/source metadata mismatch")
        if str(source) not in sources:
            digest = sha(source)
            if digest != m["source_sha256"]:
                raise ValueError("source SHA mismatch")
            frozen[str(source)] = digest
            sources[str(source)] = probe(source)
        info = sources[str(source)]
        w, h = r["coordinate_size"]
        if (
            type(w) is not int
            or type(h) is not int
            or min(w, h) <= 0
            or not math.isclose(w / h, info.width / info.height, rel_tol=1e-5)
            or not math.isclose(info.fps, r["fps"], rel_tol=1e-7)
        ):
            raise ValueError("source coordinate size or FPS mismatch")
        matches = [f for f in m["frames"] if f["frame"] == r["frame"]]
        if len(matches) != 1 or matches[0]["image_sha256"] != r["image_sha256"]:
            raise ValueError("human frame not in original manifest")
        if (
            prior is None
            or prior["video"] != r["video"]
            or (r["frame"] - prior["frame"]) / r["fps"] > policy["window_s"]
        ):
            group_id += 1
        prior = r
        case = {
            "video": r["video"],
            "source_sha256": r["source_sha256"],
            "frame": r["frame"],
            "coordinate_size": r["coordinate_size"],
            "source_size": [info.width, info.height],
            "relation": "different",
            "relation_basis": "human_confirmed_simultaneous_distinct_X_B",
            "correlation_group": group_id,
            "samples": {},
            "distances": None,
            "applied": False,
        }
        cases.append(case)
        for side in ("X", "B"):
            obj = next(
                o
                for o in r["objects"]
                if o["detector_class"] == "bey" and o["side"] == side
            )
            exact = [o for o in matches[0]["objects"] if o["index"] == obj["index"]]
            if (
                len(exact) != 1
                or exact[0]["class"] != "bey"
                or any(exact[0][k] != obj[k] for k in ("box", "score"))
            ):
                raise ValueError("confirmed object is not an original observation")
            requests.setdefault(str(source), {}).setdefault(r["frame"], []).append(
                (case, side, obj)
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".reid-contrast-", dir=output.parent
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
                        raise ValueError("decoded size mismatch")
                    for case, side, obj in frames[frame]:
                        scale = [
                            info.width / case["coordinate_size"][0],
                            info.height / case["coordinate_size"][1],
                        ]
                        box = [v * scale[i % 2] for i, v in enumerate(obj["box"])]
                        s = describe_crop(image, box, policy)
                        s.update(
                            object_index=obj["index"],
                            frame=frame,
                            side=side,
                            native_box=obj["box"],
                            source_box=box,
                            identity_basis="this_confirmed_observation_only",
                            state="separated",
                        )
                        if s["descriptor"] is not None:
                            x1, y1, x2, y2 = s["crop_bounds"]
                            crop = image[y1:y2, x1:x2]
                            relative = f"{case['video']}-{frame}-{side}.png"
                            if not cv2.imwrite(str(stage / relative), crop):
                                raise OSError("crop write failed")
                            s.update(
                                crop_file=relative,
                                crop_sha256=sha(stage / relative),
                                pixel_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
                            )
                        case["samples"][side] = s
                    remaining.remove(frame)
                    if not remaining:
                        break
            if remaining:
                raise ValueError("requested source frames not decoded")
        for c in cases:
            if all(s["descriptor"] is not None for s in c["samples"].values()):
                c["distances"] = distances(c["samples"]["X"], c["samples"]["B"])
        positives = [
            c
            for c in appearance["cases"]
            if c["human_review"]["reviewed"] and c["human_review"]["relation"] == "same"
        ]
        comparison = {}
        for field in ("color_hellinger", "gradient_hellinger"):
            pos = [
                c["comparison"][field]["median"]
                for c in positives
                if c["comparison"] and c["comparison"][field]
            ]
            neg = [
                c["distances"][field]
                for c in cases
                if c["distances"] and c["distances"][field] is not None
            ]
            comparison[field] = {
                "positive_case_medians": summary(pos),
                "negative": summary(neg),
                "range_overlap": None
                if not pos or not neg
                else bool(max(min(pos), min(neg)) <= min(max(pos), max(neg))),
            }
        for p, digest in frozen.items():
            if sha(p) != digest:
                raise ValueError("input SHA changed during export")
        shutil.copyfile(reference_path, stage / "confirmed-reference.json")
        if sha(stage / "confirmed-reference.json") != pointer["sha256"]:
            raise ValueError("frozen reference copy SHA mismatch")
        report = {
            "schema": "bxjudge.reid-contrast.v1",
            "applied": False,
            "decision": "no_threshold_selected",
            "population": population,
            "negative_frame_pairs": len(cases),
            "usable_negative_pairs": sum(c["distances"] is not None for c in cases),
            "correlation_groups": group_id,
            "source_videos": len(sources),
            "positive_cases": len(positives),
            "positive_population": appearance["population"],
            "reference_revision": pointer["revision"],
            "reference_path": str(reference_path),
            "frozen_files_sha256": {"confirmed-reference.json": pointer["sha256"]},
            "input_sha256": frozen,
            "policy": policy,
            "cases": cases,
            "comparison": comparison,
            "limitations": [
                "same_frame_negatives_vs_multiframe_positives",
                "correlated_development_frames",
                "not_independent_acceptance",
                "visibility_unverified",
                "no_scope_or_reference_bank",
                "no_binding",
            ],
            "extraction_wall_s": time.perf_counter() - started,
        }
        (stage / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        (stage / "index.html").write_text(page(report))
        stage.rename(output)
    return report


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "reference-pointer",
        "identity-manifest",
        "appearance-set",
        "config",
        "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    report = build(
        a.reference_pointer, a.identity_manifest, a.appearance_set, a.config, a.out
    )
    print(
        json.dumps(
            {
                "population": report["population"],
                "comparison": report["comparison"],
                "offline_total_wall_s": time.perf_counter() - started,
            }
        )
    )


if __name__ == "__main__":
    main()
