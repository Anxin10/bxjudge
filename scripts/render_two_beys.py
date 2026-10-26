"""Playable source-frame-aligned two-top reconstruction; interpolation is dashed."""

import argparse
import html
import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np

from bxjudge.trajectory_video import load_video_policy
from bxjudge.video import iter_frames
from scripts.reconstruct_two_beys import read, sha


def draw_frame(image, points, launchers, trail, previous, frame, labels):
    colors = {1: (0, 159, 230), 2: (178, 114, 0)}
    for p in points:
        color = colors[p["entity_id"]]
        box = p["box"]
        point = (round((box[0] + box[2]) / 2), round((box[1] + box[3]) / 2))
        old = previous.get(p["entity_id"])
        estimated = p["kind"] == "interpolated" or (
            old is not None and old["kind"] == "interpolated"
        )
        if (
            old is not None
            and old["frame"] == frame - 1
            and (not estimated or frame % 2 == 0)
        ):
            cv2.line(trail, old["point"], point, color, 2)
        previous[p["entity_id"]] = {"point": point, "frame": frame, "kind": p["kind"]}
    mask = np.any(trail != 0, axis=2)
    canvas = image.copy()
    canvas[mask] = trail[mask]
    for box in launchers:
        cv2.rectangle(
            canvas,
            tuple(map(round, box[:2])),
            tuple(map(round, box[2:])),
            (70, 190, 70),
            1,
        )
    for p in points:
        box = p["box"]
        color = colors[p["entity_id"]]
        xy = tuple(map(round, box[:2]))
        if p["kind"] == "observed":
            cv2.rectangle(canvas, xy, tuple(map(round, box[2:])), color, 2)
        else:
            cv2.drawMarker(
                canvas,
                (round((box[0] + box[2]) / 2), round((box[1] + box[3]) / 2)),
                color,
                cv2.MARKER_CROSS,
                14,
                2,
            )
        text = labels.get(p["entity_id"], f"Top {p['entity_id']} (side unknown)") + (
            " measured" if p["kind"] == "observed" else " ESTIMATED"
        )
        cv2.putText(
            canvas,
            text,
            (max(0, xy[0]), max(35, xy[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (15, 15, 15), -1)
    cv2.putText(
        canvas,
        f"Source f{frame} | solid: observed, dashed/cross: interpolated | green: launcher",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
    )
    return canvas


def build(reconstruction_path, config_path, output):
    reconstruction_path, config_path, output = (
        Path(p).resolve() for p in (reconstruction_path, config_path, output)
    )
    if output.exists():
        raise ValueError("output must be a new directory")
    data = read(reconstruction_path)
    if (
        data["schema"] != "bxjudge.two-bey-reconstruction.v1"
        or data["applied_to_referee"] is not False
    ):
        raise ValueError("invalid two-bey reconstruction evidence")
    inputs = {str(p): digest for p, digest in data["input_sha256"].items()}
    for p in (reconstruction_path, config_path, Path(__file__).resolve()):
        inputs[str(p)] = sha(p)
    for p, digest in inputs.items():
        if sha(p) != digest:
            raise ValueError("source SHA changed before rendering")
    policy = load_video_policy(config_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".two-preview-", dir=output.parent) as temp:
        stage = Path(temp) / "site"
        stage.mkdir()
        previews = []
        for row in data["rounds"]:
            width, height = row["coordinate_size"]
            grouped = defaultdict(list)
            for point in row["points"]:
                grouped[point["frame"]].append(point)
            if any(
                len(ps) > 2 or len({p["entity_id"] for p in ps}) != len(ps)
                for ps in grouped.values()
            ):
                raise ValueError("more than two physical top positions at one frame")
            labels = {}
            for eid in (1, 2):
                sides = {
                    m["expected_side"]
                    for m in row["human_observation_matches"]
                    if m["entity_id"] == eid
                }
                if len(sides) == 1:
                    labels[eid] = next(iter(sides)) + f" (Top {eid})"
            if data["origin"] == "formal_saved_two_bey_geometry":
                launcher_tracks = row["native_launcher_tracks"]
            elif data["origin"] == "offline_native_replay":
                native_path = next(
                    Path(p)
                    for p in inputs
                    if Path(p).name == Path(row["video"]).stem + "-tracks.json"
                )
                launcher_tracks = [
                    t for t in read(native_path)["tracks"] if t["cls"] == "launcher"
                ]
            else:
                raise ValueError("unknown reconstruction source mode")
            bodies = defaultdict(list)
            for t in launcher_tracks:
                for d in t["dets"]:
                    bodies[d["frame"]].append(d["box"])
            name = Path(row["video"]).stem + f"-round-{row['round']}.mp4"
            path = stage / name
            command = [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(row["fps"]),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                policy["preset"],
                "-crf",
                str(policy["crf"]),
                "-threads",
                str(policy["encoder_threads"]),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ]
            trail = np.zeros((height, width, 3), np.uint8)
            previous = {}
            count = 0
            with tempfile.TemporaryFile() as err:
                proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=err)
                try:
                    with closing(
                        iter_frames(
                            Path(row["source"]),
                            start=0,
                            long_edge=max(width, height),
                            hwaccel="cpu",
                        )
                    ) as frames:
                        for f, image in frames:
                            if f < row["first_frame"]:
                                continue
                            if f > row["last_frame"]:
                                break
                            canvas = draw_frame(
                                image,
                                grouped.get(f, []),
                                bodies.get(f, []),
                                trail,
                                previous,
                                f,
                                labels,
                            )
                            proc.stdin.write(canvas.tobytes())
                            count += 1
                    proc.stdin.close()
                    code = proc.wait()
                    if code:
                        err.seek(0)
                        raise RuntimeError(
                            "two-bey encoding failed: "
                            + err.read().decode(errors="replace")[-500:]
                        )
                finally:
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait()
            if count != row["last_frame"] - row["first_frame"] + 1:
                raise ValueError("source frames missing during rendering")
            probe = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=nb_frames",
                        "-of",
                        "json",
                        str(path),
                    ]
                )
            )
            if int(probe["streams"][0]["nb_frames"]) != count:
                raise ValueError("encoded frame count mismatch")
            previews.append(
                {
                    "video": row["video"],
                    "round": row["round"],
                    "path": name,
                    "sha256": sha(path),
                    "frame_count": count,
                    "entity_count": row["observed_entity_count"],
                    "tentative_entity_count": row["tentative_entity_count"]
                    if data["origin"] == "formal_saved_two_bey_geometry"
                    else row["observed_entity_count"],
                    "interpolated_points": row["interpolated_frames"],
                    "unassigned_observations": len(row["unassigned"]),
                    "matched_human_observations": sum(
                        m["entity_id"] is not None
                        for m in row["human_observation_matches"]
                    ),
                    "total_human_observations": len(row["human_observation_matches"]),
                }
            )
        for p, digest in inputs.items():
            if sha(p) != digest:
                raise ValueError("source changed during rendering")
        shutil.copyfile(reconstruction_path, stage / "reconstruction.json")
        manifest = {
            "schema": "bxjudge.two-bey-preview.v1",
            "input_sha256": inputs,
            "reconstruction_sha256": sha(reconstruction_path),
            "previews": previews,
            "applied_to_referee": False,
            "origin": data["origin"],
            "formal_core_applied_two_bey_tracking": data["origin"]
            == "formal_saved_two_bey_geometry",
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
        cards = []
        unconfirmed_cards = []
        for r in previews:
            title = html.escape(r["video"]) + f" · 紀錄第{r['round']}段"
            note = (
                "兩個持續身分；橘／藍實線為實測，虛線或十字為插值。"
                if r["entity_count"] == 2
                else f"此段有{r['tentative_entity_count']}個暫定物件，未確認雙顆對戰；原框保留供覆核，不建立第三個正式身分。"
            )
            target_cards = cards if r["entity_count"] == 2 else unconfirmed_cards
            target_cards.append(
                f'<article><h2>{title}</h2><p>{note}</p><video controls preload="metadata" src="{r["path"]}"></video><p>觀測到{r["entity_count"]}個實體；補點{r["interpolated_points"]}筆；未指派原框{r["unassigned_observations"]}筆。人工原框匹配{r["matched_human_observations"]}/{r["total_human_observations"]}（開發回歸，非獨立準確率）。</p></article>'
            )
        (stage / "index.html").write_text(
            """<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>兩顆陀螺連續軌跡</title><style>body{font:16px/1.7 system-ui;max-width:1280px;margin:auto;padding:24px;background:#f4f6fa;color:#192a40}article{background:#fff;border-radius:12px;padding:20px;margin:24px 0}video{width:100%;background:#000}a{color:#195e9a}</style><a href="/#identity">← 回工作台</a><h1>每回合最多兩顆的連續軌跡重建</h1><p>兩個陀螺身分持續接線，發射器綠框另顯示。實測與缺格插值分開，插值不作裁判證據。多餘框／誤偵保留在JSON，沒有刪除原始資料。此為新離線重建預覽，尚未取代正式判決流程。</p>""".replace(
                "此為新離線重建預覽，尚未取代正式判決流程。",
                "影片直接使用這次正式流程保存的兩實體觀測，不重跑關聯；新結果仍需覆核，原人工判決保留。"
                if data["origin"] == "formal_saved_two_bey_geometry"
                else "此為離線原生觀測重播，不是正式判決結果。",
            )
            + "<p>X／B文字依已確認影格作對照；不把人工對照當作模型全片身分驗收。</p>"
            + "".join(cards)
            + (
                "<details><summary>另有只觀測到一顆的未確認片段（不列為已確認雙顆對戰）</summary>"
                + "".join(unconfirmed_cards)
                + "</details>"
                if unconfirmed_cards
                else ""
            )
            + '<p><a href="reconstruction.json">實測／插值／未指派及原始來源</a> · <a href="manifest.json">影片與來源雜湊</a></p></html>'
        )
        stage.rename(output)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("reconstruction", "config", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    m = build(a.reconstruction, a.config, a.out)
    print(
        json.dumps(
            {"previews": m["previews"], "output": str(a.out)}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
