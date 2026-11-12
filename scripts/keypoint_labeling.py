"""Stadium keypoint labeling: sample frames, prelabel them, create the Label Studio project.

Plan docs/superpowers/plans/2026-11-11-stadium-keypoints.md (K1). Read only on videos and
calib/. Usage:
    uv run python scripts/keypoint_labeling.py sample [--seed 42]
    uv run python scripts/keypoint_labeling.py create-project
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# create-project runs under Label Studio's own Python, where bxjudge is not installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bxjudge import PROJECT_ROOT  # noqa: E402

KEYPOINTS = ("notch_left", "notch_right", "xtreme_left", "xtreme_right", "over_left_inner",
             "over_right_inner", "field_left_upper", "field_left_lower", "field_right_upper",
             "field_right_lower")
FLIP_IDX = [1, 0, 3, 2, 5, 4, 8, 9, 6, 7]
LABEL_TEXT = {
    "notch_left": "1 貓耳缺口左上角", "notch_right": "2 貓耳缺口右上角",
    "xtreme_left": "3 Xtreme開口左下角", "xtreme_right": "4 Xtreme開口右下角",
    "over_left_inner": "5 左Over袋下內角", "over_right_inner": "6 右Over袋下內角",
    "field_left_upper": "7a 內場左斜邊上端", "field_left_lower": "7b 內場左斜邊下端",
    "field_right_upper": "8a 內場右斜邊上端", "field_right_lower": "8b 內場右斜邊下端",
}
COLORS = ["#E74C3C", "#C0392B", "#F1C40F", "#D4AC0D", "#3498DB", "#E67E22", "#1ABC9C",
          "#16A085", "#9B59B6", "#8E44AD"]
# Provisional template points (calib/20261014_202713, 1920 px), hand-placed for prelabels
# only; the reference points come from the user's labels on task 0 (plan K2).
PROVISIONAL_TEMPLATE_POINTS = {
    "notch_left": (933, 811), "notch_right": (1056, 811), "xtreme_left": (717, 289),
    "xtreme_right": (1152, 288), "over_left_inner": (402, 417), "over_right_inner": (1477, 408),
    "field_left_upper": (289, 732), "field_left_lower": (486, 927),
    "field_right_upper": (1608, 658), "field_right_lower": (1499, 915),
}
PACKET = PROJECT_ROOT / "data" / "detector-hard-cases-20261030"
OUT = PACKET / "stadium-keypoints-20261111"
VIDEO_ROOT = Path.home() / "bx" / "videos"
QUOTA = {"20261024": 45, "20261101": 35, "20261108": 40, "20261001": 28, "20261014": 2}
TEMPLATE = "20261014_202713"
EDGE_MARGIN_S = 0.5


@dataclass(frozen=True)
class VideoRef:
    path: Path
    batch: str
    duration_s: float


@dataclass(frozen=True)
class FrameRef:
    video: VideoRef
    time_s: float


def sample_keypoint_frames(videos: list[VideoRef], quota: dict[str, int], seed: int) -> list[FrameRef]:
    """Per batch: ``quota[batch]`` frames, one per video first and a second per video only when
    the batch has fewer videos than its quota (never more than 2 per video); times are uniform
    within the video, ``EDGE_MARGIN_S`` away from both ends. Deterministic for a seed."""
    rng = random.Random(seed)
    frames: list[FrameRef] = []
    for batch in sorted(quota):
        pool = sorted((v for v in videos if v.batch == batch), key=lambda v: str(v.path))
        n = min(quota[batch], 2 * len(pool))
        order = rng.sample(pool, len(pool))
        picks = order[:n] + order[:max(0, n - len(order))]
        for v in picks:
            lo, hi = EDGE_MARGIN_S, max(EDGE_MARGIN_S, v.duration_s - EDGE_MARGIN_S)
            frames.append(FrameRef(video=v, time_s=round(rng.uniform(lo, hi), 3)))
    return frames


def label_config() -> str:
    labels = "\n".join(f'    <Label value="{LABEL_TEXT[k]}" background="{c}"/>'
                       for k, c in zip(KEYPOINTS, COLORS))
    # No stadium box: a box over the whole stadium swallowed the clicks meant for points
    # (2026-11-11); the export derives the box from the labelled points (plan K3).
    return f"""<View>
  <Header value="先在下方選一個編號，再在圖上點那個位置；只點看得到的點，被擋住、在畫面外或看不清楚的不要點，也不要猜。預標可能偏 10–20 px，每一點都要拖到正確位置或刪掉。左右以場地為準：貓耳朝向鏡頭時的左邊。"/>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <KeyPointLabels name="kp" toName="image" smart="false">
{labels}
  </KeyPointLabels>
</View>"""


def prediction(points: dict[str, tuple[float, float]], size: tuple[int, int]) -> dict:
    """Label Studio prediction: percent coordinates; points outside the image are left out."""
    w, h = size
    result = []
    for name in KEYPOINTS:
        if name not in points:
            continue
        x, y = points[name]
        if not (0 <= x < w and 0 <= y < h):
            continue
        result.append({"from_name": "kp", "to_name": "image", "type": "keypointlabels",
                       "original_width": w, "original_height": h,
                       "value": {"x": 100 * x / w, "y": 100 * y / h, "width": 0.6,
                                 "keypointlabels": [LABEL_TEXT[name]]}})
    return {"model_version": "rail-fit-prelabel", "result": result}


def _rail_fit(frame: np.ndarray):
    """(homography model px -> frame px, model) from the rail-only finder, or None."""
    from bxjudge.stadium_finder import fit_stadium, load_stadium_finder_config, load_stadium_model, rail_mask

    cfg = load_stadium_finder_config()
    model = load_stadium_model()
    fh, fw = frame.shape[:2]
    k = 1280 / max(fw, fh)
    small = cv2.resize(frame, (round(fw * k), round(fh * k)), interpolation=cv2.INTER_AREA)
    fit = fit_stadium(rail_mask(small, cfg), model, cfg)
    if fit is None:
        return None
    h = np.diag([1 / k, 1 / k, 1.0]) @ np.asarray(fit.homography, np.float64)
    return h, model


def prelabel_points(frame: np.ndarray) -> dict[str, tuple[float, float]]:
    """Points projected from the provisional template points through the rail-only fit, in
    frame px; {} when the rail fit fails."""
    from bxjudge.stadium_finder import _project

    got = _rail_fit(frame)
    if got is None:
        return {}
    h, _ = got
    names = list(PROVISIONAL_TEMPLATE_POINTS)
    pts = _project(h, np.array([PROVISIONAL_TEMPLATE_POINTS[n] for n in names], np.float64))
    return {n: (float(x), float(y)) for n, (x, y) in zip(names, pts)}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _grab(video: Path, t: float) -> np.ndarray:
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t}", "-i", str(video), "-frames:v", "1",
                        "-f", "image2pipe", "-c:v", "png", "-"], capture_output=True)
    img = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR) if r.stdout else None
    if r.returncode != 0 or img is None:
        raise RuntimeError(f"cannot read {video} at {t}s")
    return img


def _videos() -> list[VideoRef]:
    from bxjudge.manifest import probe

    out = []
    for batch in sorted(QUOTA):
        for p in sorted((VIDEO_ROOT / batch).glob("*.mp4")):
            out.append(VideoRef(path=p, batch=batch, duration_s=float(probe(p).duration)))
    return out


def cmd_sample(seed: int) -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite a labeling packet")
    images = OUT / "images"
    images.mkdir(parents=True)
    frames = sample_keypoint_frames(_videos(), QUOTA, seed)
    template_png = PROJECT_ROOT / "calib" / f"{TEMPLATE}.png"
    entries = [("task0_template", None, None, cv2.imread(str(template_png)), "template")]
    for i, f in enumerate(frames, 1):
        entries.append((f"t{i:03d}_{f.video.path.stem}", f.video, f.time_s, _grab(f.video.path, f.time_s),
                        f.video.batch))
    tasks, manifest = [], []
    for stem, video, t, img, batch in entries:
        out = images / f"{stem}.jpg"
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        pts = prelabel_points(img)
        h, w = img.shape[:2]
        rel = out.relative_to(PACKET).as_posix()
        task = {"data": {"image": f"/data/local-files/?d={rel}", "image_path": str(out), "batch": batch,
                         "video": video.path.name if video else f"{TEMPLATE}.png",
                         "time_s": t, "is_template": video is None}}
        if pts:
            task["predictions"] = [prediction(pts, (w, h))]
        tasks.append(task)
        manifest.append({"image": out.name, "image_sha256": _sha256(out), "batch": batch,
                         "source": str(video.path if video else template_png),
                         "source_sha256": _sha256(video.path if video else template_png),
                         "time_s": t, "prelabel_points": len(pts)})
        print(f"{stem}: {len(pts)} prelabel points", flush=True)
    (OUT / "label_config.xml").write_text(label_config(), encoding="utf-8")
    (OUT / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps({"seed": seed, "quota": QUOTA, "frames": manifest},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(tasks)} tasks, {sum(1 for t in tasks if 'predictions' in t)} with prelabels -> {OUT}")
    return 0


PROJECT_TITLE = "BXJUDGE 場地關鍵點 150 張"
PROJECT_DESCRIPTION = (
    "每張先拉一個 stadium 框，再點看得到的 10 個關鍵點（編號見標籤）。看不到或不確定的點不要點。"
    "左右以場地為準：貓耳朝向鏡頭時的左邊。第一張 task0_template 是 10/14 範本，請 10 點全部標，"
    "它是換算區域的基準。預標來自只靠軌道的找場地結果，可能偏 10–20 px，不等於答案。")


def cmd_create_project() -> int:
    """Create the labeling project in the running local Label Studio (Django ORM as the
    existing project owner, as the 2026-10-31 imports did). Existing projects are checked
    unchanged afterwards and every image route must return the file's exact bytes."""
    import os

    os.environ.update({
        "DJANGO_SETTINGS_MODULE": "bxjudge.labeling.label_studio_settings",
        "LABEL_STUDIO_BASE_DATA_DIR": str(PACKET / "label-studio-state"),
        "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED": "true",
        "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT": str(PACKET),
        "COLLECT_ANALYTICS": "false", "LATEST_VERSION_CHECK": "false", "SENTRY_DSN": "",
        "FRONTEND_SENTRY_DSN": ""})
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import label_studio

    sys.path.insert(0, str(Path(label_studio.__file__).parent))
    import django

    django.setup()
    from django.conf import settings
    from django.contrib.auth import get_user_model
    from django.db import transaction
    from projects.models import Project
    from rest_framework.test import APIClient
    from tasks.models import Annotation, Task

    db = Path(settings.DATABASES["default"]["NAME"]).resolve()
    assert db == (PACKET / "label-studio-state" / "label_studio.sqlite3").resolve(), db
    tasks = json.loads((OUT / "tasks.json").read_text(encoding="utf-8"))
    if Project.objects.filter(title=PROJECT_TITLE).exists():
        raise SystemExit("project already exists; inspect it, do not import twice")
    before = {p.id: (Task.objects.filter(project=p).count(), Annotation.objects.filter(project=p).count())
              for p in Project.objects.all()}
    owner = get_user_model().objects.get(pk=Project.objects.order_by("id").first().created_by_id)
    client = APIClient(HTTP_HOST="127.0.0.1:8081")
    client.force_authenticate(user=owner)

    def post(path, data):
        r = client.post(path, data, format="json")
        assert r.status_code == 201, (path, r.status_code, r.data)
        return r.data

    with transaction.atomic():
        created = post("/api/projects/", {"title": PROJECT_TITLE, "description": PROJECT_DESCRIPTION,
                                          "label_config": (OUT / "label_config.xml").read_text(encoding="utf-8"),
                                          "show_collab_predictions": True,
                                          "evaluate_predictions_automatically": False})
        pid = created["id"]
        post("/api/storages/localfiles/", {"project": pid, "title": "場地關鍵點抽格（不Sync）",
                                           "path": str(OUT / "images"), "use_blob_urls": True})
        imported = post(f"/api/projects/{pid}/import?return_task_ids=true", tasks)
        assert imported["task_count"] == len(tasks), imported
    after = {p.id: (Task.objects.filter(project=p).count(), Annotation.objects.filter(project=p).count())
             for p in Project.objects.exclude(pk=pid)}
    assert after == before, "an existing project changed"
    for task in Task.objects.filter(project_id=pid):
        r = client.get(task.data["image"])
        raw = b"".join(r.streaming_content) if hasattr(r, "streaming_content") else r.content
        assert r.status_code == 200 and raw == Path(task.data["image_path"]).read_bytes(), task.id
    print(f"project {pid}: {len(tasks)} tasks, images verified -> http://127.0.0.1:8081/projects/{pid}/data")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--seed", type=int, default=42)
    sub.add_parser("create-project")
    a = p.parse_args(argv)
    if a.cmd == "sample":
        return cmd_sample(a.seed)
    return cmd_create_project()


if __name__ == "__main__":
    sys.exit(main())
