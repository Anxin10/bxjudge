"""Shared test helpers for the whole project (later tasks add more here)."""
from __future__ import annotations

from bxjudge.types import (
    Box,
    Calibration,
    Detection,
    LaunchCheck,
    Point,
    Polygon,
    RoundResult,
    Track,
)


def rect(x1: float, y1: float, x2: float, y2: float) -> Polygon:
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def make_track(
    track_id: int,
    cls: str,
    centers: dict[int, Point],
    size: float = 20.0,
    score: float = 0.9,
) -> Track:
    half = size / 2.0
    dets = []
    for frame in sorted(centers):
        cx, cy = centers[frame]
        box: Box = (cx - half, cy - half, cx + half, cy + half)
        dets.append(Detection(frame=frame, cls=cls, box=box, score=score))
    return Track(track_id=track_id, cls=cls, dets=dets)


def make_calibration() -> Calibration:
    """640x360 rectangular stand-in calibration."""
    return Calibration(
        name="test",
        frame_size=(640, 360),
        stadium=rect(40, 20, 600, 340),
        lid=rect(60, 25, 580, 120),
        battle_area=rect(120, 60, 520, 280),
        over_left=rect(40, 200, 110, 300),
        over_right=rect(530, 200, 600, 300),
        xtreme=rect(280, 290, 360, 340),
        launch_left=rect(40, 0, 320, 120),
        launch_right=rect(320, 0, 600, 120),
        ref_frame="",
    )


def rr(
    finish,
    winner,
    touch=None,
    landed=None,
    air="no",
    *,
    video="v1.mp4",
    round=1,
    t_launch=0.0,
    review_reasons=None,
    launch=True,
) -> RoundResult:
    """Build a RoundResult. Unspecified touch/landed default to both sides "no"/True.

    ``touch`` / ``landed`` may be partial dicts (a missing side keeps the default).
    ``launch=False`` produces ``launch=None`` (no launch check performed).
    """
    t = {"X": "no", "B": "no", **(touch or {})}
    ld = {"X": True, "B": True, **(landed or {})}
    lc = (
        LaunchCheck(
            touch=t,
            landed=ld,
            landing_frame={"X": None, "B": None},
            landing_zone={"X": None, "B": None},
            air_collision=air,
            notes=[],
        )
        if launch
        else None
    )
    return RoundResult(
        video=video,
        round=round,
        t_launch=t_launch,
        t_end=t_launch + 10.0,
        launch=lc,
        finish=finish,
        winner=winner,
        points=0,
        confidence=1.0,
        needs_review=False,
        review_reasons=list(review_reasons or []),
    )


def make_ref_img(seed: int = 0, size: tuple[int, int] = (640, 360)):
    """Random BGR texture (width, height) = ``size``; stand-in for a real reference frame."""
    import numpy as np

    rng = np.random.default_rng(seed)
    w, h = size
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def cal_with(img, tmp_path, name: str = "test"):
    """``make_calibration()`` whose reference frame is ``img`` saved as PNG under ``tmp_path``."""
    import cv2

    c = make_calibration()
    c.name = name
    png = tmp_path / f"{name}.png"
    assert cv2.imwrite(str(png), img)
    c.frame_size = (img.shape[1], img.shape[0])
    c.ref_frame = str(png)
    return c


def make_smooth_img(seed: int = 0, size: tuple[int, int] = (640, 360)):
    """Spatially smooth, structured BGR image: blurred noise + gradient + filled shapes
    (closer to a real flat-floor/lid frame than white noise)."""
    import cv2
    import numpy as np

    rng = np.random.default_rng(seed)
    w, h = size
    base = cv2.GaussianBlur(rng.random((h, w), dtype=np.float32), (0, 0), 8) * 255 * 3
    base = np.clip(base - base.mean() + 110, 0, 255)
    base += np.linspace(0, 60, w, dtype=np.float32)[None, :]
    img = cv2.merge([np.clip(base, 0, 255).astype(np.uint8)] * 3)
    for _ in range(30):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(12, 40))
        cv2.circle(img, (x, y), r, tuple(int(v) for v in rng.integers(0, 256, 3)), -1)
    return img


def make_video(
    path,
    seconds: float = 2,
    fps: int = 60,
    audio: bool = False,
    size: tuple[int, int] = (320, 180),
):
    """Generate a tiny H.264 test video (ffmpeg ``testsrc``; ``sine`` audio if ``audio``)."""
    import subprocess
    from pathlib import Path

    path = Path(path)
    w, h = size
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
           "-i", f"testsrc=size={w}x{h}:rate={fps}:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd.append(str(path))
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def shift(img, dx: float, dy: float):
    """Translate img by (dx, dy) px (content moves right/down for positive values);
    the vacated border replicates the edge pixels."""
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def draw_blob(img, center: tuple[int, int], r: int, color: tuple[int, int, int] = (255, 255, 255)):
    """Draw a filled circle (a stand-in for a bey) onto img in place and return it."""
    import cv2

    cv2.circle(img, center, r, color, -1)
    return img


def parabola_then_orbit(
    land_at: int,
    land_point: Point = (300, 200),
    start: Point = (150, 30),
    orbit_radius: float = 40,
    n: int = 120,
) -> dict[int, Point]:
    """Centers for a top that falls, lands at frame ``land_at`` and then circles.

    Frames 0..land_at-1: x moves linearly from ``start`` to ``land_point``; y follows ``s ** p``
    (s = f / (land_at-1)), i.e. accelerating like a falling top. The exponent p is at least 1.5
    and is raised for long falls so the final fall steps always reach about 9 px/frame of vy
    (above the usual fall_speed_px of 6); vy exceeds 6 for roughly the second half of a ~30 frame
    fall. Frame land_at-1 is exactly ``land_point``.
    Frames land_at..n-1: slow circular motion on a circle of ``orbit_radius`` that passes through
    ``land_point`` (centre = land_point - (r, 0)); angular step 0.08 rad/frame, so speed is
    about 3.2 px/frame for r=40, well under the usual fall_speed_px. Truth landing frame = land_at.
    """
    import math

    assert land_at >= 2
    sx, sy = start
    lx, ly = land_point
    p = max(1.5, 9.0 * (land_at - 1) / max(ly - sy, 1.0))
    out: dict[int, Point] = {}
    for f in range(min(land_at, n)):
        s = f / (land_at - 1)
        out[f] = (sx + (lx - sx) * s, sy + (ly - sy) * s ** p)
    cx, cy = lx - orbit_radius, ly
    for f in range(land_at, n):
        a = 0.08 * (f - land_at)
        out[f] = (cx + orbit_radius * math.cos(a), cy + orbit_radius * math.sin(a))
    return out


def fall_then_rest(
    rest_at: int,
    rest_point: Point,
    start: Point,
    n: int = 120,
) -> dict[int, Point]:
    """Centers for a top that falls fast in a straight line and then stays motionless.

    Frames 0..rest_at-1: linear from ``start`` to ``rest_point`` (frame rest_at-1 is exactly
    ``rest_point``; the caller picks start/rest_at so the per-frame vy is above fall_speed_px).
    Frames rest_at..n-1: motionless at ``rest_point``. Truth landing frame = rest_at.
    """
    assert rest_at >= 2
    sx, sy = start
    rx, ry = rest_point
    out: dict[int, Point] = {}
    for f in range(min(rest_at, n)):
        s = f / (rest_at - 1)
        out[f] = (sx + (rx - sx) * s, sy + (ry - sy) * s)
    for f in range(rest_at, n):
        out[f] = (rx, ry)
    return out


def circle_path(center=(320, 170), r=60, deg_per_frame=4, n=120):
    """Screen-coordinate orbit; positive angular steps are clockwise."""
    import math

    return {f: (center[0] + r * math.cos(math.radians(f * deg_per_frame)),
                center[1] + r * math.sin(math.radians(f * deg_per_frame)))
            for f in range(n)}


def rotating_crops(deg_per_frame=5, n=10):
    """Fixed 64x64 random texture, rotated clockwise by positive screen angles."""
    import cv2
    import numpy as np

    texture = np.random.default_rng(42).integers(0, 256, (64, 64), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), 0)
    return {f: cv2.warpAffine(
        texture, cv2.getRotationMatrix2D((31.5, 31.5), -f * deg_per_frame, 1.0),
        (64, 64), borderMode=cv2.BORDER_REFLECT)
        for f in range(n)}


def slowing_bey(reverse_at=None, aliasing_flips_at=(), still_at=None, n=320):
    """Known right baseline is screen-CCW under the default mapping."""
    centers = {f: (250, 180) for f in range(n)}
    activity = {f: 10.0 for f in range(n)}
    rate = {f: -40.0 if f < 180 else -5.0 for f in range(n)}
    for f in aliasing_flips_at:
        rate[f] = 40.0
    if reverse_at is not None:
        rate.update({f: 5.0 for f in range(reverse_at, n)})
    if still_at is not None:
        activity.update({f: 0.0 for f in range(still_at, n)})
        rate.update({f: 0.0 for f in range(still_at, n)})
    return make_track(1, 'bey', centers), activity, rate


def scene(*, b_path_into=None, x_path_into=None, b_enter_at=100, x_enter_at=100,
          b_burst_at=None, stops=None, overlap_frames=0, n=340):
    """Separated complete tracks, with explicit stable whole-box pocket entries."""
    targets = {'xtreme': (320, 320), 'over_left': (75, 250), 'over_right': (565, 250)}
    x = {f: (180, 180) for f in range(n)}
    b = {f: (450, 180) for f in range(n)}
    for centers, zone, at in ((x, x_path_into, x_enter_at), (b, b_path_into, b_enter_at)):
        if zone:
            centers.update({f: targets[zone] for f in range(at, n)})
    for f in range(20, 20 + overlap_frames):
        x[f] = b[f] = (300, 180)
    parts = []
    if b_burst_at is not None:
        b = {f: p for f, p in b.items() if f < b_burst_at}
        parts = [make_track(3, 'bey_part', {b_burst_at: (455, 185)})]
    return ({'X': make_track(1, 'bey', x), 'B': make_track(2, 'bey', b)},
            parts, stops or {'X': None, 'B': None}, make_calibration())


def rev(
    *,
    finish="spin",
    winner="X",
    fix=None,
    needs_review=False,
    reviewed=True,
    video="v1.mp4",
    round=1,
    t_launch=0.0,
    x_touch="no",
    b_touch="no",
    air_collision="no",
    x_spin_dir="right",
    b_spin_dir="right",
    review_reasons=None,
    confidence=0.9,
    motion_evidence=None,
    calibration_id=None,
) -> dict:
    """A flat ``rounds.json`` row (Task 13 schema). Defaults: a reviewed, unflagged,
    confident spin win for X; ``fix`` becomes ``reviewer_fix`` (the human truth).
    ``calibration_id`` (the snapshot the round was judged with) is added only when given."""
    return {
        **({"calibration_id": calibration_id} if calibration_id is not None else {}),
        "video": video,
        "round": round,
        "t_launch": t_launch,
        "t_end": t_launch + 10.0,
        "x_touch": x_touch,
        "b_touch": b_touch,
        "x_landed": True,
        "b_landed": True,
        "x_land_zone": "battle_area",
        "b_land_zone": "battle_area",
        "x_spin_dir": x_spin_dir,
        "b_spin_dir": b_spin_dir,
        "air_collision": air_collision,
        "finish": finish,
        "winner": winner,
        "points": {"spin": 1, "burst": 2, "over": 2, "xtreme": 3}.get(finish, 0),
        "confidence": confidence,
        "needs_review": needs_review,
        "review_reasons": list(review_reasons or []),
        "vlm_note": None,
        "reviewed": reviewed,
        "reviewer_fix": fix,
        "launch_present": True,
        "spin_dir_present": True,
        "landing_frame": {"X": None, "B": None},
        "launch_notes": [],
        "motion_evidence": motion_evidence,
        "detector_provenance": None,
    }


def row(**kw) -> dict:
    """An UNREVIEWED flat row (``rev`` with ``reviewed=False``)."""
    return rev(**{"reviewed": False, **kw})


def placeholder(video: str, reason: str) -> dict:
    """A diagnostic non-round row (``recalibrate``, ``decode_error`` ...) for ``video``."""
    return rev(video=video, finish=None, winner=None, needs_review=True, reviewed=False,
               review_reasons=[reason])


def videos_with_calibration(ids: dict, statuses: dict | bool = True) -> dict:
    """``videos.json`` content: ``{"a": "S1", "c": None}`` -> entry ``a.mp4`` judged with snapshot
    ``S1`` (auto) and ``c.mp4`` without a snapshot (``recalibrate``).

    ``analysis_status`` (R68) is ``ok`` for a snapshot and ``recalibrate`` without one, unless
    ``statuses`` (``{"a": "decode_error"}``) overrides it; ``statuses=False`` omits it, as in a
    results folder written before R68."""
    out = {}
    for name, sid in ids.items():
        entry = {
            "duration_s": 10.0,
            "processing_s": 5.0,
            "calibration": None if sid is None else {
                "id": sid, "kind": "auto", "name": "auto-test", "score": 0.9,
                "template_id": "f" * 16},
        }
        if statuses is not False:
            default = "recalibrate" if sid is None else "ok"
            overrides = statuses if isinstance(statuses, dict) else {}
            entry["analysis_status"] = overrides.get(name, default)
        out[f"{name}.mp4"] = entry
    return out


def _fixed_auto_snapshot(out_dir, seed: int):
    """A deterministic auto snapshot (same content, same id) written into ``out_dir``."""
    from bxjudge.calibration_store import save_snapshot

    eye = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return save_snapshot(out_dir, "auto", make_calibration(), make_ref_img(seed), score=0.9,
                         template_id="f" * 16, homography=eye)


def auto_snapshot_ids() -> tuple[str, str]:
    """Ids (S1, S2) of the two fixed auto snapshots used by ``results_with_two_positions``."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        return _fixed_auto_snapshot(Path(d), 1).id, _fixed_auto_snapshot(Path(d), 2).id


def results_with_two_positions(tmp_path):
    """A results folder from one batch with two camera positions: ``a.mp4`` judged with auto
    snapshot S1 and ``b.mp4`` with S2 (``auto_snapshot_ids``), two unreviewed rounds each."""
    from dataclasses import replace

    from bxjudge.output import (
        record_video_calibration,
        record_video_status,
        record_video_timing,
        write_results,
    )
    from bxjudge.rules import load_rules

    out = tmp_path / "out"
    rounds = []
    for seed, video in ((1, "a.mp4"), (2, "b.mp4")):
        snap = _fixed_auto_snapshot(out, seed)
        rounds += [replace(rr("spin", "X", video=video, round=n, t_launch=10.0 * n),
                           calibration_id=snap.id) for n in (1, 2)]
        record_video_timing(out, video, 30.0, 10.0)
        record_video_calibration(out, video, snap)
        record_video_status(out, video, "ok")
    write_results(rounds, out, load_rules())
    return out


def read_matches(out_dir) -> list[dict]:
    """``matches.csv`` rows with their ``rounds`` JSON decoded."""
    import csv
    import json
    from pathlib import Path

    with (Path(out_dir) / "matches.csv").open(encoding="utf-8", newline="") as f:
        return [{**m, "rounds": json.loads(m["rounds"])} for m in csv.DictReader(f)]


def _ellipse_poly(cx: float, cy: float, a: float, b: float, n: int = 72) -> Polygon:
    import math

    return [(cx + a * math.cos(2 * math.pi * k / n), cy + b * math.sin(2 * math.pi * k / n))
            for k in range(n)]


def _resample_closed(pts: Polygon, step: float) -> list[Point]:
    """Points every ``step`` px of arc length along the closed polygon ``pts``."""
    import math

    out: list[Point] = []
    carry = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        seg = math.hypot(x1 - x0, y1 - y0)
        t = carry
        while t < seg:
            out.append((x0 + (x1 - x0) * t / seg, y0 + (y1 - y0) * t / seg))
            t += step
        carry = t - seg
    return out


def stadium_scene(
    size: tuple[int, int] = (1280, 720),
    floor: str = "white",
    shift: tuple[float, float] = (0, 0),
    rotation_deg: float = 0.0,
    scale: float = 1.0,
):
    """Synthetic BGR stadium frame (``size`` = (width, height)) and its exact 8-zone calibration.

    The canonical layout is defined at 1280x720: an octagonal stadium body (outer rim), a
    transparent lid drawn only as a thin ellipse outline, an elliptical battle floor ringed by
    a toothed rail, and three dark pockets (over_left, over_right, xtreme) in the rim. Every
    outline is a dashed band so edges are textured along their length. The whole scene is
    rotated by ``rotation_deg`` (counter-clockwise on screen) and scaled by ``scale`` about the
    image centre, then translated by ``shift`` output pixels; the returned polygons are the
    same transform of the canonical zones. ``floor="black"`` inverts the floor brightness.
    Deterministic: the sensor noise uses a fixed seed and does not move with the scene."""
    import cv2
    import numpy as np

    if floor not in ("white", "black"):
        raise ValueError(f"floor must be 'white' or 'black', got {floor!r}")
    w, h = size
    m = cv2.getRotationMatrix2D((640.0, 360.0), rotation_deg, scale)
    sx, sy = w / 1280.0, h / 720.0

    def tf(pts: Polygon) -> Polygon:
        out = []
        for x, y in pts:
            u = m[0, 0] * x + m[0, 1] * y + m[0, 2]
            v = m[1, 0] * x + m[1, 1] * y + m[1, 2]
            out.append((float(u * sx + shift[0]), float(v * sy + shift[1])))
        return out

    def fixed(pts: Polygon) -> np.ndarray:  # 4 fractional bits for cv2's shift argument
        return np.round(np.array(tf(pts)) * 16).astype(np.int32).reshape(-1, 1, 2)

    stadium = [(270, 70), (1010, 70), (1150, 210), (1150, 560), (1010, 700), (270, 700),
               (130, 560), (130, 210)]
    lid = _ellipse_poly(640, 330, 470, 240)
    battle = _ellipse_poly(640, 440, 420, 210)
    over_left = [(160, 240), (280, 205), (300, 300), (185, 345)]
    over_right = [(1280 - x, y) for x, y in over_left]
    xtreme = [(560, 95), (720, 95), (700, 160), (580, 160)]
    launch_left = rect(130, 70, 660, 330)
    launch_right = rect(620, 70, 1150, 330)

    img = np.full((h, w, 3), (40, 42, 45), np.uint8)
    floor_bgr = (228, 228, 228) if floor == "white" else (27, 27, 27)
    cv2.fillPoly(img, [fixed(stadium)], (150, 150, 150), cv2.LINE_AA, 4)
    for pocket in (over_left, over_right, xtreme):
        cv2.fillPoly(img, [fixed(pocket)], (70, 70, 75), cv2.LINE_AA, 4)
    cv2.fillPoly(img, [fixed(battle)], floor_bgr, cv2.LINE_AA, 4)

    def dashed(pts: Polygon, colors, thickness: float, step: float) -> None:
        q = fixed(_resample_closed(pts, step)).reshape(-1, 2)
        t = max(1, round(thickness * scale * sx))
        for i in range(len(q)):
            a, b = q[i], q[(i + 1) % len(q)]
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     colors[i % len(colors)], t, cv2.LINE_AA, 4)

    dashed(stadium, [(210, 210, 210), (90, 90, 90)], 6, 14)
    for pocket in (over_left, over_right, xtreme):
        dashed(pocket, [(200, 200, 200), (30, 30, 30)], 4, 10)
    dashed(battle, [(60, 190, 90), (40, 120, 60)], 12, 10)
    dashed(lid, [(235, 235, 235), (170, 170, 170)], 3, 20)

    noise = np.random.default_rng(0).normal(0.0, 2.0, (h, w, 1))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    calib = Calibration(
        name="stadium_scene", frame_size=(w, h), stadium=tf(stadium), lid=tf(lid),
        battle_area=tf(battle), over_left=tf(over_left), over_right=tf(over_right),
        xtreme=tf(xtreme), launch_left=tf(launch_left), launch_right=tf(launch_right),
        ref_frame="",
    )
    return img, calib


def write_scene_video(path, img, *, seconds: float = 3.0, fps: int = 30, shake=None):
    """Encode the static BGR frame ``img`` as a near-lossless H.264 clip of ``seconds`` at ``fps``
    (frame size = the image size). ``shake`` maps frame index -> (dx, dy): that frame shows the
    whole scene translated (a stadium shake as the camera sees it)."""
    import subprocess
    from pathlib import Path

    import numpy as np

    path = Path(path)
    h, w = img.shape[:2]
    n = round(seconds * fps)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset",
           "ultrafast", "-crf", "10", "-pix_fmt", "yuv420p", str(path)]
    still = np.ascontiguousarray(img, dtype=np.uint8).tobytes()
    shake = shake or {}
    data = b"".join(
        np.ascontiguousarray(shift(img, *shake[f])).tobytes() if f in shake
        else still
        for f in range(n)
    )
    subprocess.run(cmd, input=data, check=True, capture_output=True)
    return path


def add_manual_calibration(calib_dir, scene, name: str = "scene"):
    """Save ``scene`` = (image, calibration) as manual calibration ``<calib_dir>/<name>.json``
    with its reference ``<name>.png``; returns the JSON path."""
    from pathlib import Path

    import cv2

    from bxjudge.calibration import save_calibration

    img, calib = scene
    calib_dir = Path(calib_dir)
    calib_dir.mkdir(parents=True, exist_ok=True)
    png = calib_dir / f"{name}.png"
    assert cv2.imwrite(str(png), img)
    calib.name = name
    calib.ref_frame = str(png)
    save_calibration(calib, calib_dir / f"{name}.json")
    return calib_dir / f"{name}.json"


class FakeDetector:
    """Detector stand-in: two still tops (X left, B right of the battle-floor centre, 1280x720
    inference pixels) on every frame ``>= launch_frame``; records the frame indices and sizes."""

    def __init__(self, launch_frame: int = 40, x=(500.0, 440.0), b=(780.0, 440.0),
                 size: float = 40.0):
        self.launch_frame = launch_frame
        self.centers = (x, b)
        self.half = size / 2.0
        self.frames: list[int] = []
        self.shapes: set[tuple[int, int]] = set()

    def __call__(self, image, frame: int) -> list[Detection]:
        self.frames.append(frame)
        self.shapes.add((image.shape[1], image.shape[0]))
        if frame < self.launch_frame:
            return []
        return [Detection(frame, "bey", (cx - self.half, cy - self.half, cx + self.half,
                                         cy + self.half), 0.9)
                for cx, cy in self.centers]


def audit_record(rows):
    """Explicit current-schema draw for tests that construct sampled rows in memory."""
    from bxjudge.audit import audit_population

    selection = sorted([r["video"], r["round"]] for r in rows if r.get("audit_sample") is True)
    if not selection:
        return None
    return {"seed": 0, "n": len(selection), "population": audit_population(rows),
            "selection": selection, "at": "2026-10-24T00:00:00+08:00"}
