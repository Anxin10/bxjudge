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
