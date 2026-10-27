"""Progressive evidence rendering and real browser-compatible video encoding."""

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml


def test_two_bey_display_bridges_are_tagged_and_never_change_saved_observations():
    from bxjudge.trajectory_video import Observation, two_bey_display_points

    raw=[Observation('X',40,'bey',0,0.,10.,10.),Observation('X',40,'bey',3,.3,40.,10.),
         Observation('',41,'bey',0,0.,100.,100.),Observation('',41,'bey',1,.1,101.,100.)]
    result=two_bey_display_points(raw,10)
    xs=[p for p in result if p.track_id==40]
    assert [p.frame for p in xs]==[0,1,2,3]
    assert [p.kind for p in xs]==['observed','interpolated','interpolated','observed']
    assert xs[1].x==20 and xs[2].x==30
    assert len(raw)==4 and all(p.kind=='observed' for p in raw)
    assert sorted({p.track_id for p in result})==[40,41]


def test_two_bey_display_refuses_a_third_physical_top():
    from bxjudge.trajectory_video import Observation, two_bey_display_points

    with pytest.raises(ValueError,match='two'):
        two_bey_display_points([Observation('',i,'bey',0,0.,float(i),10.) for i in (1,2,3)],30)


def test_two_bey_mode_renders_an_estimated_gap_without_changing_csv(scene):
    from bxjudge.trajectory_plot import SIDE_COLORS
    from bxjudge.trajectory_video import (
        animation_frames,
        load_video_policy,
        read_observations,
    )

    _out,c,row,path,config,_=scene
    before=path.read_bytes()
    points=[p for p in read_observations(path) if p.track_id in (1,2)]
    frames=list(animation_frames(row,c,points,load_video_policy(config),coordinate_size=(640,360),
        recorded_geometry={'two_bey_tracking':{}},source_fps=10))
    assert tuple(frames[2][170,140])==SIDE_COLORS['X']
    assert path.read_bytes()==before
    assert all(p.kind=='observed' for p in points)

from bxjudge.output import trajectory_png
from tests.helpers import cal_with


@pytest.fixture
def scene(tmp_path):
    c = cal_with(np.full((360, 640, 3), 255, np.uint8), tmp_path)
    row = {
        "video": "a.mp4",
        "round": 1,
        "t_launch": 10.0,
        "t_end": 10.5,
        "finish": None,
        "landing_frame": {"X": None, "B": None},
    }
    path = trajectory_png(tmp_path, row["video"], row["round"]).with_suffix(".csv")
    path.parent.mkdir()
    fields = ["frame", "t", "side", "track_id", "cls", "x", "y", "zone", "score"]

    def write(points):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for frame, t, side, tid, x, y in points:
                w.writerow(
                    {
                        "frame": frame,
                        "t": t,
                        "side": side,
                        "track_id": tid,
                        "cls": "bey",
                        "x": x,
                        "y": y,
                        "zone": "battle_area",
                        "score": 0.9 if x != "" else "",
                    }
                )

        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "coordinate_size": [640, 360],
                    "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        )

    write(
        [
            (100, 10.0, "X", 1, 100, 170),
            (101, 10.1, "X", 1, 120, 170),
            (103, 10.3, "X", 1, 160, 170),
            (104, 10.4, "X", 1, 180, 170),
            (100, 10.0, "B", 2, 100, 230),
            (101, 10.1, "B", 2, 120, 230),
            (100, 10.0, "", 9, 100, 280),
            (101, 10.1, "", 9, 120, 280),
        ]
    )
    policy = tmp_path / "video.yaml"
    from bxjudge.trajectory_video import DEFAULT_VIDEO_CONFIG

    policy.write_text(
        yaml.safe_dump(
            {
                "fps": 10,
                "max_long_edge": 640,
                "crf": 18,
                "preset": "ultrafast",
                "encoder_threads": 1,
                "stadium_follow": yaml.safe_load(DEFAULT_VIDEO_CONFIG.read_text())[
                    "stadium_follow"
                ],
            }
        )
    )
    from tests.helpers import make_video

    make_video(tmp_path / "source.mp4", seconds=12, fps=10, size=(640, 360))
    return tmp_path, c, row, path, policy, write


def test_progressive_frames_do_not_reveal_future_or_bridge_gaps(scene):
    from bxjudge.trajectory_video import (
        animation_frames,
        load_video_policy,
        read_observations,
    )

    _, c, row, path, policy, _ = scene
    frames = list(
        animation_frames(
            row,
            c,
            read_observations(path),
            load_video_policy(policy),
            coordinate_size=(640, 360),
        )
    )
    assert len(frames) == 6
    orange, blue, gray = (0, 159, 230), (178, 114, 0), (150, 150, 150)
    assert tuple(frames[0][170, 180]) == (
        255,
        255,
        255,
    )  # final position not visible yet
    assert tuple(frames[1][170, 110]) == orange
    assert tuple(frames[1][230, 110]) == blue
    assert tuple(frames[1][280, 110]) == gray
    assert tuple(frames[-1][170, 140]) == (
        255,
        255,
        255,
    )  # frames101->103 stay disconnected
    assert tuple(frames[-1][170, 170]) == orange
    assert frames[-1].shape == (446, 640, 3)


def test_silent_lead_in_and_last_subframe_observation_are_preserved(scene):
    from bxjudge.trajectory_video import (
        animation_frames,
        load_video_policy,
        read_observations,
    )

    _, c, row, path, policy, write = scene
    write([(102, 10.25, "X", 1, 300, 180), (103, 10.31, "X", 1, 330, 180)])
    row["t_end"] = 10.31
    frames = list(
        animation_frames(
            row,
            c,
            read_observations(path),
            load_video_policy(policy),
            coordinate_size=(640, 360),
        )
    )
    assert len(frames) == 5  # t=10.00,10.10,10.20,10.30,10.31
    assert tuple(frames[0][180, 300]) == (255, 255, 255)
    assert tuple(frames[2][180, 300]) == (255, 255, 255)
    assert tuple(frames[-1][180, 330]) == (0, 159, 230)


@pytest.mark.parametrize(
    "invalid",
    [
        "partial_xy",
        "nonfinite",
        "bad_side",
        "backward_time",
        "oversized",
        "duplicate_time",
    ],
)
def test_bad_observations_do_not_create_a_movie(scene, invalid):
    from bxjudge.trajectory_video import generate_trajectory_video

    out, c, row, _path, policy, write = scene
    points = {
        "oversized": [(1, 10, "X", 1, 1e100, 180)],
        "duplicate_time": [(1, 10, "X", 1, 100, 180), (2, 10, "X", 1, 120, 180)],
        "partial_xy": [(1, 10, "X", 1, 100, "")],
        "nonfinite": [(1, 10, "X", 1, "nan", 180)],
        "bad_side": [(1, 10, "Q", 1, 100, 180)],
        "backward_time": [(1, 10.3, "X", 1, 100, 180), (2, 10.1, "X", 1, 120, 180)],
    }[invalid]
    write(points)
    with pytest.raises(ValueError):
        generate_trajectory_video(out, row, c, source=out / "source.mp4", config=policy)
    assert not list((out / "trajectories").rglob("*.mp4"))


def test_real_h264_video_has_expected_duration_and_cache_identity(scene, monkeypatch):
    from bxjudge import trajectory_video as tv

    out, c, row, csv_path, policy, write = scene
    protected = csv_path.read_bytes()
    path = tv.generate_trajectory_video(
        out, row, c, source=out / "source.mp4", config=policy
    )
    data = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)]
        )
    )
    stream = data["streams"][0]
    assert stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
    assert stream["width"] == 640 and stream["height"] == 446
    assert float(stream["duration"]) == pytest.approx(0.6, abs=0.02)
    before = path.stat().st_mtime_ns
    assert (
        tv.generate_trajectory_video(
            out, row, c, source=out / "source.mp4", config=policy
        )
        == path
    )
    assert path.stat().st_mtime_ns == before and csv_path.read_bytes() == protected
    c.name = "different snapshot metadata"
    changed = tv.generate_trajectory_video(
        out, row, c, source=out / "source.mp4", config=policy
    )
    assert changed != path and changed.is_file()
    write([(1, 10, "X", 1, 350, 200)])
    changed2 = tv.generate_trajectory_video(
        out, row, c, source=out / "source.mp4", config=policy
    )
    assert changed2 not in (path, changed)
    c_image = cv2.imread(c.ref_frame)
    c_image[200, 350] = (1, 2, 3)
    cv2.imwrite(c.ref_frame, c_image)
    assert (
        tv.generate_trajectory_video(
            out, row, c, source=out / "source.mp4", config=policy
        )
        != changed2
    )


def test_failed_encoder_never_publishes_partial_movie(scene, monkeypatch):
    from bxjudge import trajectory_video as tv

    out, c, row, _, policy, _ = scene
    real = tv.subprocess.Popen

    def failing(command, **kwargs):
        return real(["/bin/false"] if "-c:v" in command else command, **kwargs)

    monkeypatch.setattr(tv.subprocess, "Popen", failing)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        tv.generate_trajectory_video(
            out, row, c, source=out / "source.mp4", config=policy
        )
    assert not list((out / "trajectories").rglob("*.mp4"))


@pytest.mark.parametrize("change", ["time", "policy", "renderer"])
def test_cache_is_invalidated_by_timeline_policy_and_renderer(
    scene, monkeypatch, change
):
    from bxjudge import trajectory_video as tv

    out, c, row, _, policy, _ = scene
    first = tv.generate_trajectory_video(
        out, row, c, source=out / "source.mp4", config=policy
    )
    if change == "time":
        row["t_end"] += 0.1
    elif change == "policy":
        cfg = yaml.safe_load(policy.read_text())
        cfg["fps"] = 20
        policy.write_text(yaml.safe_dump(cfg))
    else:
        monkeypatch.setattr(tv, "_RENDERER_ID", "another renderer")
    second = tv.generate_trajectory_video(
        out, row, c, source=out / "source.mp4", config=policy
    )
    assert second != first and second.is_file()


def test_interruption_reaps_encoder_without_publishing(scene, monkeypatch):
    from bxjudge import trajectory_video as tv

    out, c, row, _, policy, _ = scene
    original = tv.animation_frames
    processes = []
    popen = tv.subprocess.Popen

    def interrupt(*args, **kwargs):
        yield next(original(*args, **kwargs))
        raise KeyboardInterrupt

    def record(*args, **kwargs):
        child = popen(*args, **kwargs)
        processes.append(child)
        return child

    monkeypatch.setattr(tv, "animation_frames", interrupt)
    monkeypatch.setattr(tv.subprocess, "Popen", record)
    with pytest.raises(KeyboardInterrupt):
        tv.generate_trajectory_video(
            out, row, c, source=out / "source.mp4", config=policy
        )
    assert processes and all(p.poll() is not None for p in processes)
    assert not list((out / "trajectories").rglob("*.mp4"))


def test_final_floating_point_tick_contains_last_observation(scene):
    from bxjudge import trajectory_video as tv

    _, c, row, path, policy, write = scene
    write([(0, 0.1, "X", 1, 200, 180), (1, 0.8, "X", 1, 300, 180)])
    row["t_launch"], row["t_end"] = 0.1, 0.8
    cfg = tv.load_video_policy(policy)
    cfg["fps"] = 30
    images = list(
        tv.animation_frames(
            row, c, tv.read_observations(path), cfg, coordinate_size=(640, 360)
        )
    )
    assert tuple(images[-1][180, 300]) == (0, 159, 230)


def test_isolated_middle_observation_is_visible_without_a_bridge(scene):
    from bxjudge import trajectory_video as tv

    _, c, row, path, policy, write = scene
    write(
        [
            (0, 10, "X", 1, 200, 180),
            (2, 10.2, "X", 1, 300, 180),
            (4, 10.4, "X", 1, 400, 180),
        ]
    )
    images = list(
        tv.animation_frames(
            row,
            c,
            tv.read_observations(path),
            tv.load_video_policy(policy),
            coordinate_size=(640, 360),
        )
    )
    assert tuple(images[2][180, 300]) == (0, 159, 230)
    assert tuple(images[-1][180, 300]) == (0, 159, 230)
    assert tuple(images[-1][180, 250]) == (255, 255, 255)


def test_snapshot_size_is_not_the_observation_coordinate_size(scene):
    from bxjudge import trajectory_video as tv

    _, c, row, path, policy, write = scene
    c.frame_size = (1920, 1080)
    write([(0, 10, "X", 1, 640, 360)])
    cfg = tv.load_video_policy(policy)
    cfg["max_long_edge"] = 1280
    images = list(
        tv.animation_frames(
            row, c, tv.read_observations(path), cfg, coordinate_size=(1280, 720)
        )
    )
    assert tuple(images[0][360, 640]) == (
        0,
        159,
        230,
    )  # center of 1280x720 observation plane


def test_empty_reference_is_a_handled_media_error(scene):
    from bxjudge.trajectory_video import generate_trajectory_video

    out, c, row, _, policy, _ = scene
    Path(c.ref_frame).write_bytes(b"")
    with pytest.raises(ValueError, match="reference"):
        generate_trajectory_video(out, row, c, source=out / "source.mp4", config=policy)


@pytest.mark.parametrize("bad", ["missing", "digest", "size"])
def test_coordinate_metadata_is_required_and_bound_to_csv(scene, bad):
    from bxjudge.trajectory_video import generate_trajectory_video

    out, c, row, path, policy, _ = scene
    meta = path.with_suffix(".json")
    data = {
        "coordinate_size": [640, 360],
        "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if bad == "missing":
        if meta.exists():
            meta.unlink()
    else:
        if bad == "digest":
            data["csv_sha256"] = "0" * 64
        else:
            data["coordinate_size"] = [True, 360]
        meta.write_text(json.dumps(data))
    with pytest.raises((OSError, ValueError)):
        generate_trajectory_video(out, row, c, source=out / "source.mp4", config=policy)
    assert not list((out / "trajectories").rglob("*.mp4"))


def test_static_writer_records_coordinate_plane_and_csv_digest(scene):
    from bxjudge.output import write_trajectory
    from tests.helpers import make_track, rr

    out, c, _, _, _, _ = scene
    csv_path, _ = write_trajectory(
        out, rr(None, None), {"X": make_track(1, "bey", {0: (200, 180)})}, [], c, 30, {}
    )
    meta = json.loads(csv_path.with_suffix(".json").read_text())
    assert meta == {
        "coordinate_size": [640, 360],
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }


def test_cache_includes_recorded_coordinate_plane(scene):
    from bxjudge.trajectory_video import generate_trajectory_video

    out, c, row, path, policy, _ = scene
    first = generate_trajectory_video(
        out, row, c, source=out / "source.mp4", config=policy
    )
    meta = json.loads(path.with_suffix(".json").read_text())
    meta["coordinate_size"] = [1280, 720]
    path.with_suffix(".json").write_text(json.dumps(meta))
    second = generate_trajectory_video(
        out, row, c, source=out / "source.mp4", config=policy
    )
    assert first != second


def test_real_footage_remains_visible_under_progressive_paths(scene):
    from bxjudge import trajectory_video as tv

    _, c, row, path, policy, _ = scene
    cfg = tv.load_video_policy(policy)
    # A moving synthetic top in untouched source frames, away from trajectory strokes.
    backgrounds = []
    for x in range(6):
        frame = np.full((360, 640, 3), 255, np.uint8)
        cv2.circle(frame, (350 + x * 20, 200), 10, (30, 40, 200), -1)
        backgrounds.append(frame)
    frames = list(
        tv.animation_frames(
            row,
            c,
            tv.read_observations(path),
            cfg,
            coordinate_size=(640, 360),
            background_frames=iter(backgrounds),
        )
    )
    assert tuple(frames[0][200, 350]) == (30, 40, 200)
    assert tuple(frames[3][200, 410]) == (30, 40, 200)
    assert tuple(frames[3][200, 350]) == (
        255,
        255,
        255,
    )  # old body image must not accumulate
    assert tuple(frames[3][170, 110]) == (0, 159, 230)  # trajectory still accumulates


def test_source_sampling_uses_current_real_frame_at_fractional_and_final_times(
    tmp_path,
):
    from bxjudge.manifest import probe
    from bxjudge.trajectory_video import footage_frames

    frames = [np.full((180, 320, 3), (i * 20, 30, 40), np.uint8) for i in range(10)]
    source = tmp_path / "moving.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            "320x180",
            "-r",
            "10",
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "0",
            "-pix_fmt",
            "yuv444p",
            str(source),
        ],
        input=b"".join(f.tobytes() for f in frames),
        check=True,
        capture_output=True,
    )
    images = list(footage_frames(source, 0.15, 0.36, 10, 320, probe(source)))
    assert len(images) == 4  # .15 .25 .35 .36 => frames1,2,3,3
    assert [int(img[90, 160, 0]) for img in images] == pytest.approx(
        [20, 40, 60, 60], abs=2
    )


def test_changed_source_content_never_reuses_previous_background(scene):
    from bxjudge import trajectory_video as tv
    from tests.helpers import make_video

    out, c, row, _, policy, _ = scene
    source = out / "source.mp4"
    first = tv.generate_trajectory_video(out, row, c, source=source, config=policy)
    make_video(source, seconds=12, fps=12, size=(640, 360))
    second = tv.generate_trajectory_video(out, row, c, source=source, config=policy)
    assert second != first and first.is_file() and second.is_file()


def test_missing_source_is_an_explicit_error_without_fixed_image_substitution(scene):
    from bxjudge.trajectory_video import generate_trajectory_video

    out, c, row, _, policy, _ = scene
    with pytest.raises(OSError):
        generate_trajectory_video(
            out, row, c, source=out / "missing.mp4", config=policy
        )
    assert not list((out / "trajectories").rglob("*.mp4"))


def test_source_sampling_counts_from_original_frame_zero_instead_of_approximate_seek(
    monkeypatch,
):
    from types import SimpleNamespace

    from bxjudge import trajectory_video as tv

    seen = {}

    def decode(source, **kwargs):
        seen.update(kwargs)
        for i in range(5):
            yield i, np.full((2, 2, 3), i, np.uint8)

    monkeypatch.setattr(tv, "iter_frames", decode)
    images = list(
        tv.footage_frames(
            Path("source.mp4"), 0.2, 0.3, 10, 1280, SimpleNamespace(fps=10, duration=1)
        )
    )
    assert seen["start"] == 0 and seen["end"] is None
    assert [int(x[0, 0, 0]) for x in images] == [2, 3]


def test_playback_zones_follow_current_frame_and_hide_on_failure(scene):
    from bxjudge import trajectory_video as tv
    from tests.helpers import stadium_scene

    _, _, row, path, policy, _ = scene
    ref, c = stadium_scene(size=(640, 360))
    moved, _ = stadium_scene(size=(640, 360), shift=(12, -4))
    backgrounds = [ref, moved, np.full_like(ref, 128), ref, moved, ref]
    frames = list(
        tv.animation_frames(
            row,
            c,
            tv.read_observations(path),
            tv.load_video_policy(policy),
            coordinate_size=(640, 360),
            reference=ref,
            background_frames=iter(backgrounds),
        )
    )
    green = (80, 200, 80)
    # Horizontal stadium rim y=35 translates to y=31. It must not remain at y=35.
    assert tuple(frames[0][35, 250]) == green
    # Current template evidence is independently verified, without waiting.
    assert tuple(frames[1][31, 250]) == green
    assert tuple(frames[1][35, 250]) != green
    assert not np.any(np.all(frames[2][:360] == green, axis=2))
    assert tuple(frames[3][35, 250]) != green  # recovery waits for stable observed support
    assert tuple(frames[5][35, 250]) == green
    # A bad frame keeps source footage and raw trajectory evidence, not stale zones.
    assert tuple(frames[2][100, 400]) == (128, 128, 128)
    assert tuple(frames[2][170, 110]) == tv.plot.SIDE_COLORS["X"]


def test_playback_requires_explicit_follow_policy(scene):
    from bxjudge.trajectory_video import load_video_policy

    *_, policy, _ = scene
    cfg = yaml.safe_load(policy.read_text())
    del cfg["stadium_follow"]
    policy.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="incomplete"):
        load_video_policy(policy)


def test_recorded_playback_uses_saved_geometry_instead_of_realigning(scene,monkeypatch):
    from dataclasses import asdict

    from bxjudge import trajectory_video as tv
    from bxjudge.autocalib import Alignment
    _,c,row,path,policy,_=scene
    a=asdict(Alignment(((1.,0.,12.),(0.,1.,-4.),(0.,0.,1.)),1.,0.,0.,1.,0.))
    record={'reference_size':[640,360],'samples':{100:{'frame':100,'alignment':a,'status':'tracking'},101:{'frame':101,'alignment':None,'status':'lost'}}}
    def forbidden(*args):raise AssertionError('must consume saved geometry')
    monkeypatch.setattr(tv.dynamic_calibration.DynamicCalibration,'update',forbidden)
    frames=list(tv.animation_frames(row,c,tv.read_observations(path),tv.load_video_policy(policy),coordinate_size=(640,360),background_frames=iter(np.full((360,640,3),128,np.uint8) for _ in range(6)),recorded_geometry=record,source_fps=10))
    assert tuple(frames[0][16,250])==(80,200,80)
    assert not np.any(np.all(frames[1][:360]==(80,200,80),axis=2))
