import logging
import subprocess

import numpy as np
import pytest

from bxjudge import video
from bxjudge.video import VideoDecodeError, iter_frames
from tests.helpers import make_video


def test_iter_frames_keeps_all_frames(tmp_path):
    p = make_video(tmp_path / "a.mp4", seconds=2, fps=60)
    assert sum(1 for _ in iter_frames(p, hwaccel="cpu")) == 120


def test_frame_indices_are_source_frame_numbers(tmp_path):
    p = make_video(tmp_path / "a.mp4", seconds=2, fps=60)
    frames = list(iter_frames(p, hwaccel="cpu"))
    assert [i for i, _ in frames] == list(range(120))
    f = frames[0][1]
    assert f.dtype == np.uint8 and f.shape == (180, 320, 3)  # BGR, HxWx3, not downscaled


def test_frames_differ_over_time(tmp_path):
    p = make_video(tmp_path / "a.mp4", seconds=1, fps=30)
    frames = [f for _, f in iter_frames(p, hwaccel="cpu")]
    assert not np.array_equal(frames[0], frames[-1])


def test_downscale_long_edge_even_dims(tmp_path):
    p = make_video(tmp_path / "a.mp4", seconds=1, fps=30)
    _, f = next(iter_frames(p, long_edge=100, hwaccel="cpu"))
    assert f.shape == (56, 100, 3)  # 320x180 -> 100x56.25 -> even height


def test_never_upscales(tmp_path):
    p = make_video(tmp_path / "a.mp4", seconds=1, fps=30)
    _, f = next(iter_frames(p, long_edge=1280, hwaccel="cpu"))
    assert f.shape == (180, 320, 3)


def test_start_end_window(tmp_path):
    p = make_video(tmp_path / "a.mp4", seconds=2, fps=60)
    frames = list(iter_frames(p, start=1.0, end=1.5, hwaccel="cpu"))
    idx = [i for i, _ in frames]
    assert idx[0] == 60 and idx == list(range(60, 60 + len(idx)))
    assert len(idx) == 30


def test_portrait_rotated_video_decodes_upright(tmp_path):
    src = make_video(tmp_path / "a.mp4", seconds=1, fps=30)
    rot = tmp_path / "rot.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", "90", "-i", str(src),
                    "-c", "copy", str(rot)], check=True)
    frames = list(iter_frames(rot, hwaccel="cpu"))
    assert len(frames) == 30 and frames[0][1].shape == (320, 180, 3)


def test_early_break_cleans_up(tmp_path):
    p = make_video(tmp_path / "a.mp4", seconds=2, fps=60)
    gen = iter_frames(p, hwaccel="cpu")
    next(gen)
    gen.close()  # must not hang or raise


def test_corrupt_video_raises(tmp_path):  # Review Focus 4
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00" * 1024)
    with pytest.raises(VideoDecodeError):
        list(iter_frames(bad, hwaccel="cpu"))


def test_corrupt_video_raises_with_cuda_too(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00" * 1024)
    with pytest.raises(VideoDecodeError):
        list(iter_frames(bad, hwaccel="cuda"))


def test_missing_file_raises(tmp_path):
    with pytest.raises(VideoDecodeError):
        list(iter_frames(tmp_path / "nope.mp4", hwaccel="cpu"))


def test_truncated_stream_raises_not_silent(tmp_path):
    """A file that is cut short must not quietly yield a partial/empty result."""
    p = make_video(tmp_path / "a.mp4", seconds=2, fps=60)
    data = p.read_bytes()
    cut = tmp_path / "cut.mp4"
    cut.write_bytes(data[: len(data) // 2])
    try:
        n = sum(1 for _ in iter_frames(cut, hwaccel="cpu"))
    except VideoDecodeError:
        return
    assert n > 0  # never an empty success


def test_window_past_end_is_error_not_empty(tmp_path):
    p = make_video(tmp_path / "a.mp4", seconds=1, fps=30)
    with pytest.raises(VideoDecodeError):
        list(iter_frames(p, start=5.0, hwaccel="cpu"))


def test_bad_arguments(tmp_path):
    p = make_video(tmp_path / "a.mp4", seconds=1, fps=30)
    with pytest.raises(ValueError):
        list(iter_frames(p, hwaccel="tpu"))
    with pytest.raises(ValueError):
        list(iter_frames(p, start=1.0, end=0.5, hwaccel="cpu"))


def test_cuda_failure_falls_back_to_cpu_once(tmp_path, monkeypatch, caplog):
    p = make_video(tmp_path / "a.mp4", seconds=1, fps=30)
    monkeypatch.setattr(video, "_hwaccel_args", lambda hw: ["-hwaccel", "no_such_accel"] if hw == "cuda" else [])
    with caplog.at_level(logging.WARNING, logger="bxjudge.video"):
        n = sum(1 for _ in iter_frames(p, hwaccel="cuda"))
    assert n == 30
    assert any("cpu" in r.getMessage().lower() for r in caplog.records)


def test_cpu_failure_is_not_retried(tmp_path, monkeypatch):
    p = make_video(tmp_path / "a.mp4", seconds=1, fps=30)
    calls = []
    real = video._decode

    def counting(*a, **k):
        calls.append(k.get("hwaccel", a[4] if len(a) > 4 else None))
        return real(*a, **k)

    monkeypatch.setattr(video, "_hwaccel_args", lambda hw: ["-hwaccel", "no_such_accel"])
    monkeypatch.setattr(video, "_decode", counting)
    with pytest.raises(VideoDecodeError):
        list(iter_frames(p, hwaccel="cpu"))
    assert len(calls) == 1


@pytest.mark.gpu
def test_cuda_decodes_all_frames(tmp_path):
    p = make_video(tmp_path / "a.mp4", seconds=2, fps=60)
    assert sum(1 for _ in iter_frames(p, hwaccel="cuda")) == 120
