import logging
import subprocess
from threading import Event

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


# Prefetch must preserve the source stream and release decoder resources even
# when a model stops consuming, fails, or the producer cannot decode a frame.
def test_prefetch_keeps_order_indices_and_original_frame_objects():
    from bxjudge.video import prefetched

    values = [(f, np.full((2, 3, 3), f, np.uint8)) for f in (7, 8, 12, 13)]
    with prefetched(iter(values), cancelled=Event()) as frames:
        actual = list(frames)
    assert [f for f, _ in actual] == [7, 8, 12, 13]
    assert all(a[1] is b[1] for a, b in zip(actual, values, strict=True))


def test_prefetch_propagates_original_decode_error_and_closes_generator():
    from bxjudge.video import VideoDecodeError, prefetched

    events = []
    error = VideoDecodeError("source decode failure")

    def source():
        try:
            yield (8, np.zeros((2, 2, 3), np.uint8))
            raise error
        finally:
            events.append("closed")

    with pytest.raises(VideoDecodeError) as caught, prefetched(source(), cancelled=Event()) as frames:
        assert next(frames)[0] == 8
        next(frames)
    assert caught.value is error
    assert events == ["closed"]


def test_prefetch_consumer_failure_cancels_full_queue_without_leaking_thread():
    import threading

    from bxjudge.video import prefetched

    closed = threading.Event()
    filled = threading.Event()
    source_threads = []

    def source():
        try:
            source_threads.append(threading.current_thread())
            for frame in range(1000):
                if frame == 3:
                    filled.set()
                yield frame, np.zeros((2, 2, 3), np.uint8)
        finally:
            closed.set()

    with pytest.raises(RuntimeError, match="detector failed"), prefetched(source(), cancelled=Event()) as frames:
        assert next(frames)[0] == 0
        assert filled.wait(timeout=2)
        raise RuntimeError("detector failed")
    assert closed.is_set()
    assert len(source_threads) == 1 and not source_threads[0].is_alive()


def test_empty_prefetch_closes_without_hanging():
    from bxjudge.video import prefetched

    with prefetched(iter([]), cancelled=Event()) as frames:
        assert list(frames) == []

def test_prefetch_early_exit_reports_source_close_failure():
    from bxjudge.video import prefetched

    class BrokenClose:
        def __iter__(self):
            return self

        def __next__(self):
            return 0, np.zeros((2, 2, 3), np.uint8)

        def close(self):
            raise RuntimeError("source close failed")

    with pytest.raises(RuntimeError, match="source close failed"), prefetched(
        BrokenClose(), cancelled=Event()
    ) as frames:
        next(frames)


@pytest.mark.parametrize("nested", [False, True])
def test_prefetch_cancels_decoder_blocked_on_pipe(monkeypatch, tmp_path, nested):
    import sys
    import time

    from bxjudge.video import _decode, prefetched

    real_popen = subprocess.Popen
    blocked = Event()
    cancelled = Event()
    processes = []

    class ObservedPipe:
        def __init__(self, pipe):
            self.pipe = pipe
            self.reads = 0

        def readinto(self, target):
            self.reads += 1
            if self.reads == 2:
                blocked.set()
            return self.pipe.readinto(target)

        def close(self):
            self.pipe.close()

    def stalled_decoder(cmd, **kwargs):
        process = real_popen(
            [sys.executable, "-u", "-c",
             ("import sys,time;sys.stdout.buffer.write(bytes(12));"
             "sys.stdout.buffer.flush();time.sleep(30)")],
            **kwargs,
        )
        process.stdout = ObservedPipe(process.stdout)
        processes.append(process)
        return process

    monkeypatch.setattr(video.subprocess, "Popen", stalled_decoder)
    source = _decode(
        tmp_path / "unused.mp4", start=0, end=None, out_w=2, out_h=2,
        hwaccel="cpu", first_index=0, skip=0, cancelled=cancelled,
    )
    def transformed():
        with prefetched(source, cancelled=cancelled) as ready:
            yield from ready

    started = time.perf_counter()
    try:
        with pytest.raises(RuntimeError, match="detector failed"), prefetched(
            transformed() if nested else source, cancelled=cancelled
        ) as frames:
            assert next(frames)[0] == 0
            assert blocked.wait(2)
            raise RuntimeError("detector failed")
        assert time.perf_counter() - started < 3
        assert processes and all(p.poll() is not None for p in processes)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()

def test_prefetch_pre_cancelled_stream_fails_instead_of_hanging():
    from concurrent.futures import CancelledError

    from bxjudge.video import prefetched

    cancelled = Event()
    cancelled.set()
    with pytest.raises(CancelledError), prefetched(iter([]), cancelled=cancelled) as frames:
        list(frames)


def test_prefetch_external_cancel_never_returns_a_partial_success():
    from concurrent.futures import CancelledError

    from bxjudge.video import prefetched

    cancelled = Event()
    waiting = Event()
    closed = Event()

    def source():
        try:
            yield 0, np.zeros((2, 2, 3), np.uint8)
            waiting.set()
            cancelled.wait(2)
        finally:
            closed.set()

    with pytest.raises(CancelledError), prefetched(source(), cancelled=cancelled) as frames:
        assert next(frames)[0] == 0
        assert waiting.wait(2)
        cancelled.set()
        next(frames)
    assert closed.is_set()


def test_nested_prefetch_completion_keeps_all_buffered_frames():
    from bxjudge.video import prefetched

    cancelled = Event()
    values = [(f, np.full((2, 2, 3), f, np.uint8)) for f in range(17)]

    def transformed():
        with prefetched(iter(values), cancelled=cancelled) as raw:
            yield from raw

    with prefetched(transformed(), cancelled=cancelled) as ready:
        actual = list(ready)
    assert [f for f, _ in actual] == list(range(17))
    assert all(a[1] is b[1] for a, b in zip(actual, values, strict=True))
    assert not cancelled.is_set()


def test_nested_prefetch_processing_failure_preserves_original_error():
    from bxjudge.video import prefetched

    cancelled = Event()
    closed = Event()
    error = ValueError("geometry failed")

    def source():
        try:
            for f in range(100):
                yield f, np.zeros((2, 2, 3), np.uint8)
        finally:
            closed.set()

    def transformed():
        with prefetched(source(), cancelled=cancelled) as raw:
            for value in raw:
                if value[0] == 3:
                    raise error
                yield value

    with pytest.raises(ValueError) as caught, prefetched(
        transformed(), cancelled=cancelled
    ) as ready:
        list(ready)
    assert caught.value is error
    assert closed.is_set()


def test_three_stage_prefetch_keeps_upstream_error_when_a_middle_stage_is_busy():
    # Detection -> alignment -> tracking: the decode error cancels the shared
    # event while alignment still holds frame 0. Closing alignment afterwards
    # must still surface the decode error, not a generic cancellation.
    from bxjudge.video import VideoDecodeError, prefetched

    cancelled = Event()
    error = VideoDecodeError("decode failed")

    def source():
        yield 0, np.zeros((2, 2, 3), np.uint8)
        raise error

    def detected():
        with prefetched(source(), cancelled=cancelled) as raw:
            yield from raw

    def aligned():
        with prefetched(detected(), cancelled=cancelled) as ready:
            for value in ready:
                assert cancelled.wait(2)
                yield value

    with pytest.raises(VideoDecodeError) as caught, prefetched(
        aligned(), cancelled=cancelled
    ) as frames:
        list(frames)
    assert caught.value is error


def test_nested_prefetch_normal_break_does_not_raise_its_own_cancellation():
    from bxjudge.video import prefetched

    cancelled = Event()
    outer_waiting = Event()
    closed = Event()

    def source():
        try:
            yield 0, np.zeros((2, 2, 3), np.uint8)
            assert cancelled.wait(2)
        finally:
            closed.set()

    def transformed():
        with prefetched(source(), cancelled=cancelled) as raw:
            yield next(raw)
            outer_waiting.set()
            yield from raw

    with prefetched(transformed(), cancelled=cancelled) as ready:
        assert next(ready)[0] == 0
        assert outer_waiting.wait(2)
    assert cancelled.is_set() and closed.is_set()


def test_early_break_keeps_explicit_source_cancellation_error():
    from concurrent.futures import CancelledError

    from bxjudge.video import prefetched

    cancelled = Event()
    failed = Event()
    error = CancelledError("source cancellation failure")

    def source():
        yield 0, np.zeros((2, 2, 3), np.uint8)
        failed.set()
        raise error

    with pytest.raises(CancelledError) as caught, prefetched(
        source(), cancelled=cancelled
    ) as ready:
        next(ready)
        assert failed.wait(2)
    assert caught.value is error
