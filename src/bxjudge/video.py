"""ffmpeg-based frame decoding: stream BGR frames through a rawvideo pipe.

Frame indices count at the SOURCE frame rate; frames are never dropped (``-fps_mode
passthrough``). Frames are streamed one at a time, so a 1 GB 4K file never sits in memory.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from concurrent.futures import CancelledError
from contextlib import contextmanager
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Iterator

import numpy as np

from bxjudge.manifest import probe

log = logging.getLogger(__name__)

_HWACCELS = ("cuda", "cpu")


class VideoDecodeError(Exception):
    """A video could not be decoded (corrupt, missing, no frames, ffmpeg failure)."""


def _hwaccel_args(hwaccel: str) -> list[str]:
    # No `-hwaccel_output_format`: frames come back to system memory so the scale filter and
    # the BGR conversion run on the CPU regardless of where the decoding happened.
    return ["-hwaccel", "cuda"] if hwaccel == "cuda" else []


def _even(x: float) -> int:
    return max(2, int(round(x / 2.0)) * 2)


def _out_size(width: int, height: int, long_edge: int) -> tuple[int, int]:
    """Downscale so the long edge is <= ``long_edge`` (never upscale), even dims, same aspect."""
    scale = min(1.0, long_edge / max(width, height))
    if scale >= 1.0:
        return width - width % 2 or 2, height - height % 2 or 2
    return _even(width * scale), _even(height * scale)


def _decode(
    path: Path,
    *,
    start: float,
    end: float | None,
    out_w: int,
    out_h: int,
    hwaccel: str,
    first_index: int,
    skip: int,
    cancelled: Event | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """One ffmpeg run. Yields ``(frame_index, frame)`` skipping the first ``skip`` frames;
    raises ``VideoDecodeError`` on ffmpeg failure, a truncated frame, or zero frames."""
    if cancelled is not None and cancelled.is_set():
        return
    cmd = ["ffmpeg", "-v", "error", "-nostdin", *_hwaccel_args(hwaccel)]
    if start > 0:
        cmd += ["-ss", f"{start:.6f}"]
    cmd += ["-i", str(path)]
    if end is not None:
        cmd += ["-t", f"{end - start:.6f}"]
    cmd += ["-an", "-sn", "-vf", f"scale={out_w}:{out_h}", "-pix_fmt", "bgr24",
            "-fps_mode", "passthrough", "-f", "rawvideo", "-"]

    frame_bytes = out_w * out_h * 3
    n = 0
    with tempfile.TemporaryFile() as err:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, bufsize=0)
        except FileNotFoundError as e:
            raise VideoDecodeError("ffmpeg not found; install ffmpeg") from e
        finished = Event()

        def cancel_decode():
            while not finished.wait(0.05):
                if cancelled.is_set():
                    proc.kill()
                    return

        watcher = (
            Thread(target=cancel_decode, name="bxjudge-decode-cancel")
            if cancelled is not None else None
        )
        try:
            if watcher is not None:
                watcher.start()
            assert proc.stdout is not None
            while True:
                buf = bytearray(frame_bytes)
                view = memoryview(buf)
                got = 0
                while got < frame_bytes:
                    k = proc.stdout.readinto(view[got:])
                    if not k:
                        break
                    got += k
                if cancelled is not None and cancelled.is_set():
                    return
                if got == 0:
                    break
                if got < frame_bytes:
                    raise VideoDecodeError(f"{path}: truncated frame {first_index + n} "
                                           f"({got}/{frame_bytes} bytes)")
                idx = first_index + n
                n += 1
                if n > skip:
                    yield idx, np.frombuffer(buf, np.uint8).reshape(out_h, out_w, 3)
            rc = proc.wait()
        finally:
            finished.set()
            if watcher is not None and watcher.ident is not None:
                watcher.join()
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            if proc.stdout:
                proc.stdout.close()
        if cancelled is not None and cancelled.is_set():
            return
        if rc != 0:
            err.seek(0)
            msg = err.read().decode(errors="replace").strip()
            raise VideoDecodeError(f"{path}: ffmpeg exited with {rc} ({hwaccel}): {msg[-500:]}")
        if n == 0:
            err.seek(0)
            msg = err.read().decode(errors="replace").strip()
            raise VideoDecodeError(f"{path}: no frames decoded ({hwaccel}) {msg[-500:]}")


def iter_frames(
    path: Path,
    start: float = 0.0,
    end: float | None = None,
    long_edge: int = 1280,
    hwaccel: str = "cuda",
    *,
    cancelled: Event | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(source_frame_index, BGR frame)`` for ``[start, end)`` seconds.

    Frames are downscaled so the long edge is <= ``long_edge`` (aspect kept, even dims). With
    ``hwaccel="cuda"`` NVDEC is tried first; if it fails, the decode is retried once on the CPU
    (continuing after the frames already delivered). Any remaining failure, a non-zero ffmpeg
    exit, or a decode that yields zero frames raises ``VideoDecodeError``.
    """
    if hwaccel not in _HWACCELS:
        raise ValueError(f"hwaccel must be one of {_HWACCELS}, got {hwaccel!r}")
    if start < 0 or (end is not None and end <= start):
        raise ValueError(f"bad window: start={start}, end={end}")

    path = Path(path)
    try:
        info = probe(path)
    except (ValueError, OSError) as e:
        raise VideoDecodeError(f"cannot read {path}: {e}") from e

    out_w, out_h = _out_size(info.width, info.height, long_edge)
    first_index = int(round(start * info.fps))
    attempts = ["cuda", "cpu"] if hwaccel == "cuda" else ["cpu"]
    delivered = 0
    for i, hw in enumerate(attempts):
        try:
            for idx, frame in _decode(
                path, start=start, end=end, out_w=out_w, out_h=out_h,
                hwaccel=hw, first_index=first_index, skip=delivered, cancelled=cancelled,
            ):
                delivered += 1
                yield idx, frame
            return
        except VideoDecodeError as e:
            if cancelled is not None and cancelled.is_set():
                return
            if i + 1 >= len(attempts):
                raise
            log.warning("cuda decode of %s failed (%s); retrying on cpu", path.name, e)


class _PrefetchCancelled(CancelledError):
    """Shared-event cancellation, distinct from an explicit source failure."""


@contextmanager
def prefetched(
    source: Iterator[tuple[int, np.ndarray]],
    *,
    cancelled: Event,
) -> Iterator[Iterator[tuple[int, np.ndarray]]]:
    """Overlap frame preparation with consumption, retaining order and frame objects.

    Only the producer advances/closes the source. Two queued frames bound memory;
    The source must observe the shared cancellation event (as iter_frames does).
    Exiting cancels decoding/backpressure and joins the producer before returning.
    Normal exhaustion leaves the shared event unset so stages can be composed.
    A source failure is delivered after the frames already prepared, never hidden.
    """
    pending: Queue[tuple[int, np.ndarray] | BaseException | None] = Queue(maxsize=2)
    stopped = Event()
    failure = None
    exhausted = False

    def send(value):
        while not stopped.is_set() and not cancelled.is_set():
            try:
                pending.put(value, timeout=0.05)
                return
            except Full:
                continue

    def produce():
        nonlocal failure
        error = None
        try:
            for item in source:
                if stopped.is_set() or cancelled.is_set():
                    break
                send(item)
        except BaseException as exc:  # noqa: BLE001 - forward worker failures to consumer
            error = exc
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                try:
                    close()
                except BaseException as exc:  # noqa: BLE001 - forward close failures too
                    if error is None:
                        error = exc
            failure = error
            send(error)

    def consume():
        nonlocal exhausted
        while True:
            if cancelled.is_set():
                worker.join()
                if failure is not None:
                    raise failure
                raise _PrefetchCancelled("frame decoding cancelled")
            try:
                item = pending.get(timeout=0.05)
            except Empty:
                continue
            if item is None:
                exhausted = True
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    worker = Thread(target=produce, name="bxjudge-frame-prefetch")
    worker.start()
    frames = consume()
    completed = False
    closed_downstream = False
    try:
        yield frames
        completed = True
    except GeneratorExit:
        # A later stage may stop on the shared event this source's own failure
        # raised, closing this stage before it could forward that failure.
        closed_downstream = True
        raise
    finally:
        stopped.set()
        cancelled_here = not cancelled.is_set() and not exhausted
        if not exhausted:
            cancelled.set()
        worker.join()
        frames.close()
        if completed and failure is not None:  # noqa: SIM102 - distinguish cleanup below
            # Normal early exit may cancel an upstream prefetch waiting for data.
            # Do not mistake our own cleanup for a source/close failure.
            if not (cancelled_here and isinstance(failure, _PrefetchCancelled)):
                raise failure
        if (
            closed_downstream
            and failure is not None
            and not isinstance(failure, _PrefetchCancelled)
        ):
            raise failure
