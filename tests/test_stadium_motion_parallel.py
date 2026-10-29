"""Raw-motion ECC runs on a bounded worker pool (task 7 speed, R66a).

On 214505 the serial raw-motion stage held processing at 0.99x playback; two workers
measured 0.76x with identical displacement values. The parallel path must keep the
serial contract: same order, same frame objects, same values, failures never hidden.
"""
import threading

import cv2
import numpy as np
import pytest

from bxjudge.stadium_motion import RAW_MOTION_WORKERS, iter_stadium_displacement
from tests.helpers import make_calibration, shift


def frames_of(ref_img):
    small = cv2.resize(ref_img, (320, 180), interpolation=cv2.INTER_AREA)
    flat = np.full_like(ref_img, 128)  # does not converge: nan
    return [(0, ref_img), (1, shift(ref_img, 3, 0)), (2, flat), (3, small),
            (4, shift(small, 0, -5)), (5, shift(ref_img, 3, 4)), (6, ref_img)]


def test_parallel_matches_serial_order_frames_and_values(ref_img):
    frames = frames_of(ref_img)
    serial = list(iter_stadium_displacement(frames, make_calibration(), ref_img))
    parallel = list(iter_stadium_displacement(frames, make_calibration(), ref_img, workers=3))
    assert [i for i, _, _ in parallel] == [i for i, _, _ in serial] == list(range(7))
    assert all(p[1] is f for p, (_, f) in zip(parallel, frames, strict=True))
    for (_, _, a), (_, _, b) in zip(serial, parallel, strict=True):
        assert (np.isnan(a) and np.isnan(b)) or a == b


def test_parallel_reads_at_most_two_frames_per_worker_ahead(ref_img):
    pulled = []

    def source():
        for i in range(20):
            pulled.append(i)
            yield i, shift(ref_img, i % 4, 0)

    it = iter_stadium_displacement(source(), make_calibration(), ref_img, workers=2)
    next(it)
    assert len(pulled) <= 2 * 2 + 1
    it.close()


def test_a_source_failure_follows_the_frames_already_read(ref_img):
    def source():
        yield 0, ref_img
        yield 1, shift(ref_img, 3, 0)
        raise RuntimeError("decode failed")

    seen = []
    with pytest.raises(RuntimeError, match="decode failed"):
        for idx, _, _ in iter_stadium_displacement(source(), make_calibration(), ref_img, workers=2):
            seen.append(idx)
    assert seen == [0, 1]


def test_closing_early_leaves_no_worker_running(ref_img):
    frames = [(i, shift(ref_img, i % 4, 0)) for i in range(12)]
    it = iter_stadium_displacement(frames, make_calibration(), ref_img, workers=2)
    next(it)
    it.close()
    assert not [t for t in threading.enumerate() if t.name.startswith("bxjudge-raw-motion")]


def test_workers_must_be_positive(ref_img):
    with pytest.raises(ValueError, match="workers"):
        list(iter_stadium_displacement([(0, ref_img)], make_calibration(), ref_img, workers=0))


def test_the_pipeline_uses_two_workers():
    assert RAW_MOTION_WORKERS == 2


# Codex task-7 speed review (2026-10-29): same contract as video.prefetched.
def test_closing_early_still_reports_a_source_failure_already_read(ref_img):
    def source():
        yield 0, ref_img
        yield 1, shift(ref_img, 3, 0)
        raise RuntimeError("decode failed")

    it = iter_stadium_displacement(source(), make_calibration(), ref_img, workers=2)
    next(it)
    with pytest.raises(RuntimeError, match="decode failed"):
        it.close()


def test_closing_early_does_not_report_an_upstream_cancellation(ref_img):
    from concurrent.futures import CancelledError

    def source():
        yield 0, ref_img
        yield 1, shift(ref_img, 3, 0)
        raise CancelledError("frame decoding cancelled")

    it = iter_stadium_displacement(source(), make_calibration(), ref_img, workers=2)
    next(it)
    it.close()
