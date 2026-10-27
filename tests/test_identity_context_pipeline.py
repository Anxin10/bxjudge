"""Launcher ownership remains observable when stadium registration is unknown."""

import json
from pathlib import Path

import pytest

from bxjudge import pipeline
from bxjudge.dynamic_calibration import (
    DEFAULT_DYNAMIC_CONFIG,
    DynamicCalibration,
    GeometrySample,
)
from tests.helpers import make_track
from tests.test_pipeline import _kw, calibrated, info


@pytest.mark.parametrize("starts_free", [False, True])
def test_observed_launcher_context_does_not_require_stadium_geometry(
    tmp_path, monkeypatch, th, rules, starts_free
):
    positions_x = {f: (400, 160) for f in range(26)}
    positions_b = {f: (880, 160) for f in range(26)}
    if starts_free:
        # A raw free observation cannot be hidden by missing registration and
        # replaced with later pickup/attachment evidence.
        positions_x[0] = (400, 360)
        positions_b[0] = (880, 360)
    tracks = [
        make_track(1, "bey", positions_x),
        make_track(2, "bey", positions_b),
        make_track(3, "launcher", {f: (400, 80) for f in range(26)}, size=120),
        make_track(4, "launcher", {f: (880, 80) for f in range(26)}, size=120),
    ]
    calibrated(monkeypatch, tmp_path, tracks=tracks)

    def unknown_geometry(self, frame, image):
        sample = GeometrySample(frame, None, 0, "lost")
        self.samples[frame] = sample
        self.diagnostics[frame] = {
            "tracking": {"reason": "synthetic_geometry_missing"},
            "relocation": None, "reason": "synthetic_geometry_missing",
        }
        return sample

    monkeypatch.setattr(DynamicCalibration, "update", unknown_geometry)
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"unchanged source")
    [result] = pipeline.judge_video(
        info(source),
        tmp_path,
        lambda *a: [],
        rules,
        th,
        **_kw(tmp_path),
        dynamic_config=DEFAULT_DYNAMIC_CONFIG,
        two_bey_config=Path("config/two_bey_tracking.yaml"),
    )
    record = json.loads(
        (tmp_path / "out" / result.geometry_evidence["path"]).read_bytes()
    )
    expected = {} if starts_free else {"1": "X", "2": "B"}
    assert record["identity"]["anchors"]["sides"] == expected
    assert all(s["alignment"] is None for s in record["samples"])
    assert result.needs_review and result.winner is None and result.finish is None
