"""E9: retry later windows when the stadium is hidden at the start (fixed cameras only)."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from bxjudge import stadium_finder as sf

CFG = Path("config/stadium_finder.yaml")


def test_default_off():
    cfg = sf.load_stadium_finder_config(CFG)
    assert cfg.later_windows_enabled is False


def test_old_config_without_keys_still_loads(tmp_path):
    raw = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    del raw["later_windows_enabled"], raw["later_windows_max_start_s"]
    path = tmp_path / "sf.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = sf.load_stadium_finder_config(path)
    assert cfg.later_windows_enabled is False


@pytest.mark.parametrize("key,value", [("later_windows_enabled", 1),
                                       ("later_windows_max_start_s", 0),
                                       ("later_windows_max_start_s", True)])
def test_invalid_values_rejected(tmp_path, key, value):
    raw = {**yaml.safe_load(CFG.read_text(encoding="utf-8")), key: value}
    path = tmp_path / "sf.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        sf.load_stadium_finder_config(path)


def _starts(monkeypatch, enabled, later_windows, duration=54.3, found_at=None):
    seen = []

    def fake(video, model, cfg, acfg, fps, start, **kw):
        seen.append(start)
        return "found" if start == found_at else None

    monkeypatch.setattr(sf, "_find_in_window", fake)
    monkeypatch.setattr("bxjudge.manifest.probe",
                        lambda p: SimpleNamespace(fps=120.0, duration=duration))
    monkeypatch.setattr("bxjudge.autocalib.load_autocalib_config", lambda: None)
    cfg = replace(sf.load_stadium_finder_config(CFG), later_windows_enabled=enabled)
    result = sf.find_stadium(Path("v.mp4"), None, cfg, later_windows=later_windows)
    return seen, result


def test_only_first_window_unless_both_flags(monkeypatch):
    assert _starts(monkeypatch, False, True)[0] == [0.0]
    assert _starts(monkeypatch, True, False)[0] == [0.0]


def test_later_windows_until_max_start(monkeypatch):
    seen, result = _starts(monkeypatch, True, True)
    assert seen == [0.0, 6.0, 12.0, 18.0, 24.0, 30.0]
    assert result is None


def test_later_windows_stop_at_first_found_and_video_end(monkeypatch):
    assert _starts(monkeypatch, True, True, found_at=18.0) == ([0.0, 6.0, 12.0, 18.0], "found")
    assert _starts(monkeypatch, True, True, duration=13.0)[0] == [0.0, 6.0, 12.0]
