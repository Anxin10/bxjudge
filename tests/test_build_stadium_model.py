"""scripts/build_stadium_model.py never writes over manual calibrations (Codex round 8).

Every test works on a copy of the template under tmp_path with PROJECT_ROOT pointed there,
so even a missing guard can never touch the real calib/ (an earlier draft of this test did).
"""
from __future__ import annotations

import importlib.util
import shutil

import pytest

from bxjudge import PROJECT_ROOT


def _load():
    spec = importlib.util.spec_from_file_location(
        "build_stadium_model", PROJECT_ROOT / "scripts" / "build_stadium_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bm = _load()


@pytest.fixture()
def fake_root(tmp_path, monkeypatch):
    calib = tmp_path / "calib"
    calib.mkdir()
    for suffix in (".json", ".png"):
        source = PROJECT_ROOT / "calib" / f"20261014_202713{suffix}"
        if not source.exists():  # calib/ is local-only (gitignored); absent in CI
            pytest.skip(f"local calibration template missing: {source.name}")
        shutil.copy(source, calib)
    monkeypatch.setattr(bm, "PROJECT_ROOT", tmp_path)
    return tmp_path


def test_model_written_into_calib_is_refused(fake_root):
    template = fake_root / "calib" / "20261014_202713.json"
    before = template.read_bytes()
    with pytest.raises(SystemExit):
        bm.main(["--template", str(template), "--out", str(template)])
    with pytest.raises(SystemExit):
        bm.main(["--template", str(template), "--out", str(fake_root / "calib" / "new.json")])
    assert template.read_bytes() == before
    assert not (fake_root / "calib" / "new.json").exists()


def test_preview_over_the_reference_image_is_refused(fake_root):
    template = fake_root / "calib" / "20261014_202713.json"
    image = fake_root / "calib" / "20261014_202713.png"
    before = image.read_bytes()
    with pytest.raises(SystemExit):
        bm.main(["--template", str(template), "--out", str(fake_root / "m.json"),
                 "--preview", str(image)])
    assert image.read_bytes() == before
    assert not (fake_root / "m.json").exists()  # refused before anything is written


def test_template_or_its_image_outside_calib_are_protected_too(tmp_path):
    t, img = tmp_path / "t.json", tmp_path / "t.png"
    with pytest.raises(SystemExit):
        bm.check_targets([t], template=t, reference=img, calib=tmp_path / "calib")
    with pytest.raises(SystemExit):
        bm.check_targets([img], template=t, reference=img, calib=tmp_path / "calib")
    bm.check_targets([tmp_path / "ok.json"], template=t, reference=img, calib=tmp_path / "calib")
