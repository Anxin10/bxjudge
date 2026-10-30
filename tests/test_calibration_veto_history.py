"""All active vetoes for an own-frame calibration survive later selections."""

import json

import numpy as np
import pytest

from bxjudge.autocalib import load_autocalib_config
from bxjudge.calibration_store import (
    select_snapshot,
    unveto_calibration,
    veto_calibration,
    vetoed_ids,
)
from bxjudge.config import load_thresholds
from tests.helpers import add_manual_calibration, stadium_scene


@pytest.fixture
def history(tmp_path):
    calib, out = tmp_path / "calib", tmp_path / "out"
    image, calibration = stadium_scene()
    noisy = np.clip(image + np.random.default_rng(1).normal(0, 45, image.shape), 0, 255)
    add_manual_calibration(calib, (noisy.astype(np.uint8), calibration), name="A_old")
    frame = stadium_scene(shift=(45, 20))[0]
    cfg, th = load_autocalib_config(), load_thresholds()
    a = select_snapshot(frame, calib, out, th, cfg)
    assert a.kind == "auto" and a.calibration.name == "auto-A_old"
    veto_calibration(out, a.id, autocalib=cfg, video="clip.mp4")
    add_manual_calibration(calib, stadium_scene(shift=(37, 15)), name="B_new")
    b = select_snapshot(frame, calib, out, th, cfg, vetoed_ids(out), video="clip.mp4")
    assert b.kind == "auto" and b.calibration.name == "auto-B_new"
    veto_calibration(out, b.id, autocalib=cfg, video="clip.mp4")
    return calib, out, frame, th, cfg, a, b


@pytest.mark.parametrize("add_unrelated", [False, True])
@pytest.mark.parametrize("source_changed", [False, True])
def test_sequential_vetoes_never_resurrect_an_unchanged_template(
    history, add_unrelated, source_changed
):
    calib, out, frame, th, cfg, a, b = history
    if add_unrelated:
        add_manual_calibration(calib, stadium_scene(rotation_deg=30), name="unrelated")
    if source_changed:
        frame = stadium_scene(shift=(49, 20))[0]
    assert select_snapshot(frame, calib, out, th, cfg, vetoed_ids(out), video="clip.mp4") is None
    assert vetoed_ids(out) == {a.id, b.id}


@pytest.mark.parametrize("release", ["a", "b"])
def test_unveto_releases_only_the_chosen_template(history, release):
    calib, out, frame, th, cfg, a, b = history
    allowed, blocked = (a, b) if release == "a" else (b, a)
    unveto_calibration(out, allowed.id, autocalib=cfg)
    selected = select_snapshot(frame, calib, out, th, cfg, vetoed_ids(out), video="clip.mp4")
    assert selected.template_id == allowed.template_id
    assert vetoed_ids(out) == {blocked.id}


def test_history_does_not_veto_a_neighbouring_videos_own_frame(history):
    calib, out, _, th, cfg, a, b = history
    frame = stadium_scene(shift=(46, 21))[0]
    selected = select_snapshot(frame, calib, out, th, cfg, vetoed_ids(out), video="neighbour.mp4")
    assert selected.kind == "auto" and selected.id not in {a.id, b.id}


def test_earlier_veto_remains_when_current_snapshot_is_not_vetoed(history):
    calib, out, frame, th, cfg, _, b = history
    unveto_calibration(out, b.id, autocalib=cfg)
    (calib / "B_new.json").unlink()
    add_manual_calibration(calib, stadium_scene(rotation_deg=30), name="unrelated")
    assert select_snapshot(frame, calib, out, th, cfg, vetoed_ids(out), video="clip.mp4") is None


def test_changed_template_can_replace_its_vetoed_version(history):
    calib, out, frame, th, cfg, a, b = history
    path = calib / "A_old.json"
    data = json.loads(path.read_text())
    data["name"] = "A_revised"
    path.write_text(json.dumps(data))
    selected = select_snapshot(frame, calib, out, th, cfg, vetoed_ids(out), video="clip.mp4")
    assert selected.kind == "auto" and selected.calibration.name == "auto-A_revised"
    assert selected.template_id not in {a.template_id, b.template_id}


@pytest.mark.parametrize("damage", ["missing", "undecodable"])
def test_unreadable_veto_evidence_cannot_be_silently_skipped(history, damage):
    calib, out, frame, th, cfg, a, _b = history
    image = out / "calibrations" / f"{a.id}.png"
    if damage == "missing":
        image.unlink()
    else:
        image.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="reference image"):
        select_snapshot(frame, calib, out, th, cfg, vetoed_ids(out), video="clip.mp4")


def test_explicit_video_scope_keeps_vetoes_when_source_frame_changes(history):
    calib, out, _, th, cfg, _, _b = history
    changed = stadium_scene(shift=(49, 20))[0]
    assert select_snapshot(changed, calib, out, th, cfg, vetoed_ids(out), video="clip.mp4") is None
    # A different video at the same pixels is not inferred to share the veto scope.
    neighbour = select_snapshot(changed, calib, out, th, cfg, vetoed_ids(out), video="other.mp4")
    assert neighbour.kind == "auto"


def test_old_veto_survives_successful_source_replacement_and_another_veto(history):
    calib, out, _, th, cfg, a, b = history
    unveto_calibration(out, b.id, autocalib=cfg)
    changed = stadium_scene(shift=(49, 20))[0]
    replacement = select_snapshot(changed, calib, out, th, cfg, vetoed_ids(out), video="clip.mp4")
    assert replacement.template_id == b.template_id
    veto_calibration(out, replacement.id, autocalib=cfg, video="clip.mp4")
    assert select_snapshot(changed, calib, out, th, cfg, vetoed_ids(out), video="clip.mp4") is None
    assert a.id in vetoed_ids(out)


def test_active_vetoes_require_explicit_video_identity(history):
    calib, out, frame, th, cfg, _, _b = history
    with pytest.raises(ValueError, match="video identity is required"):
        select_snapshot(frame, calib, out, th, cfg, vetoed_ids(out))


def test_unscoped_veto_format_is_rejected(history):
    _, out, _, _, _, a, _b = history
    path = out / "calibration_vetoes.json"
    data = json.loads(path.read_text())
    del data[a.id]["videos"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unscoped veto records are not supported"):
        vetoed_ids(out)


def test_reapplying_a_veto_preserves_existing_video_bindings(history):
    _, out, _, _, cfg, a, _b = history
    veto_calibration(out, a.id, autocalib=cfg, video="other.mp4")
    data = json.loads((out / "calibration_vetoes.json").read_text())
    assert data[a.id]["videos"] == ["clip.mp4", "other.mp4"]
