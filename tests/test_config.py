import dataclasses

import pytest
import yaml

from bxjudge.config import Thresholds, load_thresholds


def test_load_default_thresholds():
    t = load_thresholds()
    assert t.fps_default == 60 and isinstance(t.fps_default, int)
    assert t.review_confidence == 0.7
    assert t.decode_hwaccel == "cuda"
    assert t.shake_px == 2.0


def _write(tmp_path, mutate):
    data = yaml.safe_load(open("config/thresholds.yaml", encoding="utf-8"))
    mutate(data)
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def test_unknown_key_fails(tmp_path):
    p = _write(tmp_path, lambda d: d.update(bogus=1))
    with pytest.raises(ValueError, match="bogus"):
        load_thresholds(p)


def test_missing_key_fails(tmp_path):
    p = _write(tmp_path, lambda d: d.pop("still_px"))
    with pytest.raises(ValueError, match="still_px"):
        load_thresholds(p)


def test_bad_hwaccel_and_type_fail(tmp_path):
    with pytest.raises(ValueError):
        load_thresholds(_write(tmp_path, lambda d: d.update(decode_hwaccel="tpu")))
    with pytest.raises(ValueError):
        load_thresholds(_write(tmp_path, lambda d: d.update(still_frames="30")))


def test_yaml_keys_match_dataclass_fields():
    keys = set(yaml.safe_load(open("config/thresholds.yaml", encoding="utf-8")))
    assert keys == {f.name for f in dataclasses.fields(Thresholds)}
