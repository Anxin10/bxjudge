import json
import os
import subprocess
from datetime import datetime

import pytest

from bxjudge.manifest import mark_processed, parse_recorded_at, probe, scan
from tests.helpers import make_video


def test_parse_recorded_at():
    assert parse_recorded_at("20261001_205842.mp4") == datetime(2026, 10, 1, 20, 58, 42)
    assert parse_recorded_at("clip.mp4") is None
    assert parse_recorded_at("20261301_205842.mp4") is None  # impossible month


def test_probe(tmp_path):
    p = make_video(tmp_path / "20261001_000001.mp4", seconds=2, fps=60, audio=True)
    i = probe(p)
    assert i.fps == 60 and i.has_audio and abs(i.duration - 2) < 0.1
    assert (i.width, i.height) == (320, 180)
    assert i.codec == "h264"
    assert i.size == p.stat().st_size and i.mtime == p.stat().st_mtime
    assert i.recorded_at == datetime(2026, 10, 1, 0, 0, 1)
    assert i.path == p


def test_probe_no_audio_and_no_timestamp(tmp_path):
    i = probe(make_video(tmp_path / "clip.mp4", seconds=1, fps=30))
    assert not i.has_audio and i.fps == 30 and i.recorded_at is None


def test_probe_applies_display_rotation(tmp_path):
    src = make_video(tmp_path / "a.mp4", seconds=1)
    rot = tmp_path / "rot.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", "90", "-i", str(src),
                    "-c", "copy", str(rot)], check=True)
    i = probe(rot)
    assert (i.width, i.height) == (180, 320)  # what the decoder will actually output


def test_probe_garbage_raises(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00" * 1024)
    with pytest.raises(ValueError):
        probe(bad)


def test_scan_idempotent(tmp_path):  # Review Focus 5
    make_video(tmp_path / "a.mp4")
    db = tmp_path / "db.json"
    first = scan(tmp_path, db)
    [mark_processed(i, db) for i in first]
    assert len(first) == 1 and scan(tmp_path, db) == []


def test_scan_does_not_mark_processed(tmp_path):
    make_video(tmp_path / "a.mp4")
    db = tmp_path / "db.json"
    assert len(scan(tmp_path, db)) == 1
    assert len(scan(tmp_path, db)) == 1  # scanning alone never records anything
    assert not db.exists()


def test_scan_returns_changed_file_again(tmp_path):
    p = make_video(tmp_path / "a.mp4")
    db = tmp_path / "db.json"
    for i in scan(tmp_path, db):
        mark_processed(i, db)
    st = p.stat()
    os.utime(p, (st.st_atime, st.st_mtime + 5))  # same size, new mtime
    assert [i.path for i in scan(tmp_path, db)] == [p]


def test_scan_finds_new_files_sorted_and_recurses(tmp_path):
    make_video(tmp_path / "b.mp4", seconds=1)
    (tmp_path / "sub").mkdir()
    make_video(tmp_path / "sub" / "a.mp4", seconds=1)
    (tmp_path / "notes.txt").write_text("x")
    db = tmp_path / "db.json"
    found = scan(tmp_path, db)
    assert [i.path.name for i in found] == ["b.mp4", "a.mp4"] or [i.path.name for i in found] == ["a.mp4", "b.mp4"]
    assert len(found) == 2
    for i in found[:1]:
        mark_processed(i, db)
    assert len(scan(tmp_path, db)) == 1


def test_db_is_json_keyed_by_absolute_path(tmp_path):
    p = make_video(tmp_path / "a.mp4", seconds=1)
    db = tmp_path / "db.json"
    (info,) = scan(tmp_path, db)
    mark_processed(info, db)
    data = json.loads(db.read_text(encoding="utf-8"))
    entry = data[str(p.resolve())]
    assert entry["size"] == info.size and entry["mtime"] == info.mtime


def test_scan_skips_unreadable_video_and_reports(tmp_path):
    make_video(tmp_path / "ok.mp4", seconds=1)
    (tmp_path / "bad.mp4").write_bytes(b"\x00" * 64)
    errors = []
    found = scan(tmp_path, tmp_path / "db.json", on_error=lambda p, e: errors.append(p.name))
    assert [i.path.name for i in found] == ["ok.mp4"]
    assert errors == ["bad.mp4"]


def test_corrupt_db_raises(tmp_path):
    make_video(tmp_path / "a.mp4", seconds=1)
    db = tmp_path / "db.json"
    db.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        scan(tmp_path, db)


def test_cli_scan_lists_new_videos_without_marking(tmp_path, capsys):
    from bxjudge.cli import main

    make_video(tmp_path / "20261001_000001.mp4", seconds=1, fps=60, audio=True)
    db = tmp_path / "db.json"
    assert main(["scan", str(tmp_path), "--db", str(db)]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    line = out[0]
    assert "20261001_000001.mp4" in line and "60.00fps" in line and "320x180" in line
    assert "h264" in line and "audio=yes" in line
    assert not db.exists()
    assert main(["scan", str(tmp_path), "--db", str(db)]) == 0  # still listed: not processed
    assert len(capsys.readouterr().out.strip().splitlines()) == 1


def test_cli_scan_missing_folder(tmp_path):
    from bxjudge.cli import main

    assert main(["scan", str(tmp_path / "nope"), "--db", str(tmp_path / "db.json")]) == 2
