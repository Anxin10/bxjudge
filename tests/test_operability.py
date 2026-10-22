"""Real CLI and local transport boundaries, without model loads."""
import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from bxjudge import cli, vlm
from tests.helpers import rr


@pytest.mark.parametrize("case", ["video", "dataset", "labels", "yaml", "run", "vlm"])
@pytest.mark.parametrize("debug", [False, True])
def test_real_cli_expected_errors(tmp_path, case, debug):
    empty = tmp_path / "empty.json"
    empty.write_text("[]")
    bad = tmp_path / "bad.yaml"
    bad.write_text("x: [")
    commands = {
        "video": ["calibrate", str(tmp_path / "missing.mp4"), "--name", "missing"],
        "dataset": ["train", "--kind", "yolo26", "--dataset", str(tmp_path / "missing"),
                    "--out", str(tmp_path / "train")],
        "labels": ["label", "export", str(empty), "--out", str(tmp_path / "dataset")],
        "yaml": ["label", "sample", str(tmp_path), "--config", str(bad)],
        "run": ["run", str(tmp_path / "missing")],
        "vlm": ["vlm", str(tmp_path / "missing")],
    }
    process = subprocess.run([sys.executable, "-m", "bxjudge.cli",
                              *(["--debug"] if debug else []), *commands[case]],
                             check=False, capture_output=True, text=True, env=os.environ.copy())
    assert process.returncode != 0
    assert ("Traceback" in process.stderr) is debug
    if not debug:
        assert process.stderr.startswith("error:")


def test_debug_after_nested_command(tmp_path):
    with pytest.raises(FileNotFoundError):
        cli.main(["label", "export", str(tmp_path / "missing"), "--debug"])


def test_cancel_and_unexpected_errors(monkeypatch, capsys):
    def fail(args):
        raise KeyboardInterrupt
    monkeypatch.setattr(cli, "_cmd_calibrate", fail)
    assert cli.main(["calibrate", "missing", "--name", "x"]) == 130
    assert "cancelled" in capsys.readouterr().err
    def unexpected(args):
        raise TypeError("programmer bug")
    monkeypatch.setattr(cli, "_cmd_calibrate", unexpected)
    with pytest.raises(TypeError, match="programmer bug"):
        cli.main(["calibrate", "missing", "--name", "x"])


class LocalEndpoint:
    def __init__(self, response, status=200, redirect=None):
        self.hits = []
        endpoint = self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                endpoint.hits.append((self.command, self.path))
                self.send_response(status)
                if redirect:
                    self.send_header("Location", redirect)
                self.end_headers()
                self.wfile.write(response)
            def do_POST(self):
                endpoint.hits.append((self.command, self.path))
                self.send_response(500)
                self.end_headers()
            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.mark.parametrize("response,status", [(b"not json", 200), (b"{}", 200),
                                              (b'{"version": ""}', 200), (b"down", 503)])
def test_preflight_unavailable_skips_frames_once(response, status, caplog):
    endpoint = LocalEndpoint(response, status)
    cfg = replace(vlm.load_vlm_config(), host=endpoint.url)
    rounds = [replace(rr("spin", "X", round=i), needs_review=True) for i in range(3)]
    try:
        with vlm_session(cfg) as session:
            for row in rounds:
                [out] = vlm.review_low_confidence([row], lambda _: pytest.fail("decoded frames"),
                                                session=session)
                assert out.vlm_note.startswith(vlm.UNAVAILABLE_PREFIX)
                assert replace(out, vlm_note=None) == row
            assert not session.requests
        assert endpoint.hits == [("GET", "/api/version")]
        assert caplog.text.count("VLM unavailable") == 1
    finally:
        endpoint.close()


@contextmanager
def vlm_session(cfg):
    session = vlm.VlmSession(cfg)
    try:
        yield session
    finally:
        session.close()


def test_preflight_transport_no_proxy_no_redirect(monkeypatch):
    target = LocalEndpoint(b'{"version": ""}')
    redirect = LocalEndpoint(b"", 307, target.url + "/api/version")
    for key in ("http_proxy", "HTTP_PROXY", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(key, target.url)
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setenv("NO_PROXY", "")
    try:
        cfg = replace(vlm.load_vlm_config(), host=redirect.url)
        with vlm_session(cfg) as session:
            [out] = vlm.review_low_confidence([replace(rr("spin", "X"), needs_review=True)],
                                            lambda _: pytest.fail("decoded frames"), session=session)
            assert "redirect" in out.vlm_note
        assert redirect.hits == [("GET", "/api/version")]
        assert not target.hits
    finally:
        redirect.close()
        target.close()


def test_preflight_success_is_before_frames_without_loading(monkeypatch):
    cfg = vlm.load_vlm_config()
    endpoint = LocalEndpoint(json.dumps({"version": "0.40.7"}).encode())
    cfg = replace(cfg, host=endpoint.url)
    try:
        def frames(row):
            assert endpoint.hits == [("GET", "/api/version")]
            return []
        vlm.review_low_confidence([replace(rr("spin", "X"), needs_review=True)], frames, config=cfg)
        assert endpoint.hits == [("GET", "/api/version")]
    finally:
        endpoint.close()


def test_unavailable_persists_only_notes_and_keeps_reviewed_rows(tmp_path, monkeypatch):
    from bxjudge.config import load_thresholds
    from bxjudge.output import write_results
    from bxjudge.rules import load_rules

    rules = load_rules("config/rules.yaml")
    rows = [replace(rr("spin", "X", round=i), needs_review=True) for i in range(1, 4)]
    rows[-1].reviewed = True
    write_results(rows, tmp_path, rules)
    before = json.loads((tmp_path / "rounds.json").read_text())
    endpoint = LocalEndpoint(b"down", 503)
    monkeypatch.setattr(vlm, "event_frames", lambda *a, **k: pytest.fail("frames decoded"))
    try:
        summary = vlm.review_results_dir(tmp_path, calib_dir=tmp_path,
                    th=load_thresholds("config/thresholds.yaml"), rules=rules,
                    config=replace(vlm.load_vlm_config(), host=endpoint.url))
        after = json.loads((tmp_path / "rounds.json").read_text())
        assert summary == {"asked": 2, "answered": 0, "unavailable": 2, "models": {}}
        assert after[-1] == before[-1]
        for original, noted in zip(before[:2], after[:2]):
            assert noted["vlm_note"].startswith(vlm.UNAVAILABLE_PREFIX)
            assert {k: v for k, v in noted.items() if k != "vlm_note"} == {
                   k: v for k, v in original.items() if k != "vlm_note"}
        assert endpoint.hits == [("GET", "/api/version")]
    finally:
        endpoint.close()


def test_invalid_endpoint_rejected_before_network_or_frames(monkeypatch):
    monkeypatch.setattr(vlm, "_opener", lambda: pytest.fail("network before validation"))
    with pytest.raises(vlm.VlmConfigError):
        vlm.review_low_confidence([replace(rr("spin", "X"), needs_review=True)],
             lambda _: pytest.fail("frames before validation"),
             config=replace(vlm.load_vlm_config(), host="http://example.com:11435"))


def test_refused_endpoint_skips_frames(caplog):
    import socket

    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        # Bound but not listening: refuses connections without racing another process.
        host = f"http://127.0.0.1:{reserved.getsockname()[1]}"
        [out] = vlm.review_low_confidence([replace(rr("spin", "X"), needs_review=True)],
             lambda _: pytest.fail("frames decoded"),
             config=replace(vlm.load_vlm_config(), host=host))
    assert out.vlm_note.startswith(vlm.UNAVAILABLE_PREFIX)
    assert caplog.text.count("VLM unavailable") == 1


@pytest.mark.parametrize("command", ["eval", "tune", "label"])
@pytest.mark.parametrize("debug", [False, True])
def test_malformed_external_data_cli_boundary(tmp_path, command, debug):
    import yaml

    if command == "label":
        policy = yaml.safe_load(Path("config/labeling.yaml").read_text(encoding="utf-8"))
        policy["prompts"] = None
        config = tmp_path / "labeling.yaml"
        config.write_text(yaml.safe_dump(policy))
        argv = ["label", "sample", str(tmp_path), "--config", str(config)]
        expected = "prompts"
    else:
        (tmp_path / "rounds.json").write_text("null")
        argv = [command, str(tmp_path), "--report", str(tmp_path / "report.md")]
        expected = "rounds.json"
    process = subprocess.run([sys.executable, "-m", "bxjudge.cli",
                              *(["--debug"] if debug else []), *argv],
                             check=False, capture_output=True, text=True, env=os.environ.copy())
    assert process.returncode == (1 if debug else 2)
    assert ("Traceback" in process.stderr) is debug
    assert expected in process.stderr
    assert "ValueError" in process.stderr if debug else process.stderr.startswith("error:")
    assert not (tmp_path / "report.md").exists()
