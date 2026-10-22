"""Local VLM second opinion: loopback-only, note-only, complete-snapshot persistence."""

import base64
import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest

from bxjudge import output, vlm
from bxjudge.vlm import (
    VlmConfigError,
    VlmError,
    ask_vlm,
    question_for,
    review_low_confidence,
)
from tests.helpers import make_calibration, make_video, rr

PRIMARY, FALLBACK = "qwen3.8:27b", "qwen3-vl:8b"
TOUCH_Q = "發射時是否有發射器或手碰到盤面？是哪一側？"
FINAL_Q = "左方開始的陀螺最後停在哪個區域，還是先停止旋轉？"
AIR_Q = "兩顆陀螺落地前是否在空中相撞？"


def res(video, round, *, finish=None, winner=None, needs_review=False, reasons=(), **kw):
    r = rr(finish, winner, video=video, round=round, review_reasons=list(reasons), **kw)
    r.needs_review = needs_review
    return r


def blank():
    return np.zeros((72, 128, 3), np.uint8)


def reply(content, **change):
    """Exact Ollama 0.40.7 /api/chat non-streaming response shape."""
    data = {
        "model": "m",
        "created_at": "2026-10-22T00:00:00Z",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
        "total_duration": 1,
        "load_duration": 1,
        "prompt_eval_count": 1,
        "eval_count": 1,
    }
    data.update(change)
    return data


def cfg(**change):
    return replace(vlm.load_vlm_config(), **change)


# --------------------------------------------------------------------------- brief


def test_vlm_note_only(monkeypatch):
    monkeypatch.setattr(vlm, "_post", lambda *a, **k: reply('{"answer":"over","reason":"..."}'))
    r = res("a.mp4", 1, finish="spin", winner="X", needs_review=True)
    [out] = review_low_confidence([r], lambda _: [blank()])
    assert out.finish == "spin" and out.winner == "X" and "over" in out.vlm_note


def test_fallback_on_error(monkeypatch):
    calls = []

    def fake(host, payload, **_):
        calls.append(payload["model"])
        if payload["model"] == "qwen3.8:27b":
            raise VlmError("out of memory")
        return reply('{"answer":"no","reason":"x"}')

    monkeypatch.setattr(vlm, "_post", fake)
    assert ask_vlm([blank()], "q").model == "qwen3-vl:8b" and calls == ["qwen3.8:27b", "qwen3-vl:8b"]


# --------------------------------------------------------------------------- config


def test_default_config_is_isolated_local_runtime():
    c = vlm.load_vlm_config()
    assert c.host == "http://127.0.0.1:11435"
    assert (c.model, c.fallback) == (PRIMARY, FALLBACK)
    assert c.think is False and c.num_predict >= 512 and c.timeout_s >= 141
    assert 6 <= c.frames <= 10 and c.event_window_s == 1.0


@pytest.mark.parametrize(
    "change",
    [
        {"extra": 1},
        {"host": "http://192.168.1.5:11435"},
        {"host": "http://localhost:11435"},
        {"host": "https://ollama.com"},
        {"model": "gpt-oss:120b-cloud"},
        {"fallback": "qwen3:cloud"},
        {"frames": 0},
        {"frames": True},
        {"timeout_s": -1},
        {"think": "no"},
        {"after_run": 1},
        {"temperature": -0.5},
        {"temperature": "low"},
        {"num_predict": 1.5},
        {"keep_alive": None},
    ],
)
def test_invalid_config_fails_loudly(tmp_path, change):
    import yaml

    raw = yaml.safe_load(Path("config/vlm.yaml").read_text(encoding="utf-8"))
    raw.update(change)
    path = tmp_path / "vlm.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        vlm.load_vlm_config(path)


def test_missing_config_key_fails(tmp_path):
    import yaml

    raw = yaml.safe_load(Path("config/vlm.yaml").read_text(encoding="utf-8"))
    del raw["fallback"]
    path = tmp_path / "vlm.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="fallback"):
        vlm.load_vlm_config(path)


@pytest.mark.parametrize(
    "host", ["http://127.0.0.1:11435", "http://127.0.0.2:9", "http://[::1]:11435", "http://127.0.0.1:11435/"]
)
def test_loopback_literal_hosts_accepted(host):
    assert vlm.validate_endpoint(host).startswith("http://")


@pytest.mark.parametrize(
    "host",
    [
        "http://localhost:11435",  # a name can resolve anywhere
        "http://10.0.0.2:11435",
        "http://0.0.0.0:11435",
        "http://user:pw@127.0.0.1:11435",
        "http://127.0.0.1:11435/proxy?x=1",
        "file:///etc/passwd",
        "127.0.0.1:11435",
        "http://127.0.0.1.nip.io:11435",
    ],
)
def test_non_loopback_or_odd_endpoints_rejected(host):
    with pytest.raises(VlmConfigError):
        vlm.validate_endpoint(host)


@pytest.mark.parametrize("model", ["gpt-oss:120b-cloud", "qwen3:cloud", "deepseek-v3.1:671b-cloud", "x-cloud", ""])
def test_cloud_or_empty_model_tags_rejected(model):
    with pytest.raises(VlmConfigError):
        vlm.validate_model(model)


# --------------------------------------------------------------------------- ordering


def test_ask_vlm_validates_endpoint_before_touching_images(monkeypatch):
    encoded, posted = [], []
    monkeypatch.setattr(vlm, "_encode", lambda image: encoded.append(1) or "x")
    monkeypatch.setattr(vlm, "_post", lambda *a, **k: posted.append(1))

    class Exploding(list):
        def __iter__(self):
            raise AssertionError("images read before endpoint validation")

        def __len__(self):
            raise AssertionError("images read before endpoint validation")

    for kwargs in ({"host": "http://203.0.113.9:11434"}, {"model": "qwen3:cloud"}, {"fallback": "a-cloud"}):
        with pytest.raises(VlmConfigError):
            ask_vlm(Exploding([blank()]), "q", **kwargs)
    assert encoded == [] and posted == []


@pytest.mark.parametrize(
    "change", [{"host": "http://198.51.100.7:11435"}, {"model": "big:cloud"}, {"fallback": "small-cloud"}]
)
def test_review_validates_endpoint_before_any_frame_access(monkeypatch, change):
    seen = []
    monkeypatch.setattr(vlm, "_post", lambda *a, **k: seen.append("post"))
    r = res("a.mp4", 1, needs_review=True)
    with pytest.raises(VlmConfigError):
        review_low_confidence([r], lambda _: seen.append("frames") or [blank()], config=cfg(**change))
    assert seen == []


def test_post_revalidates_endpoint_itself():
    with pytest.raises(VlmConfigError):
        vlm._post("http://192.0.2.1:11434", {"model": PRIMARY}, timeout=1)


# --------------------------------------------------------------------------- HTTP boundary


class _Recorder:
    def __init__(self, handler):
        self.hits = []
        recorder = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                recorder.hits.append(self.path)
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                handler(self, body)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _send_json(h, data, code=200):
    payload = json.dumps(data).encode()
    h.send_response(code)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(payload)))
    h.end_headers()
    h.wfile.write(payload)


def test_http_redirects_are_refused():
    target = _Recorder(lambda h, body: _send_json(h, reply('{"answer":"a","reason":"r"}')))

    def redirect(h, body):
        h.send_response(307)
        h.send_header("Location", target.url + "/api/chat")
        h.send_header("Content-Length", "0")
        h.end_headers()

    source = _Recorder(redirect)
    try:
        with pytest.raises(VlmError, match="redirect"):
            vlm._post(source.url, {"model": PRIMARY}, timeout=5)
        assert source.hits == ["/api/chat"] and target.hits == []
    finally:
        source.close()
        target.close()


def test_environment_proxy_is_never_used(monkeypatch):
    proxy = _Recorder(lambda h, body: _send_json(h, {"error": "proxied"}, 502))
    direct = _Recorder(lambda h, body: _send_json(h, reply('{"answer":"a","reason":"r"}')))
    for name in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.setenv(name, proxy.url)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    try:
        assert vlm._post(direct.url, {"model": PRIMARY}, timeout=5)["done"] is True
        assert proxy.hits == [] and direct.hits == ["/api/chat"]
    finally:
        proxy.close()
        direct.close()


def test_http_error_body_becomes_vlm_error_and_real_fallback(monkeypatch):
    def handler(h, body):
        model = json.loads(body)["model"]
        if model == "missing:tag":
            _send_json(h, {"error": "model 'missing:tag' not found"}, 404)
        else:
            _send_json(h, reply('{"answer":"yes","reason":"seen"}', model=model))

    server = _Recorder(handler)
    try:
        answer = ask_vlm([blank()], "q", model="missing:tag", host=server.url)
        assert answer.model == FALLBACK and answer.answer == "yes"
        with pytest.raises(VlmError, match="404"):
            vlm._post(server.url, {"model": "missing:tag"}, timeout=5)
    finally:
        server.close()


def test_connection_refused_is_vlm_error():
    server = _Recorder(lambda h, body: None)
    url = server.url
    server.close()
    with pytest.raises(VlmError):
        vlm._post(url, {"model": PRIMARY}, timeout=2)


# --------------------------------------------------------------------------- response contract


@pytest.mark.parametrize(
    "bad",
    [
        reply("", done_reason="length"),  # the measured 8b failure
        reply('{"answer":"over","reason":"r"}', done_reason="length"),
        reply('{"answer":"over","reason":"r"}', done=False),
        {"message": {"content": '{"answer":"over","reason":"r"}'}},  # no completion metadata
        reply('{"answer":"over","reason":"r"}', done_reason=None),
        reply(""),
        reply("   "),
        reply('{"answer":"over"'),
        reply('["over"]'),
        reply('{"answer":"over"}'),
        reply('{"answer":"","reason":"r"}'),
        reply('{"answer":"over","reason":7}'),
        {"error": "out of memory"},
        "not a dict",
    ],
)
def test_truncated_empty_or_malformed_reply_falls_back(monkeypatch, bad):
    calls = []

    def fake(host, payload, **_):
        calls.append(payload["model"])
        return bad if payload["model"] == PRIMARY else reply('{"answer":"no","reason":"x"}')

    monkeypatch.setattr(vlm, "_post", fake)
    answer = ask_vlm([blank()], "q")
    assert (answer.model, answer.answer, answer.reason) == (FALLBACK, "no", "x")
    assert calls == [PRIMARY, FALLBACK]


def test_both_models_failing_raises_vlm_error(monkeypatch):
    monkeypatch.setattr(vlm, "_post", lambda *a, **k: reply("", done_reason="length"))
    with pytest.raises(VlmError, match=PRIMARY):
        ask_vlm([blank()], "q")


def test_payload_is_local_structured_request(monkeypatch):
    seen = []
    monkeypatch.setattr(vlm, "_post", lambda host, payload, **k: seen.append((host, payload, k)) or reply('{"answer":"a","reason":"r"}'))
    image = blank()
    image[10:20, 10:20] = (0, 0, 255)
    c = vlm.load_vlm_config()
    ask_vlm([image, blank()], "問題？")
    host, payload, kwargs = seen[0]
    assert host == c.host and kwargs["timeout"] == c.timeout_s
    assert kwargs.get("path", "/api/chat") == "/api/chat"
    assert payload["model"] == PRIMARY and payload["stream"] is False
    assert payload["think"] is False and payload["keep_alive"] == c.keep_alive
    # temperature null = the model's own sampling (greedy decoding looped the 8b thinking build)
    assert c.temperature is None
    assert payload["options"] == {"num_ctx": c.num_ctx, "num_predict": c.num_predict}
    ask_vlm([image], "q", config=replace(c, temperature=0.2))
    assert seen[-1][1]["options"]["temperature"] == 0.2
    schema = payload["format"]
    assert schema["type"] == "object" and set(schema["required"]) == {"answer", "reason"}
    [message] = payload["messages"]
    assert message["role"] == "user" and "問題？" in message["content"]
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(message["images"][0]), np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(decoded, image) and len(message["images"]) == 2


def test_errors_and_logs_never_contain_media(monkeypatch, caplog):
    image = np.random.default_rng(0).integers(0, 256, (40, 40, 3), dtype=np.uint8)
    b64 = vlm._encode(image)

    def fake(host, payload, **_):
        raise VlmError("HTTP 500: boom")

    monkeypatch.setattr(vlm, "_post", fake)
    caplog.set_level("DEBUG")
    r = res("a.mp4", 1, needs_review=True)
    [out] = review_low_confidence([r], lambda _: [image])
    assert out.vlm_note.startswith(vlm.UNAVAILABLE_PREFIX)
    assert b64[:64] not in out.vlm_note and b64[:64] not in caplog.text


def test_empty_image_list_rejected():
    with pytest.raises(ValueError):
        ask_vlm([], "q")


# --------------------------------------------------------------------------- questions


def test_question_texts_by_reason():
    touch = res("a.mp4", 1, needs_review=True, reasons=["shake_cause_unknown@12"])
    swap = res("a.mp4", 1, needs_review=True, reasons=["id_swap_risk"])
    low = res("a.mp4", 1, needs_review=True)
    low.confidence = 0.2
    air = res("a.mp4", 1, needs_review=True, air="maybe")
    assert question_for(touch) == TOUCH_Q
    assert question_for(swap) == FINAL_Q
    assert question_for(low) == FINAL_Q
    assert question_for(air) == AIR_Q


def test_question_priority_and_informational_notes():
    both = res("a.mp4", 1, needs_review=True, air="maybe", reasons=["id_swap_risk", "shake_cause_unknown@3"])
    assert question_for(both) == TOUCH_Q
    swap_air = res("a.mp4", 1, needs_review=True, air="maybe", reasons=["id_swap_risk"])
    assert question_for(swap_air) == FINAL_Q
    after = res("a.mp4", 1, needs_review=True, reasons=["after_finish:id_swap_risk"])
    assert question_for(after) == FINAL_Q  # default question, not the swap trigger
    no_launch = res("a.mp4", 1, needs_review=True, launch=False)
    assert question_for(no_launch) == FINAL_Q


def test_event_time_uses_playback_seconds():
    touch = res("a.mp4", 1, needs_review=True, t_launch=2.0, reasons=["shake_cause_unknown@90", "shake_cause_unknown@75"])
    touch.motion_evidence = {"fps": 30.0}
    assert vlm.event_time(touch) == pytest.approx(2.5)  # earliest frame / playback fps, never x8
    touch.motion_evidence = None
    assert vlm.event_time(touch) == 2.0
    air = res("a.mp4", 1, needs_review=True, t_launch=1.0, air="maybe")
    air.launch.landing_frame = {"X": 60, "B": 45}
    air.motion_evidence = {"fps": 30.0}
    assert vlm.event_time(air) == pytest.approx(1.5)
    final = res("a.mp4", 1, needs_review=True, t_launch=1.0)
    assert vlm.event_time(final) == final.t_end


# --------------------------------------------------------------------------- selection


def test_reviewed_non_round_and_confident_rows_are_skipped(monkeypatch):
    monkeypatch.setattr(vlm, "_post", lambda *a, **k: reply('{"answer":"a","reason":"r"}'))
    reviewed = res("a.mp4", 1, needs_review=True, finish="spin", winner="X")
    reviewed.reviewed = True
    reviewed.reviewer_fix = {"winner": "B"}
    placeholders = [
        res("a.mp4", n, needs_review=True, reasons=[reason], launch=False)
        for n, reason in enumerate(["recalibrate", "no_round", "decode_error", "analysis_error"], 2)
    ]
    confident = res("a.mp4", 6, finish="over", winner="B")
    noted = res("a.mp4", 7, needs_review=True)
    noted.vlm_note = "[qwen3-vl:8b] earlier answer"
    retry = res("a.mp4", 8, needs_review=True)
    retry.vlm_note = vlm.UNAVAILABLE_PREFIX + "connection refused"
    todo = res("a.mp4", 9, needs_review=True)
    rows = [reviewed, *placeholders, confident, noted, retry, todo]
    asked = []
    out = review_low_confidence(rows, lambda r: asked.append(r.round) or [blank()])
    assert asked == [8, 9]
    for before, after in zip(rows[:-2], out[:-2]):
        assert after == before
    assert out[-1].vlm_note.startswith("[") and "a" in out[-1].vlm_note
    assert not out[-2].vlm_note.startswith(vlm.UNAVAILABLE_PREFIX)
    assert todo.vlm_note is None  # inputs are not mutated


def test_note_never_touches_judgment_fields(monkeypatch):
    monkeypatch.setattr(vlm, "_post", lambda *a, **k: reply('{"answer":"B wins by xtreme","reason":"r"}'))
    r = res("a.mp4", 1, finish="spin", winner="X", needs_review=True, reasons=["id_swap_risk"])
    r.points, r.confidence = 1, 0.3
    r.reviewer_fix = {"x_land_zone": "lid"}
    [out] = review_low_confidence([r], lambda _: [blank()])
    assert replace(out, vlm_note=None) == r
    assert FINAL_Q in out.vlm_note and FALLBACK not in out.vlm_note and PRIMARY in out.vlm_note


def test_missing_or_broken_frames_give_unavailable_note_and_continue(monkeypatch):
    monkeypatch.setattr(vlm, "_post", lambda *a, **k: reply('{"answer":"a","reason":"r"}'))
    from bxjudge.video import VideoDecodeError

    def frames(r):
        if r.round == 1:
            raise VideoDecodeError("truncated frame 7")
        if r.round == 2:
            return []
        if r.round == 3:
            raise FileNotFoundError("/videos/gone.mp4")
        return [blank()]

    rows = [res("a.mp4", n, needs_review=True) for n in (1, 2, 3, 4)]
    out = review_low_confidence(rows, frames)
    assert [o.vlm_note.startswith(vlm.UNAVAILABLE_PREFIX) for o in out] == [True, True, True, False]
    assert "truncated frame 7" in out[0].vlm_note and "gone.mp4" in out[2].vlm_note


def test_models_released_at_phase_end_and_on_error(monkeypatch):
    calls = []

    def fake(host, payload, path="/api/chat", **_):
        calls.append((path, payload["model"], payload.get("keep_alive")))
        if path == "/api/generate":
            return {"model": payload["model"], "done": True, "done_reason": "unload"}
        return reply('{"answer":"a","reason":"r"}')

    monkeypatch.setattr(vlm, "_post", fake)
    review_low_confidence([res("a.mp4", 1, needs_review=True)], lambda _: [blank()])
    assert calls[-2:] == [("/api/generate", PRIMARY, 0), ("/api/generate", FALLBACK, 0)]

    calls.clear()

    def interrupted(r):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        review_low_confidence(
            [res("a.mp4", 1, needs_review=True), res("a.mp4", 2, needs_review=True)],
            lambda r: [blank()] if r.round == 1 else interrupted(r),
        )
    assert calls[-2:] == [("/api/generate", PRIMARY, 0), ("/api/generate", FALLBACK, 0)]

    calls.clear()
    review_low_confidence([res("a.mp4", 1, finish="spin", winner="X")], lambda _: [blank()])
    assert calls == []  # nothing asked, nothing loaded, nothing to release


def test_release_failure_does_not_mask_results(monkeypatch):
    def fake(host, payload, path="/api/chat", **_):
        if path == "/api/generate":
            raise VlmError("HTTP 404: model not found")
        return reply('{"answer":"a","reason":"r"}')

    monkeypatch.setattr(vlm, "_post", fake)
    [out] = review_low_confidence([res("a.mp4", 1, needs_review=True)], lambda _: [blank()])
    assert "a" in out.vlm_note


# --------------------------------------------------------------------------- fix round 1


def _chat_models(calls):
    return [model for path, model in calls if path == "/api/chat"]


@pytest.mark.parametrize(
    "failure",
    [
        VlmError("HTTP 500: model requires more system memory (cuda OOM)"),
        VlmError("HTTP 404: model 'qwen3.8:27b' not found"),
        VlmError("request to http://127.0.0.1:11435/api/chat failed: timed out"),
        VlmError("server error: out of memory"),
    ],
)
def test_service_failure_makes_fallback_sticky_for_the_batch(monkeypatch, failure):
    calls = []

    def fake(host, payload, path="/api/chat", **_):
        calls.append((path, payload["model"]))
        if path == "/api/generate":
            return {"model": payload["model"], "done": True, "done_reason": "unload"}
        if payload["model"] == PRIMARY:
            raise failure
        return reply('{"answer":"a","reason":"r"}')

    monkeypatch.setattr(vlm, "_post", fake)
    rows = [res("a.mp4", n, needs_review=True) for n in range(1, 6)]
    out = review_low_confidence(rows, lambda r: [blank()])
    assert _chat_models(calls) == [PRIMARY] + [FALLBACK] * 5
    assert all(o.vlm_note.startswith(f"[{FALLBACK}]") for o in out)
    # the release at phase end still covers both models
    assert calls[-2:] == [("/api/generate", PRIMARY), ("/api/generate", FALLBACK)]


def test_sticky_fallback_spans_videos_in_one_results_dir_phase(tmp_path, monkeypatch, rules, th):
    source = tmp_path / "a.mp4"
    source.write_bytes(b"")
    out = _results(tmp_path, rules, source.resolve())
    record = json.loads((out / "videos.json").read_text())
    record["b.mp4"]["source_video"] = str(source.resolve())
    (out / "videos.json").write_text(json.dumps(record))
    calls = []

    def fake(host, payload, path="/api/chat", **_):
        calls.append((path, payload["model"]))
        if path == "/api/chat" and payload["model"] == PRIMARY:
            raise VlmError("HTTP 500: cuda OOM")
        return reply('{"answer":"a","reason":"r"}')

    monkeypatch.setattr(vlm, "_post", fake)
    monkeypatch.setattr(vlm, "event_frames", lambda *a, **k: [blank()])
    summary = vlm.review_results_dir(out, calib_dir=tmp_path, th=th, rules=rules)
    assert _chat_models(calls) == [PRIMARY, FALLBACK, FALLBACK, FALLBACK]
    assert summary["models"] == {FALLBACK: 3}


def test_single_bad_reply_does_not_make_fallback_sticky(monkeypatch):
    calls = []
    replies = iter([reply("", done_reason="length")])

    def fake(host, payload, path="/api/chat", **_):
        calls.append((path, payload["model"]))
        if path == "/api/chat" and payload["model"] == PRIMARY:
            return next(replies, reply('{"answer":"p","reason":"r"}'))
        return reply('{"answer":"f","reason":"r"}')

    monkeypatch.setattr(vlm, "_post", fake)
    rows = [res("a.mp4", n, needs_review=True) for n in (1, 2, 3)]
    out = review_low_confidence(rows, lambda r: [blank()])
    assert _chat_models(calls) == [PRIMARY, FALLBACK, PRIMARY, PRIMARY]
    assert [o.vlm_note.split("]")[0] for o in out] == [f"[{FALLBACK}", f"[{PRIMARY}", f"[{PRIMARY}"]


def test_sticky_session_with_failing_fallback_notes_each_row(monkeypatch):
    calls = []

    def fake(host, payload, path="/api/chat", **_):
        calls.append((path, payload["model"]))
        raise VlmError("request failed: connection refused")

    monkeypatch.setattr(vlm, "_post", fake)
    rows = [res("a.mp4", n, needs_review=True) for n in (1, 2, 3)]
    out = review_low_confidence(rows, lambda r: [blank()])
    assert _chat_models(calls) == [PRIMARY, FALLBACK, FALLBACK, FALLBACK]
    assert all(o.vlm_note.startswith(vlm.UNAVAILABLE_PREFIX) for o in out)
    assert PRIMARY in out[1].vlm_note  # notes still say why the primary is skipped


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((8, 8, 3), np.float32),
        np.zeros((0, 0, 3), np.uint8),
        np.zeros((2, 8, 8, 3), np.uint8),
        None,
        "frame.png",
    ],
    ids=["float32", "empty", "4d", "none", "str"],
)
def test_unencodable_frame_is_a_row_note_not_a_batch_abort(monkeypatch, bad):
    calls = []
    monkeypatch.setattr(
        vlm, "_post", lambda host, payload, **k: calls.append(payload["model"]) or reply('{"answer":"a","reason":"r"}')
    )
    rows = [res("a.mp4", n, needs_review=True) for n in (1, 2)]
    out = review_low_confidence(rows, lambda r: [blank(), bad] if r.round == 1 else [blank()])
    assert out[0].vlm_note.startswith(vlm.UNAVAILABLE_PREFIX + "frames: ")
    assert out[1].vlm_note.startswith(f"[{PRIMARY}]")
    assert calls == [PRIMARY, PRIMARY, FALLBACK]  # row 2 only, then the two releases


def test_ask_vlm_rejects_unencodable_frame_before_any_request(monkeypatch):
    monkeypatch.setattr(vlm, "_post", lambda *a, **k: pytest.fail("request with a bad frame"))
    with pytest.raises(vlm.VlmFramesError):
        ask_vlm([np.zeros((4, 4), np.float64)], "q")


# --------------------------------------------------------------------------- frames


def test_event_frames_decode_playback_window_with_scaled_zone_overlay(tmp_path, monkeypatch, th):
    video = make_video(tmp_path / "v.mp4", seconds=3, fps=30, size=(320, 180))
    c = make_calibration()  # 640x360 calibration -> 320x180 decoded
    seen = []
    monkeypatch.setattr(vlm, "find_calibration", lambda frame, d, s: seen.append(frame.shape) or c)
    windows = []
    real = vlm.iter_frames

    def recording(path, start=0.0, end=None, **kw):
        windows.append((start, end, kw))
        yield from real(path, start, end, **kw)

    monkeypatch.setattr(vlm, "iter_frames", recording)
    config = cfg(frames=5, event_window_s=0.5)
    frames = vlm.event_frames(video, 1.5, tmp_path, config, th)
    assert len(frames) == 5
    assert all(f.shape == (180, 320, 3) for f in frames)
    assert seen and seen[0][:2] == (180, 320)
    # calibration chosen on the first-second frame, then one bounded event window
    assert windows[0][0] == 0.0
    start, end, kw = windows[1]
    assert start == pytest.approx(1.0) and end == pytest.approx(2.0 + 1 / 30)
    assert kw["long_edge"] == config.image_long_edge
    # battle_area of the 640x360 calibration is (120,60)-(520,280): at 320x180 its
    # top edge runs along y=30 between x=60 and x=260 and is drawn in green.
    edge = frames[0][29:32, 100:200].reshape(-1, 3).astype(int)
    assert (edge[:, 1] - edge[:, 2] > 60).any()


def test_event_frames_without_matching_calibration_is_unavailable(tmp_path, monkeypatch, th):
    video = make_video(tmp_path / "v.mp4", seconds=1, fps=30, size=(320, 180))
    monkeypatch.setattr(vlm, "find_calibration", lambda *a: None)
    with pytest.raises(vlm.VlmFramesError, match="calibration"):
        vlm.event_frames(video, 0.5, tmp_path, cfg(), th)


def test_event_frames_reuse_cached_calibration(tmp_path, monkeypatch, th):
    video = make_video(tmp_path / "v.mp4", seconds=2, fps=30, size=(320, 180))
    calls = []
    monkeypatch.setattr(vlm, "find_calibration", lambda *a: calls.append(1) or make_calibration())
    cache = {}
    vlm.event_frames(video, 0.5, tmp_path, cfg(frames=3), th, calibrations=cache)
    vlm.event_frames(video, 1.5, tmp_path, cfg(frames=3), th, calibrations=cache)
    assert calls == [1]


# --------------------------------------------------------------------------- results directory


def _results(tmp_path, rules, source):
    out = tmp_path / "out"
    a1 = res("a.mp4", 1, finish="spin", winner="X", needs_review=True, reasons=["id_swap_risk"])
    a1.motion_evidence = {"fps": 30.0, "window": [0, 3]}
    a1.detector_provenance = {"kind": "yolo26", "dataset": {"provenance": {"synthetic": True}}}
    a2 = res("a.mp4", 2, finish="over", winner="B", t_launch=20.0)
    a3 = res("a.mp4", 3, finish="spin", winner="X", needs_review=True, t_launch=40.0)
    a3.reviewer_fix = {"winner": "B"}  # unreviewed draft fix survives reruns
    a4 = res("a.mp4", 4, finish="burst", winner="X", needs_review=True, t_launch=60.0)
    b1 = res("b.mp4", 1, finish="xtreme", winner="B", needs_review=True)
    c1 = res("c.mp4", 1, needs_review=True, reasons=["recalibrate"], launch=False)
    output.write_results([a1, a2, a3, a4], out, rules)
    output.write_results([b1], out, rules)
    output.write_results([c1], out, rules)
    rows = json.loads((out / "rounds.json").read_text())
    rows[3]["reviewed"] = True  # a.mp4 round 4: human truth
    rows[3]["reviewer_fix"] = {"finish": "spin", "winner": "B", "points": 1}
    (out / "rounds.json").write_text(json.dumps(rows))
    output.write_results([], out, rules)  # refresh CSV/matches exports after the review
    for video, src in (("a.mp4", source), ("b.mp4", None), ("c.mp4", source)):
        output.record_video_timing(out, video, 10.0, 2.5, source_video=src)
    return out


def test_results_dir_writes_notes_through_complete_snapshots(tmp_path, monkeypatch, rules, th):
    source = tmp_path / "a.mp4"
    source.write_bytes(b"")
    out = _results(tmp_path, rules, source.resolve())
    before = json.loads((out / "rounds.json").read_text())
    matches_before = (out / "matches.csv").read_text()
    monkeypatch.setattr(vlm, "_post", lambda *a, **k: reply('{"answer":"lid-answer-7","reason":"r"}'))
    asked = []

    def frames(path, t, calib_dir, config, th_, **kw):
        asked.append((Path(path), t))
        return [blank()]

    monkeypatch.setattr(vlm, "event_frames", frames)
    summary = vlm.review_results_dir(out, calib_dir=tmp_path, th=th, rules=rules)
    after = json.loads((out / "rounds.json").read_text())
    assert [(r["video"], r["round"]) for r in after] == [(r["video"], r["round"]) for r in before]
    changed = {(r["video"], r["round"]) for r, o in zip(after, before) if r != o}
    assert changed == {("a.mp4", 1), ("a.mp4", 3), ("b.mp4", 1)}
    for r, o in zip(after, before):
        assert {k: v for k, v in r.items() if k != "vlm_note"} == {k: v for k, v in o.items() if k != "vlm_note"}
    notes = {(r["video"], r["round"]): r["vlm_note"] for r in after}
    assert "lid-answer-7" in notes[("a.mp4", 1)] and FINAL_Q in notes[("a.mp4", 1)]
    assert notes[("b.mp4", 1)].startswith(vlm.UNAVAILABLE_PREFIX) and "source_video" in notes[("b.mp4", 1)]
    assert notes[("a.mp4", 4)] is None and notes[("c.mp4", 1)] is None and notes[("a.mp4", 2)] is None
    assert after[2]["reviewer_fix"] == {"winner": "B"}
    assert (out / "matches.csv").read_text() == matches_before
    assert "lid-answer-7" in (out / "rounds.csv").read_text()
    assert asked == [(source.resolve(), before[0]["t_end"]), (source.resolve(), before[2]["t_end"])]
    timing = json.loads((out / "videos.json").read_text())
    assert timing["a.mp4"]["processing_s"] == 2.5 and timing["a.mp4"]["vlm_s"] >= 0
    assert "vlm_s" in timing["b.mp4"] and "vlm_s" not in timing["c.mp4"]
    assert summary["answered"] == 2 and summary["unavailable"] == 1


def test_results_dir_relative_or_missing_source_is_unavailable(tmp_path, monkeypatch, rules, th):
    out = _results(tmp_path, rules, Path("relative/a.mp4"))
    monkeypatch.setattr(vlm, "_post", lambda *a, **k: reply('{"answer":"x","reason":"r"}'))
    monkeypatch.setattr(vlm, "event_frames", lambda *a, **k: pytest.fail("must not guess paths"))
    vlm.review_results_dir(out, calib_dir=tmp_path, th=th, rules=rules)
    notes = [r["vlm_note"] for r in json.loads((out / "rounds.json").read_text()) if r["vlm_note"]]
    assert notes and all(n.startswith(vlm.UNAVAILABLE_PREFIX) for n in notes)


def test_results_dir_rerun_only_retries_unavailable_and_respects_concurrent_review(
    tmp_path, monkeypatch, rules, th
):
    source = tmp_path / "a.mp4"
    source.write_bytes(b"")
    out = _results(tmp_path, rules, source.resolve())
    monkeypatch.setattr(vlm, "_post", lambda *a, **k: reply('{"answer":"first","reason":"r"}'))
    monkeypatch.setattr(vlm, "event_frames", lambda *a, **k: [blank()])
    vlm.review_results_dir(out, calib_dir=tmp_path, th=th, rules=rules, videos={"a.mp4"})
    rows = json.loads((out / "rounds.json").read_text())
    assert rows[4]["vlm_note"] is None  # b.mp4 outside the requested videos
    asked = []

    def frames_then_human_review(path, t, *a, **k):
        asked.append(t)
        rows = json.loads((out / "rounds.json").read_text())
        rows[4]["reviewed"] = True  # reviewer confirms b.mp4 meanwhile
        rows[4]["reviewer_fix"] = {"winner": "X"}
        (out / "rounds.json").write_text(json.dumps(rows))
        return [blank()]

    monkeypatch.setattr(vlm, "event_frames", frames_then_human_review)
    record = {k: v for k, v in json.loads((out / "videos.json").read_text()).items()}
    record["b.mp4"]["source_video"] = str(source.resolve())
    (out / "videos.json").write_text(json.dumps(record))
    vlm.review_results_dir(out, calib_dir=tmp_path, th=th, rules=rules)
    rows = json.loads((out / "rounds.json").read_text())
    assert len(asked) == 1  # a.mp4 rows already have notes
    assert rows[4]["reviewed"] is True and rows[4]["reviewer_fix"] == {"winner": "X"}
    assert rows[4]["vlm_note"] is None  # protected reviewed row left exactly as the reviewer saved it
    assert "first" in rows[0]["vlm_note"]


def test_results_dir_validates_endpoint_before_reading_frames(tmp_path, monkeypatch, rules, th):
    source = tmp_path / "a.mp4"
    source.write_bytes(b"")
    out = _results(tmp_path, rules, source.resolve())
    monkeypatch.setattr(vlm, "event_frames", lambda *a, **k: pytest.fail("frames before validation"))
    monkeypatch.setattr(vlm, "_post", lambda *a, **k: pytest.fail("request before validation"))
    with pytest.raises(VlmConfigError):
        vlm.review_results_dir(out, calib_dir=tmp_path, th=th, rules=rules, config=cfg(host="http://192.0.2.4:1"))


def test_results_dir_releases_models_once_per_phase(tmp_path, monkeypatch, rules, th):
    source = tmp_path / "a.mp4"
    source.write_bytes(b"")
    out = _results(tmp_path, rules, source.resolve())
    record = json.loads((out / "videos.json").read_text())
    record["b.mp4"]["source_video"] = str(source.resolve())
    (out / "videos.json").write_text(json.dumps(record))
    paths = []

    def fake(host, payload, path="/api/chat", **_):
        paths.append(path)
        return reply('{"answer":"a","reason":"r"}')

    monkeypatch.setattr(vlm, "_post", fake)
    monkeypatch.setattr(vlm, "event_frames", lambda *a, **k: [blank()])
    vlm.review_results_dir(out, calib_dir=tmp_path, th=th, rules=rules)
    assert paths == ["/api/chat"] * 3 + ["/api/generate"] * 2


# --------------------------------------------------------------------------- CLI


def test_vlm_cli_runs_results_dir_and_reports_errors(tmp_path, monkeypatch, capsys):
    from bxjudge.cli import main

    calls = []
    monkeypatch.setattr(
        vlm,
        "review_results_dir",
        lambda out, **kw: calls.append((out, kw)) or {"asked": 1, "answered": 1, "unavailable": 0, "models": {}},
    )
    assert main(["vlm", str(tmp_path), "--calib", "calib"]) == 0
    out, kw = calls[0]
    assert out == tmp_path and kw["calib_dir"] == Path("calib")
    assert kw["config"].host == "http://127.0.0.1:11435"
    assert "answered" in capsys.readouterr().out

    def fail(out, **kw):
        raise VlmConfigError("non-loopback host")

    monkeypatch.setattr(vlm, "review_results_dir", fail)
    assert main(["vlm", str(tmp_path)]) == 2
    assert "non-loopback" in capsys.readouterr().err
    assert main(["vlm", str(tmp_path / "missing")]) == 2
