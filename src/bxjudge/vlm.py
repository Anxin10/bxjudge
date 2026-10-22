"""Local VLM second opinion for low-confidence rounds (spec section 8).

Only unreviewed, real ``needs_review`` rounds are asked, and the answer only ever lands
in ``vlm_note``: finish / winner / points / reviewer fixes are never touched. Frames
never leave this machine: the endpoint must be a literal loopback address (checked
before any frame is read, encoded or sent), cloud model tags are refused, and HTTP
proxies and redirects are disabled. Defaults live in ``config/vlm.yaml`` (R26/R40).
"""

from __future__ import annotations

import base64
import http.client
import ipaddress
import json
import logging
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass, fields, replace
from pathlib import Path

import cv2
import numpy as np
import yaml

from .calibration import ZONES, find_calibration, scale_calibration
from .config import Thresholds
from .manifest import probe
from .output import record_vlm_timing, row_to_result, write_results
from .rules import Rules, is_not_a_round
from .types import Calibration, RoundResult
from .video import iter_frames

log = logging.getLogger(__name__)

UNAVAILABLE_PREFIX = "vlm_unavailable: "

TOUCH_QUESTION = "發射時是否有發射器或手碰到盤面？是哪一側？"
FINAL_QUESTION = "左方開始的陀螺最後停在哪個區域，還是先停止旋轉？"
AIR_QUESTION = "兩顆陀螺落地前是否在空中相撞？"

_PROMPT = (
    "你是 Beyblade X 比賽影片的覆核助手。以下圖片依時間順序取自同一回合事件前後各約 "
    "{window:g} 秒（慢動作影片的播放時間，右下角為時間）。綠色框線與英文標籤是校準區域："
    "stadium 整個場地、lid 上盤蓋子、battle_area 下盤戰鬥區、over_left / over_right 擊飛區域、"
    "xtreme 極限區域、launch_left / launch_right 左右發射區。畫面左方為 X 方、右方為 B 方。"
    "只根據畫面回答；看不清楚就回答「無法判斷」。以 JSON 回覆：answer 為簡短答案，"
    "reason 為簡短理由，使用繁體中文。\n問題：{question}"
)
_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "reason": {"type": "string"}},
    "required": ["answer", "reason"],
}
_ZONE_COLOR = (80, 200, 80)  # BGR, same as the trajectory plots
_TIME_COLOR = (0, 255, 255)
_DETAIL_CHARS = 300


class VlmError(RuntimeError):
    """A VLM request failed or returned an incomplete / invalid reply."""


class VlmReplyError(VlmError):
    """One reply was truncated, empty or malformed (the service itself answered)."""


class VlmConfigError(ValueError):
    """The VLM configuration would send frames off this machine or is invalid."""


class VlmFramesError(RuntimeError):
    """The event frames for a round cannot be produced."""


@dataclass(frozen=True)
class VlmConfig:
    host: str
    model: str
    fallback: str
    after_run: bool
    timeout_s: float
    keep_alive: str | int
    think: bool
    num_ctx: int
    num_predict: int
    temperature: float | None  # None: the model's own sampling defaults
    event_window_s: float
    frames: int
    image_long_edge: int


@dataclass
class VlmAnswer:
    answer: str
    reason: str
    model: str


# --------------------------------------------------------------------------- configuration


def validate_endpoint(host: str) -> str:
    """Return the normalized base URL if ``host`` is a literal loopback HTTP endpoint.

    Host names are refused (they can resolve anywhere), as are credentials, paths,
    queries and fragments.
    """
    if not isinstance(host, str):
        raise VlmConfigError(f"VLM host must be a string, got {host!r}")
    parts = urllib.parse.urlsplit(host)
    if (
        parts.scheme not in ("http", "https")
        or parts.username is not None
        or parts.password is not None
        or parts.path not in ("", "/")
        or parts.query
        or parts.fragment
    ):
        raise VlmConfigError(f"VLM host must be http://<loopback address>:<port>, got {host!r}")
    try:
        address = ipaddress.ip_address(parts.hostname or "")
        parts.port  # noqa: B018 - raises ValueError on an invalid port
    except ValueError:
        raise VlmConfigError(
            f"VLM host must be a literal loopback address, not a name: {host!r}"
        ) from None
    if not address.is_loopback:
        raise VlmConfigError(f"VLM host {host!r} is not loopback; frames must stay local")
    return f"{parts.scheme}://{parts.netloc}"


def validate_model(name: str) -> str:
    """Return ``name`` unless it is empty or a cloud model tag."""
    if not isinstance(name, str) or not name or any(ch.isspace() for ch in name):
        raise VlmConfigError(f"VLM model must be a non-empty local tag, got {name!r}")
    if "cloud" in re.split(r"[:/._-]", name.lower()):
        raise VlmConfigError(f"cloud model tag {name!r} is not allowed (local models only)")
    return name


def _validated(cfg: VlmConfig) -> VlmConfig:
    """Check endpoint and both models; this runs before any frame is touched."""
    host = validate_endpoint(cfg.host)
    validate_model(cfg.model)
    validate_model(cfg.fallback)
    return replace(cfg, host=host)


def load_vlm_config(path: str | Path = "config/vlm.yaml") -> VlmConfig:
    """Load and validate the VLM config; unknown/missing keys or bad values raise ``ValueError``."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping")  # noqa: TRY004 - config errors are ValueError like load_thresholds
    names = {f.name for f in fields(VlmConfig)}
    unknown, missing = sorted(set(raw) - names, key=str), sorted(names - set(raw))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    values = dict(raw)
    for name in ("after_run", "think"):
        if type(values[name]) is not bool:
            raise ValueError(f"{path}: {name} must be true/false, got {values[name]!r}")
    for name in ("num_ctx", "num_predict", "frames", "image_long_edge"):
        if type(values[name]) is not int or values[name] <= 0:
            raise ValueError(f"{path}: {name} must be a positive int, got {values[name]!r}")
    for name, minimum in (("timeout_s", 0.0), ("event_window_s", 0.0), ("temperature", None)):
        v = values[name]
        if name == "temperature" and v is None:
            continue  # null: the model's own sampling defaults
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{path}: {name} must be a number, got {v!r}")
        if (minimum is not None and v <= minimum) or v < 0:
            raise ValueError(f"{path}: {name} out of range: {v!r}")
        values[name] = float(v)
    keep = values["keep_alive"]
    if not ((isinstance(keep, str) and keep) or (type(keep) is int and keep >= 0)):
        raise ValueError(f"{path}: keep_alive must be a duration string or seconds, got {keep!r}")
    return _validated(VlmConfig(**values))


# --------------------------------------------------------------------------- HTTP


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        fp.close()
        raise VlmError(f"refusing HTTP redirect {code} from the local VLM endpoint")


def _opener() -> urllib.request.OpenerDirector:
    # No proxies (an environment proxy would carry frames off-host) and no redirects.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def _short(text) -> str:
    text = str(text)
    return text if len(text) <= _DETAIL_CHARS else text[:_DETAIL_CHARS] + "..."


def _http_detail(error: urllib.error.HTTPError) -> str:
    try:
        body = error.read().decode(errors="replace")
    except OSError:
        return error.reason or ""
    finally:
        error.close()
    try:
        return _short(json.loads(body).get("error", body))
    except (ValueError, AttributeError):
        return _short(body)


def _post(host: str, payload: dict, *, path: str = "/api/chat", timeout: float) -> dict:
    """POST JSON to the validated loopback endpoint. Errors never echo the payload."""
    url = validate_endpoint(host) + path
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _opener().open(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        raise VlmError(f"HTTP {error.code}: {_http_detail(error)}") from None
    except (urllib.error.URLError, OSError, http.client.HTTPException) as error:
        raise VlmError(f"request to {url} failed: {getattr(error, 'reason', error)}") from None
    try:
        return json.loads(body)
    except ValueError:
        raise VlmError(f"non-JSON response from {url}") from None


def _encode(image: np.ndarray) -> str:
    """PNG/base64 of one frame; anything unencodable is a frames problem of that row."""
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim not in (2, 3)
        or image.size == 0
    ):
        raise VlmFramesError(
            "frame is not a non-empty uint8 grayscale/BGR array: "
            + (f"{image.dtype} {image.shape}" if isinstance(image, np.ndarray) else type(image).__name__)
        )
    try:
        ok, png = cv2.imencode(".png", image)
    except cv2.error as error:
        raise VlmFramesError(f"cannot encode frame {image.shape} as PNG: {_short(error)}") from None
    if not ok:
        raise VlmFramesError(f"cannot encode frame {image.shape} as PNG")
    return base64.b64encode(png.tobytes()).decode("ascii")


def _parse(data, model: str) -> VlmAnswer:
    """Accept only a completed (done/stop) reply with a non-empty schema answer.

    A server-reported ``error`` is a service failure (``VlmError``); every other defect is
    a defect of this one reply (``VlmReplyError``).
    """
    if not isinstance(data, dict):
        raise VlmReplyError(f"unexpected response type {type(data).__name__}")
    if "error" in data:
        raise VlmError(f"server error: {_short(data['error'])}")
    if data.get("done") is not True or data.get("done_reason") != "stop":
        raise VlmReplyError(
            f"incomplete reply (done={data.get('done')!r}, done_reason={data.get('done_reason')!r})"
        )
    message = data.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise VlmReplyError("empty reply content")
    try:
        parsed = json.loads(content)
    except ValueError:
        raise VlmReplyError(f"reply is not JSON: {_short(content)}") from None
    answer = parsed.get("answer") if isinstance(parsed, dict) else None
    reason = parsed.get("reason") if isinstance(parsed, dict) else None
    if not (isinstance(answer, str) and answer.strip() and isinstance(reason, str) and reason.strip()):
        raise VlmReplyError(f"reply lacks a non-empty answer/reason: {_short(content)}")
    return VlmAnswer(answer.strip(), reason.strip(), model)


# --------------------------------------------------------------------------- asking


def ask_vlm(
    images: Sequence[np.ndarray],
    question: str,
    model: str | None = None,
    fallback: str | None = None,
    host: str | None = None,
    *,
    config: VlmConfig | None = None,
) -> VlmAnswer:
    """Ask the local VLM about ``images`` via ``/api/chat`` with a JSON-schema reply.

    Omitted ``model`` / ``fallback`` / ``host`` come from ``config/vlm.yaml``. Any primary
    failure (HTTP error such as OOM or a missing tag, transport error, truncated, empty or
    malformed reply) falls back to ``fallback``; if that fails too, ``VlmError``.
    """
    cfg = config if config is not None else load_vlm_config()
    base = validate_endpoint(cfg.host if host is None else host)
    models = [validate_model(cfg.model if model is None else model)]
    second = validate_model(cfg.fallback if fallback is None else fallback)
    if second != models[0]:
        models.append(second)
    if not images:
        raise ValueError("ask_vlm needs at least one image")
    return _first_answer(base, models, [_encode(image) for image in images], question, cfg)


def _attempt(base: str, name: str, encoded: list[str], question: str, cfg: VlmConfig) -> VlmAnswer:
    payload = {
        "model": name,
        "messages": [
            {
                "role": "user",
                "content": _PROMPT.format(window=cfg.event_window_s, question=question),
                "images": encoded,
            }
        ],
        "stream": False,
        "think": cfg.think,
        "format": _SCHEMA,
        "keep_alive": cfg.keep_alive,
        "options": {"num_ctx": cfg.num_ctx, "num_predict": cfg.num_predict}
        | ({} if cfg.temperature is None else {"temperature": cfg.temperature}),
    }
    began = time.perf_counter()
    try:
        answer = _parse(_post(base, payload, timeout=cfg.timeout_s), name)
    except VlmError as error:
        log.warning("VLM %s failed after %.1fs: %s", name, time.perf_counter() - began, error)
        raise
    log.info("VLM %s answered in %.1fs (%d images)", name, time.perf_counter() - began, len(encoded))
    return answer


def _first_answer(
    base: str,
    models: list[str],
    encoded: list[str],
    question: str,
    cfg: VlmConfig,
    on_failure: Callable[[str, VlmError], None] | None = None,
) -> VlmAnswer:
    failures = []
    for name in models:
        try:
            return _attempt(base, name, encoded, question, cfg)
        except VlmError as error:
            failures.append(f"{name}: {error}")
            if on_failure is not None:
                on_failure(name, error)
    raise VlmError("; ".join(failures))


class VlmSession:
    """One VLM phase: shares loaded models across rows, releases them at the end.

    Fallback is sticky for the session once the primary fails as a *service* (HTTP error
    such as OOM or a missing tag, transport error, timeout, server-reported error): every
    retry would cost another model load/eviction and up to ``timeout_s``. A defect of one
    reply (``VlmReplyError``: truncated, empty, malformed) is about that answer, not the
    service, so the next row still asks the primary.
    """

    def __init__(self, config: VlmConfig):
        self.config = _validated(config)
        self.used = False
        self.models: Counter[str] = Counter()
        self.primary_down: str | None = None

    def _mark(self, name: str, error: VlmError) -> None:
        if name == self.config.model and not isinstance(error, VlmReplyError):
            self.primary_down = _short(error)
            log.warning(
                "VLM primary %s unavailable (%s); using %s for the rest of this phase",
                name,
                self.primary_down,
                self.config.fallback,
            )

    def ask(self, images: Sequence[np.ndarray], question: str) -> VlmAnswer:
        cfg = self.config
        if not images:
            raise ValueError("VLM needs at least one image")
        encoded = [_encode(image) for image in images]  # VlmFramesError before any request
        self.used = True
        models = list(dict.fromkeys((cfg.model, cfg.fallback)))
        skipped = self.primary_down is not None and len(models) == 2
        if skipped:
            models = models[1:]
        try:
            answer = _first_answer(cfg.host, models, encoded, question, cfg, self._mark)
        except VlmError as error:
            if skipped:
                raise VlmError(
                    f"{cfg.model}: skipped after earlier failure ({self.primary_down}); {error}"
                ) from None
            raise
        self.models[answer.model] += 1
        return answer

    def close(self) -> None:
        if not self.used:
            return
        self.used = False
        for name in dict.fromkeys((self.config.model, self.config.fallback)):
            try:
                _post(
                    self.config.host,
                    {"model": name, "keep_alive": 0},
                    path="/api/generate",
                    timeout=self.config.timeout_s,
                )
            except VlmError as error:
                log.warning("could not release VLM %s: %s", name, error)


# --------------------------------------------------------------------------- questions


def _kind(r: RoundResult) -> str:
    names = {reason.split("@", 1)[0] for reason in r.review_reasons}
    if "shake_cause_unknown" in names:
        return "touch"
    if "id_swap_risk" in names:
        return "final"
    if r.launch is not None and r.launch.air_collision == "maybe":
        return "air"
    return "final"


_QUESTIONS = {"touch": TOUCH_QUESTION, "final": FINAL_QUESTION, "air": AIR_QUESTION}


def question_for(r: RoundResult) -> str:
    """The structured question for a review-required round.

    Priority: unexplained launch shake (touch), ID-swap risk (final zone), maybe air
    collision; any other low-confidence round asks the final-zone question.
    """
    return _QUESTIONS[_kind(r)]


def event_time(r: RoundResult) -> float:
    """Playback seconds of the event the question is about (source frames / playback fps)."""
    kind = _kind(r)
    fps = (r.motion_evidence or {}).get("fps")
    valid_fps = isinstance(fps, (int, float)) and not isinstance(fps, bool) and fps > 0
    if kind == "touch":
        frames = [
            int(reason.split("@", 1)[1])
            for reason in r.review_reasons
            if reason.startswith("shake_cause_unknown@") and reason.split("@", 1)[1].isdigit()
        ]
        return min(frames) / fps if frames and valid_fps else r.t_launch
    if kind == "air":
        landed = [f for f in (r.launch.landing_frame or {}).values() if f is not None]
        return min(landed) / fps if landed and valid_fps else r.t_launch
    return r.t_end


def needs_vlm(r: RoundResult) -> bool:
    """Unreviewed real review-required rounds without a successful note yet."""
    return (
        r.needs_review
        and not r.reviewed
        and not is_not_a_round(r.review_reasons)
        and (r.vlm_note is None or r.vlm_note.startswith(UNAVAILABLE_PREFIX))
    )


def _describe(error: BaseException) -> str:
    return _short(f"{type(error).__name__}: {error}")


def _note(r: RoundResult, get_frames, session: VlmSession) -> str:
    question = question_for(r)
    try:
        images = list(get_frames(r))
    except Exception as error:  # noqa: BLE001 - any frame failure is noted; the batch continues
        log.warning("frames unavailable for %s round %s: %s", r.video, r.round, _describe(error))
        return f"{UNAVAILABLE_PREFIX}frames: {_describe(error)}"
    if not images:
        return f"{UNAVAILABLE_PREFIX}frames: none decoded near the event"
    try:
        answer = session.ask(images, question)
    except VlmFramesError as error:  # e.g. a non-uint8 / empty / unencodable frame
        log.warning("unusable frames for %s round %s: %s", r.video, r.round, _describe(error))
        return f"{UNAVAILABLE_PREFIX}frames: {_describe(error)}"
    except VlmError as error:
        return f"{UNAVAILABLE_PREFIX}{_short(error)}"
    return f"[{answer.model}] 問：{question} 答：{answer.answer}（理由：{answer.reason}）"


def review_low_confidence(
    results: list[RoundResult],
    get_frames: Callable[[RoundResult], list[np.ndarray]],
    *,
    config: VlmConfig | None = None,
    session: VlmSession | None = None,
) -> list[RoundResult]:
    """Return ``results`` with ``vlm_note`` set on each ``needs_vlm`` row (copies).

    The endpoint and models are validated before ``get_frames`` is ever called. Other
    rows are returned unchanged; inputs are not mutated. Without a caller ``session``
    the loaded models are released before returning (also on error).
    """
    own = session is None
    if own:
        session = VlmSession(config if config is not None else load_vlm_config())
    out = []
    try:
        for r in results:
            out.append(replace(r, vlm_note=_note(r, get_frames, session)) if needs_vlm(r) else r)
    finally:
        if own:
            session.close()
    return out


# --------------------------------------------------------------------------- frames


def _label(image: np.ndarray, text: str, point, color) -> None:
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    x = max(0, min(int(point[0]), image.shape[1] - width))
    y = max(height, min(int(point[1]), image.shape[0] - baseline))
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def _overlay(frame: np.ndarray, c: Calibration, t: float) -> np.ndarray:
    image = np.array(frame, dtype=np.uint8, copy=True)
    for name, _, _ in ZONES:
        pts = np.rint(getattr(c, name)).astype(np.int32)
        cv2.polylines(image, [pts], True, _ZONE_COLOR, 2)
        _label(image, name, pts[0], _ZONE_COLOR)
    _label(image, f"t={t:.2f}s", (image.shape[1], image.shape[0]), _TIME_COLOR)
    return image


def _select_calibration(source: Path, fps: float, calib_dir: Path, th: Thresholds) -> Calibration:
    """The same calibration choice as the judge: first-second frame at inference size."""
    last = None
    with closing(iter_frames(source, long_edge=th.infer_long_edge, hwaccel=th.decode_hwaccel)) as frames:
        for idx, frame in frames:
            last = frame
            if idx >= round(fps):
                break
    if last is None:
        raise VlmFramesError(f"{source}: no frames decoded")
    c = find_calibration(last, calib_dir, th.calibration_min_similarity)
    if c is None:
        raise VlmFramesError(f"{source}: no matching calibration in {calib_dir}")
    return c


def event_frames(
    source: Path,
    t: float,
    calib_dir: Path,
    config: VlmConfig,
    th: Thresholds,
    *,
    calibrations: dict | None = None,
) -> list[np.ndarray]:
    """``config.frames`` frames spread over ``t +- event_window_s`` playback seconds.

    Each frame carries all 8 calibration polygons scaled to the decoded size and its
    playback time. ``calibrations`` caches the per-source calibration choice.
    """
    source = Path(source)
    info = probe(source)
    if calibrations is not None and source in calibrations:
        c = calibrations[source]
    else:
        c = _select_calibration(source, info.fps, calib_dir, th)
        if calibrations is not None:
            calibrations[source] = c
    lo = max(0.0, t - config.event_window_s)
    hi = max(lo, min(t + config.event_window_s, info.duration))
    wanted = sorted({round(x * info.fps) for x in np.linspace(lo, hi, config.frames)})
    picked = {}
    with closing(
        iter_frames(
            source,
            start=lo,
            end=hi + 1 / info.fps,
            long_edge=config.image_long_edge,
            hwaccel=th.decode_hwaccel,
        )
    ) as frames:
        for idx, frame in frames:
            if idx in wanted:
                picked[idx] = frame
            if idx >= wanted[-1]:
                break
    if not picked:
        raise VlmFramesError(f"{source}: no frames decoded in [{lo:.2f}, {hi:.2f}] s")
    first = next(iter(picked.values()))
    scaled = scale_calibration(c, (first.shape[1], first.shape[0]))
    return [_overlay(frame, scaled, idx / info.fps) for idx, frame in sorted(picked.items())]


# --------------------------------------------------------------------------- results directory


def _source(entry) -> Path | str:
    """Absolute recorded source path, or the reason it cannot be used."""
    value = entry.get("source_video") if isinstance(entry, dict) else None
    if not value:
        return "videos.json has no source_video for this video"
    path = Path(value)
    if not path.is_absolute():
        return f"source_video {value!r} is not an absolute path"
    if not path.is_file():
        return f"source_video {value!r} does not exist"
    return path


def _persist_notes(out_dir: Path, video: str, asked: dict, rules: Rules) -> None:
    """Rewrite ``video``'s COMPLETE fresh round set with notes on still-unreviewed rows (R36)."""
    rows = json.loads((out_dir / "rounds.json").read_text(encoding="utf-8"))
    snapshot = []
    for row in rows:
        if row["video"] != video:
            continue
        r = row_to_result(row)
        note = asked.get((r.video, r.round))
        if note is not None and not r.reviewed and note[1] == (r.t_launch, r.t_end):
            r.vlm_note = note[0]
        snapshot.append(r)
    if snapshot:
        write_results(snapshot, out_dir, rules)


def review_results_dir(
    out_dir: Path,
    *,
    calib_dir: Path,
    th: Thresholds,
    rules: Rules,
    config: VlmConfig | None = None,
    videos: set[str] | None = None,
) -> dict:
    """Ask the VLM about every ``needs_vlm`` round in a results directory.

    Source media come only from the absolute ``source_video`` in ``videos.json``. Each
    video's notes are persisted through ``write_results`` as a complete snapshot and its
    VLM time is recorded as ``vlm_s`` beside the core ``processing_s``.
    """
    session = VlmSession(config if config is not None else load_vlm_config())
    out_dir = Path(out_dir)
    rounds = out_dir / "rounds.json"
    if not rounds.is_file():
        raise ValueError(f"{out_dir}: no rounds.json (run `bxjudge run` first)")
    by_video: dict[str, list[RoundResult]] = {}
    for row in json.loads(rounds.read_text(encoding="utf-8")):
        by_video.setdefault(row["video"], []).append(row_to_result(row))
    timing_path = out_dir / "videos.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else {}
    summary = {"asked": 0, "answered": 0, "unavailable": 0}
    calibrations: dict = {}
    try:
        for video, results in sorted(by_video.items()):
            if (videos is not None and video not in videos) or not any(map(needs_vlm, results)):
                continue
            began = time.perf_counter()
            source = _source(timing.get(video))

            def get_frames(r: RoundResult, source=source) -> list[np.ndarray]:
                if isinstance(source, str):
                    raise VlmFramesError(source)
                return event_frames(
                    source, event_time(r), calib_dir, session.config, th, calibrations=calibrations
                )

            reviewed = review_low_confidence(results, get_frames, session=session)
            asked = {
                (r.video, r.round): (r.vlm_note, (r.t_launch, r.t_end))
                for r, before in zip(reviewed, results)
                if r is not before
            }
            _persist_notes(out_dir, video, asked, rules)
            record_vlm_timing(out_dir, video, time.perf_counter() - began)
            for note, _ in asked.values():
                summary["asked"] += 1
                summary["unavailable" if note.startswith(UNAVAILABLE_PREFIX) else "answered"] += 1
            log.info("%s: VLM %.1fs for %d round(s)", video, time.perf_counter() - began, len(asked))
    finally:
        session.close()
    summary["models"] = dict(session.models)
    return summary
