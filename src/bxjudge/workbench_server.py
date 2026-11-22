"""Loopback workbench: independent immutable stores for active review pages."""

import argparse
import functools
import hashlib
import json
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from .identity_review import validate as validate_identity
from .reid_review import MANIFEST_SCHEMA, validate_submission
from .review_store import save_immutable


def _reject_constant(value):
    raise ValueError("非有限 JSON 數值")


class WorkbenchHandler(SimpleHTTPRequestHandler):
    _cache_header_sent = False

    def send_header(self, keyword, value):
        if keyword.lower() == "cache-control":
            self._cache_header_sent = True
        super().send_header(keyword, value)

    def end_headers(self):
        # Regenerated pages must not be served stale from the browser cache.
        if not self._cache_header_sent:
            super().send_header("Cache-Control", "no-cache")
        self._cache_header_sent = False
        super().end_headers()

    def reply(self, status, body):
        data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_video(self):
        name = unquote(self.path.split("?", 1)[0].removeprefix("/video/"))
        path = self.server.videos.get(name)
        if path is None or not path.is_file():
            return self.reply(404, {"error": "影片不在本次 run 的允許清單"})
        size = path.stat().st_size
        start, end = 0, size - 1
        ranged = self.headers.get("Range")
        if ranged:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", ranged.strip())
            if match is None or match[1] == match[2] == "":
                start = size  # malformed: force 416 below
            elif match[1] == "":
                start = max(0, size - int(match[2]))
            else:
                start = int(match[1])
                end = min(end, int(match[2])) if match[2] else end
            if start >= size or start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
        self.send_response(206 if ranged else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if ranged:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = stream.read(min(1 << 20, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break  # Browsers abort range requests while seeking.
                remaining -= len(chunk)
        return None

    def do_GET(self):
        if self.path.startswith("/video/"):
            return self.send_video()
        save_route = self.path.removesuffix("/saved") + "/save"
        route = (
            self.server.review_routes.get(save_route)
            if self.path.endswith("/saved")
            else None
        )
        if route is None or route[4] != "reid":
            return super().do_GET()
        manifest, digest, folder, validator, _prefix, _ = route
        snapshots = []
        for path in sorted(
            folder.glob("reid-*.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True
        ):
            raw = path.read_bytes()
            actual = hashlib.sha256(raw).hexdigest()
            if path.name != f"reid-{actual}.json":
                continue
            try:
                payload = json.loads(raw, parse_constant=_reject_constant)
                stats = validator(payload, manifest, digest)
            except (ValueError, TypeError, UnicodeError):
                continue
            snapshots.append(
                {"filename": path.name, "sha256": actual, "payload": payload, **stats}
            )
        return self.reply(200, {"snapshots": snapshots})

    def do_POST(self):
        route = self.server.review_routes.get(self.path)
        if route is None:
            return self.reply(404, {"error": "未知儲存路徑"})
        manifest, digest, folder, validator, prefix, limit = route
        host = self.headers.get("Host", "")
        hosts = {
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        }
        if host not in hosts or self.headers.get("Origin") != "http://" + host:
            return self.reply(403, {"error": "僅接受本機核對頁儲存"})
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self.reply(415, {"error": "需要 JSON 核對資料"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self.reply(400, {"error": "無效資料長度"})
        if not 0 < length <= limit:
            return self.reply(413, {"error": "核對資料長度無效或過大"})
        try:
            self.connection.settimeout(10)
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("資料未傳完整")
            payload = json.loads(raw, parse_constant=_reject_constant)
            summary = validator(payload, manifest, digest)
            if prefix == "reid" and payload["supersedes"] is not None:
                parent = folder / f"reid-{payload['supersedes']}.json"
                if not parent.is_file():
                    raise ValueError("修訂來源不在專案紀錄中；草稿保留")
                parent_raw = parent.read_bytes()
                if hashlib.sha256(parent_raw).hexdigest() != payload["supersedes"]:
                    raise ValueError("修訂來源雜湊不符")
                validator(
                    json.loads(parent_raw, parse_constant=_reject_constant),
                    manifest,
                    digest,
                )
            target, saved_sha = save_immutable(folder, payload, prefix=prefix)
        except (ValueError, TypeError, UnicodeError) as exc:
            return self.reply(422, {"error": str(exc)})
        except OSError:
            return self.reply(
                500, {"error": "本機寫檔失敗，專案尚未儲存；請保留頁面或匯出備份"}
            )
        return self.reply(
            201,
            {"saved": True, "filename": target.name, "sha256": saved_sha, **summary},
        )


def video_allowlist(runs: list[Path]) -> dict[str, Path]:
    """Original videos named by each run's videos.json; read-only, existing files only."""
    allowed = {}
    for run in runs:
        for name, info in json.loads((Path(run) / "videos.json").read_text()).items():
            if Path(name).name != name or not name.lower().endswith(".mp4"):
                raise ValueError(f"invalid video name in videos.json: {name!r}")
            source = (info or {}).get("source_video")
            if source and Path(source).is_file():
                allowed[name] = Path(source).resolve()
    return allowed


def make_server(
    site: Path,
    identity_submissions: Path,
    reid_sets: dict[str, Path],
    port: int,
    videos: dict[str, Path] | None = None,
):
    site = Path(site).resolve()
    if (
        not isinstance(reid_sets, dict)
        or not reid_sets
        or any(re.fullmatch(r"reid-[a-z0-9-]+", name) is None for name in reid_sets)
    ):
        raise ValueError("invalid explicit reid sets")
    folders = [
        Path(identity_submissions).resolve(),
        *(Path(p).resolve() for p in reid_sets.values()),
    ]
    if len(folders) != len(set(folders)):
        raise ValueError("review stores must be distinct")
    routes = {}
    for name, folder, validator, prefix in (
        ("identity-audit", identity_submissions, validate_identity, "labels"),
        *(
            (name, folder, validate_submission, "reid")
            for name, folder in reid_sets.items()
        ),
    ):
        raw = (site / name / "manifest.json").read_bytes()
        manifest = json.loads(raw, parse_constant=_reject_constant)
        if prefix == "reid" and manifest["schema"] != MANIFEST_SCHEMA:
            raise ValueError("invalid reidentification review manifest")
        routes[f"/{name}/save"] = (
            manifest,
            hashlib.sha256(raw).hexdigest(),
            Path(folder).resolve(),
            validator,
            prefix,
            max(64 * 1024, 8 * len(raw)),
        )
    server = ThreadingHTTPServer(
        ("127.0.0.1", port), functools.partial(WorkbenchHandler, directory=str(site))
    )
    server.review_routes = routes
    server.videos = dict(videos or {})
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--identity-submissions", type=Path, required=True)
    parser.add_argument(
        "--reid-set",
        nargs=2,
        action="append",
        required=True,
        metavar=("NAME", "SUBMISSIONS"),
    )
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--video-run",
        type=Path,
        action="append",
        default=[],
        help="run dir whose videos.json source_video files may be streamed read-only",
    )
    args = parser.parse_args()
    reid_sets = {name: Path(folder) for name, folder in args.reid_set}
    if len(reid_sets) != len(args.reid_set):
        parser.error("duplicate reid set names")
    server = make_server(
        args.site,
        args.identity_submissions,
        reid_sets,
        args.port,
        videos=video_allowlist(args.video_run),
    )
    print(f"Local workbench http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
