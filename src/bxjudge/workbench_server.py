"""Loopback workbench: independent immutable stores for both active review pages."""

import argparse
import functools
import hashlib
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .identity_review import validate as validate_identity
from .reid_review import MANIFEST_SCHEMA, validate_submission
from .review_store import save_immutable


def _reject_constant(value):
    raise ValueError("非有限 JSON 數值")


class WorkbenchHandler(SimpleHTTPRequestHandler):
    def reply(self, status, body):
        data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path != "/reid-review/saved":
            return super().do_GET()
        manifest, digest, folder, validator, _prefix, _ = self.server.review_routes[
            "/reid-review/save"
        ]
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
            return self.reply(500, {"error": "本機寫檔失敗，專案尚未儲存；請保留頁面或匯出備份"})
        return self.reply(
            201,
            {"saved": True, "filename": target.name, "sha256": saved_sha, **summary},
        )


def make_server(
    site: Path, identity_submissions: Path, reid_submissions: Path, port: int
):
    site = Path(site).resolve()
    routes = {}
    for name, folder, validator, prefix in (
        ("identity-audit", identity_submissions, validate_identity, "labels"),
        ("reid-review", reid_submissions, validate_submission, "reid"),
    ):
        raw = (site / name / "manifest.json").read_bytes()
        manifest = json.loads(raw, parse_constant=_reject_constant)
        if name == "reid-review" and manifest["schema"] != MANIFEST_SCHEMA:
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
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--identity-submissions", type=Path, required=True)
    parser.add_argument("--reid-submissions", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    server = make_server(
        args.site, args.identity_submissions, args.reid_submissions, args.port
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
