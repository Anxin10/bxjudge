import hashlib
import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from tests.test_reid_review import fixtures


def identity_fixture():
    obj = {
        "index": 0,
        "class": "bey",
        "box": [1, 2, 3, 4],
        "score": 0.9,
        "track_ids": [1],
    }
    case = {
        "video": "synthetic.mp4",
        "source_sha256": "a" * 64,
        "fps": 30,
        "coordinate_size": [640, 360],
        "frames": [{"frame": 0, "image_sha256": "b" * 64, "objects": [obj]}],
    }
    m = {"base_commit": "synthetic", "cases": [case]}
    c = {k: case[k] for k in ("video", "source_sha256", "fps", "coordinate_size")}
    c["frames"] = [
        {
            "frame": 0,
            "image_sha256": "b" * 64,
            "confirmed": True,
            "missed_or_bad_detection": False,
            "note": "SYNTHETIC ONLY",
            "objects": [
                {
                    "index": 0,
                    "detector_class": "bey",
                    "box": [1, 2, 3, 4],
                    "score": 0.9,
                    "narrow_track_ids": [1],
                    "side": "unknown",
                    "state": "unknown",
                }
            ],
        }
    ]
    p = {
        "schema": "bxjudge.identity-launch-labels.v1",
        "manifest_sha256": "",
        "base_commit": "synthetic",
        "exported_at": "2026-10-26T00:00:00Z",
        "scope": "SYNTHETIC TEST ONLY",
        "cases": [c],
    }
    return m, p


@pytest.fixture
def server(tmp_path):
    from bxjudge.workbench_server import make_server

    site = tmp_path / "site"
    old, payload = identity_fixture()
    m, _, new = fixtures()
    for name, data in (("identity-audit", old), ("reid-review", m)):
        path = site / name
        path.mkdir(parents=True)
        (path / "manifest.json").write_text(json.dumps(data))
    payload["manifest_sha256"] = hashlib.sha256(
        (site / "identity-audit/manifest.json").read_bytes()
    ).hexdigest()
    new["manifest_sha256"] = hashlib.sha256(
        (site / "reid-review/manifest.json").read_bytes()
    ).hexdigest()
    new["cases"][0].update(reviewed=True, relation="uncertain", scope="unknown")
    srv = make_server(site, tmp_path / "old", tmp_path / "new", port=0)
    thread = Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv, tmp_path, payload, new
    srv.shutdown()
    srv.server_close()
    thread.join()


def post(srv, path, payload, origin=None):
    base = f"http://127.0.0.1:{srv.server_port}"
    req = Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Origin": origin or base},
        method="POST",
    )
    try:
        with urlopen(req, timeout=3) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, json.load(exc)


def test_active_old_and_new_review_routes_save_independently(server):
    srv, root, old, new = server
    assert post(srv, "/identity-audit/save", old)[0] == 201
    assert post(srv, "/reid-review/save", new)[0] == 201
    assert len(list((root / "old").glob("labels-*.json"))) == 1
    assert len(list((root / "new").glob("reid-*.json"))) == 1


def test_cross_origin_empty_and_wrong_manifest_never_write_labels(server):
    srv, root, _old, new = server
    assert post(srv, "/reid-review/save", new, "http://untrusted.invalid")[0] == 403
    assert post(srv, "/reid-review/save", {})[0] == 422
    new["manifest_sha256"] = "0" * 64
    assert post(srv, "/reid-review/save", new)[0] == 422
    assert not (root / "new").exists()


def test_reid_save_receipt_matches_content_hash_and_allows_manual_restore(server):
    srv, root, _old, new = server
    status, receipt = post(srv, "/reid-review/save", new)
    assert status == 201 and receipt["uncertain"] == 1
    data = (root / "new" / receipt["filename"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == receipt["sha256"]
    assert json.loads(data) == new
    with urlopen(
        f"http://127.0.0.1:{srv.server_port}/reid-review/saved", timeout=3
    ) as response:
        snapshots = json.load(response)["snapshots"]
    assert len(snapshots) == 1 and snapshots[0]["sha256"] == receipt["sha256"]
    assert snapshots[0]["payload"] == new


def test_reid_revision_parent_must_exist_in_the_same_manifest(server):
    srv, root, _old, new = server
    new["supersedes"] = "f" * 64
    assert post(srv, "/reid-review/save", new)[0] == 422
    assert not (root / "new").exists()
