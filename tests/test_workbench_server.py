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
    challenge = {
        **m,
        "cases": [dict(c, case_id=c["case_id"] + "-challenge") for c in m["cases"]],
    }
    for name, data in (
        ("identity-audit", old),
        ("reid-review", m),
        ("reid-challenge", challenge),
    ):
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
    srv = make_server(
        site,
        tmp_path / "old",
        {"reid-review": tmp_path / "new", "reid-challenge": tmp_path / "challenge"},
        port=0,
    )
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


def test_challenge_manifest_labels_and_revisions_are_isolated_from_original_review(
    server,
):
    import copy

    srv, root, _old, new = server
    assert post(srv, "/reid-review/save", new)[0] == 201
    assert post(srv, "/reid-challenge/save", new)[0] == 422
    base = f"http://127.0.0.1:{srv.server_port}"
    manifest_raw = (root / "site/reid-challenge/manifest.json").read_bytes()
    challenge = copy.deepcopy(new)
    challenge["manifest_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
    for c in challenge["cases"]:
        c["case_id"] += "-challenge"
    status, receipt = post(srv, "/reid-challenge/save", challenge)
    assert status == 201
    with urlopen(base + "/reid-challenge/saved", timeout=3) as response:
        saved = json.load(response)["snapshots"]
    assert len(saved) == 1 and saved[0]["sha256"] == receipt["sha256"]
    assert (root / "challenge" / receipt["filename"]).exists()
    assert not (root / "new" / receipt["filename"]).exists()
    assert post(srv, "/reid-review/save", challenge)[0] == 422


def get(srv, path, headers=None):
    req = Request(f"http://127.0.0.1:{srv.server_port}{path}", headers=headers or {})
    try:
        with urlopen(req, timeout=3) as response:
            return response.status, dict(response.headers), response.read()
    except HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


@pytest.fixture
def video_server(tmp_path):
    from bxjudge.workbench_server import make_server

    site = tmp_path / "site"
    for name in ("identity-audit", "reid-review"):
        (site / name).mkdir(parents=True)
    (site / "identity-audit/manifest.json").write_text("{}")
    m, _, _ = fixtures()
    (site / "reid-review/manifest.json").write_text(json.dumps(m))
    (tmp_path / "secret.mp4").write_bytes(b"no")
    clip = tmp_path / "videos" / "a.mp4"
    clip.parent.mkdir()
    clip.write_bytes(bytes(range(100)))
    srv = make_server(site, tmp_path / "old", {"reid-review": tmp_path / "new"}, 0,
                      videos={"a.mp4": clip})
    thread = Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv, clip
    srv.shutdown()
    srv.server_close()
    thread.join()


def test_video_route_serves_allowlisted_file_with_ranges(video_server):
    srv, clip = video_server
    status, headers, body = get(srv, "/video/a.mp4")
    assert status == 200 and body == clip.read_bytes()
    assert headers["Accept-Ranges"] == "bytes" and headers["Content-Type"] == "video/mp4"
    status, headers, body = get(srv, "/video/a.mp4", {"Range": "bytes=10-19"})
    assert status == 206 and body == bytes(range(10, 20))
    assert headers["Content-Range"] == "bytes 10-19/100"
    status, _, body = get(srv, "/video/a.mp4", {"Range": "bytes=95-"})
    assert status == 206 and body == bytes(range(95, 100))
    status, _, body = get(srv, "/video/a.mp4", {"Range": "bytes=-3"})
    assert status == 206 and body == bytes([97, 98, 99])
    status, headers, _ = get(srv, "/video/a.mp4", {"Range": "bytes=200-"})
    assert status == 416 and headers["Content-Range"] == "bytes */100"


def test_static_pages_revalidate_and_json_replies_stay_no_store(video_server):
    srv, _ = video_server
    status, headers, _ = get(srv, "/reid-review/manifest.json")
    assert status == 200 and headers["Cache-Control"] == "no-cache"
    status, headers, _ = get(srv, "/reid-review/saved")
    assert status == 200 and headers["Cache-Control"] == "no-store"


def test_video_route_refuses_unlisted_and_traversal_names(video_server):
    srv, _ = video_server
    for path in ("/video/secret.mp4", "/video/..%2Fsecret.mp4", "/video/../secret.mp4",
                 "/video/", "/video/a.mp4/x"):
        assert get(srv, path)[0] == 404


def test_video_allowlist_reads_run_videos_json(tmp_path):
    from bxjudge.workbench_server import video_allowlist

    clip = tmp_path / "v.mp4"
    clip.write_bytes(b"x")
    run = tmp_path / "run"
    run.mkdir()
    (run / "videos.json").write_text(json.dumps({
        "v.mp4": {"source_video": str(clip)},
        "gone.mp4": {"source_video": str(tmp_path / "gone.mp4")},
        "x.mp4": {},
    }))
    assert video_allowlist([run]) == {"v.mp4": clip.resolve()}
    with pytest.raises(ValueError, match="name"):
        (run / "videos.json").write_text(json.dumps({"../v.mp4": {"source_video": str(clip)}}))
        video_allowlist([run])


def test_review_routes_cannot_share_storage_or_escape_their_url_prefix(tmp_path):
    from bxjudge.workbench_server import make_server

    with pytest.raises(ValueError, match="distinct"):
        make_server(
            tmp_path, tmp_path / "labels", {"reid-review": tmp_path / "labels"}, 0
        )
    with pytest.raises(ValueError, match="explicit"):
        make_server(tmp_path, tmp_path / "labels", {"../x": tmp_path / "new"}, 0)
