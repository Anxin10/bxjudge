import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest


def fixtures():
    manifest = {
        "schema": "bxjudge.reid-review-manifest.v1",
        "cases": [{"case_id": "a"}, {"case_id": "b"}],
    }
    digest = hashlib.sha256(json.dumps(manifest).encode()).hexdigest()
    fields = {
        "reviewed": False,
        "relation": "unreviewed",
        "side_before": "unreviewed",
        "side_after": "unreviewed",
        "visibility_before": "unreviewed",
        "visibility_after": "unreviewed",
        "collision": "unreviewed",
        "scope": "unreviewed",
        "note": "",
    }
    payload = {
        "schema": "bxjudge.reid-review-labels.v1",
        "manifest_sha256": digest,
        "submitted_at": "2026-10-26T00:00:00Z",
        "supersedes": None,
        "cases": [{"case_id": c["case_id"], **fields} for c in manifest["cases"]],
    }
    return manifest, digest, payload


def test_explicit_uncertainty_is_reviewed_but_not_a_same_object_truth():
    from bxjudge.reid_review import validate_submission

    manifest, digest, p = fixtures()
    p["cases"][0].update(reviewed=True, relation="uncertain", scope="unknown")
    stats = validate_submission(p, manifest, digest)
    assert stats == {"reviewed_cases": 1, "same": 0, "different": 0, "uncertain": 1}
    assert p["cases"][0]["side_before"] == "unreviewed"


def test_blank_case_and_default_confirmation_are_not_ground_truth():
    from bxjudge.reid_review import validate_submission

    m, d, p = fixtures()
    for payload in (
        p,
        {**p, "cases": [{**p["cases"][0], "reviewed": True}, p["cases"][1]]},
    ):
        with pytest.raises(ValueError):
            validate_submission(payload, m, d)


@pytest.mark.parametrize(
    "field,value",
    [
        ("reviewed", 1),
        ("relation", "same-ish"),
        ("side_before", "dark"),
        ("visibility_after", "perfect"),
        ("scope", "guessed"),
        ("collision", True),
    ],
)
def test_label_enum_and_boolean_types_are_strict(field, value):
    from bxjudge.reid_review import validate_submission

    m, d, p = fixtures()
    p["cases"][0].update(reviewed=True, relation="same", scope="same_round")
    p["cases"][0][field] = value
    with pytest.raises(ValueError):
        validate_submission(p, m, d)


def test_manifest_and_case_identity_cannot_be_replaced():
    from bxjudge.reid_review import validate_submission

    m, d, p = fixtures()
    p["cases"][0].update(reviewed=True, relation="same", scope="same_round")
    for mutate in (
        lambda q: q.update(manifest_sha256="0" * 64),
        lambda q: q["cases"][0].update(case_id="another"),
        lambda q: q["cases"][0].update(source_sha256="fake"),
        lambda q: q.update(applied=True),
    ):
        candidate = copy.deepcopy(p)
        mutate(candidate)
        with pytest.raises(ValueError):
            validate_submission(candidate, m, d)


def test_revised_labels_create_new_files_and_repeat_is_idempotent(tmp_path):
    from bxjudge.review_store import save_immutable

    _m, _d, p = fixtures()
    p["cases"][0].update(reviewed=True, relation="same", scope="same_round")
    target, digest = save_immutable(tmp_path, p, prefix="reid")
    original = target.read_bytes()
    stamp = target.stat().st_mtime_ns
    with ThreadPoolExecutor(4) as executor:
        result = list(
            executor.map(lambda _: save_immutable(tmp_path, p, prefix="reid"), range(4))
        )
    assert all(path == target for path, _ in result)
    assert target.read_bytes() == original and target.stat().st_mtime_ns == stamp
    p["supersedes"] = digest
    p["cases"][0]["relation"] = "uncertain"
    revised, _ = save_immutable(tmp_path, p, prefix="reid")
    assert revised != target and target.read_bytes() == original
    assert len(list(tmp_path.glob("reid-*.json"))) == 2
    assert not list(tmp_path.glob(".pending-*"))


def test_revision_id_must_be_a_digest_and_unknown_side_is_not_forced():
    from bxjudge.reid_review import validate_submission

    m, d, p = fixtures()
    p["cases"][0].update(
        reviewed=True,
        relation="same",
        scope="unknown",
        side_before="unknown",
        side_after="unknown",
    )
    assert validate_submission(p, m, d)["same"] == 1
    p["supersedes"] = "../../other"
    with pytest.raises(ValueError):
        validate_submission(p, m, d)
