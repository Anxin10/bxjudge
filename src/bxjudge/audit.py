"""Current frozen audit schema and population shared by drawing and evaluation."""
from __future__ import annotations

from datetime import datetime

from .rules import is_not_a_round


def audit_population(rows: list[dict]) -> list[list]:
    return sorted(
        [r["video"], r["round"]] for r in rows
        if not is_not_a_round(r.get("review_reasons") or []) and r["needs_review"] is False
    )


def validate_audit(meta) -> dict:
    """Reject malformed or obsolete records with ValueError, before using their keys."""
    def invalid():
        raise ValueError("malformed audit sample record; redraw with audit-sample --replace")

    if not isinstance(meta, dict) or set(meta) != {"seed", "n", "population", "selection", "at"}:
        invalid()
    if type(meta["seed"]) is not int or type(meta["n"]) is not int or meta["n"] < 1:
        invalid()
    for field in ("population", "selection"):
        keys = meta[field]
        if not isinstance(keys, list) or not all(
            isinstance(k, list) and len(k) == 2 and isinstance(k[0], str) and bool(k[0])
            and type(k[1]) is int and k[1] >= 1 for k in keys
        ):
            invalid()
        if len({tuple(k) for k in keys}) != len(keys):
            invalid()
    if (len(meta["selection"]) != meta["n"]
            or not {tuple(k) for k in meta["selection"]} <= {tuple(k) for k in meta["population"]}):
        invalid()
    if not isinstance(meta["at"], str):
        invalid()
    try:
        datetime.fromisoformat(meta["at"])
    except ValueError:
        invalid()
    return meta
