"""Human review corrections: ``reviewer_fix`` overrides plus an append-only audit log.

Original model judgments in ``rounds.json`` are never overwritten: a correction only
sets ``reviewer_fix[field]`` and ``reviewed``. Every actual change is appended to
``corrections.jsonl`` with the value it replaced (the then-current effective value).
Exports are rebuilt through :func:`bxjudge.output.save_rounds` over the complete row
set, so untouched rounds are never dropped (R36).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, get_args

from .output import effective, read_rounds, save_rounds
from .rules import FINISHES, SIDES, Rules, finish_points, is_not_a_round, load_rules
from .types import AirState, SpinDir, TouchState, Zone

LOG = "corrections.jsonl"

_TOUCH = get_args(TouchState)
_SPIN = get_args(SpinDir)
_AIR = get_args(AirState)
_ZONES = get_args(Zone)

# field -> allowed values (``bool`` means exactly True/False)
EDITABLE_FIELDS: dict[str, tuple | type] = {
    "x_touch": _TOUCH,
    "b_touch": _TOUCH,
    "x_landed": bool,
    "b_landed": bool,
    "x_land_zone": (*_ZONES, None),
    "b_land_zone": (*_ZONES, None),
    "x_spin_dir": _SPIN,
    "b_spin_dir": _SPIN,
    "air_collision": _AIR,
    "finish": (*FINISHES, None),
    "winner": (*SIDES, None),
    "match_start": bool,
}
_LAUNCH_FIELDS = frozenset(
    {"x_touch", "b_touch", "x_landed", "b_landed", "x_land_zone", "b_land_zone", "air_collision"}
)
_SPIN_FIELDS = frozenset({"x_spin_dir", "b_spin_dir"})


class CorrectionError(ValueError):
    """A correction that cannot be applied; nothing was written."""


@dataclass(frozen=True)
class Correction:
    video: str
    round: int
    field: str
    old: Any
    new: Any
    at: str


def now() -> str:
    """Local ISO timestamp for ``Correction.at``."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _same(a, b) -> bool:
    return type(a) is type(b) and a == b


def _check_value(field: str, value) -> None:
    allowed = EDITABLE_FIELDS.get(field)
    if allowed is None:
        raise CorrectionError(f"{field!r} is not a reviewer-editable field")
    if allowed is bool:
        ok = isinstance(value, bool)
    else:
        ok = any(_same(value, v) for v in allowed)
    if not ok:
        raise CorrectionError(f"{field}: invalid value {value!r}")


def current_value(row: dict, field: str):
    """The round's current effective value of ``field`` (absent match_start = False)."""
    eff = effective(row)
    return eff.get("match_start", False) if field == "match_start" else eff.get(field)


def _find(rows: list[dict], video: str, round: int) -> dict:
    for row in rows:
        if row["video"] == video and row["round"] == round:
            if is_not_a_round(row.get("review_reasons") or []):
                raise CorrectionError(
                    f"{video} round {round} is not a round ({row['review_reasons']}): "
                    "fix the calibration/source and reanalyze instead"
                )
            return row
    raise CorrectionError(f"no round {round} for {video!r} in rounds.json")


def _check_consistency(row: dict, changed: set[str]) -> None:
    eff = effective(row)
    if changed & {"finish", "winner"} and (eff["finish"] is None) != (eff["winner"] is None):
        raise CorrectionError(
            f"finish {eff['finish']!r} and winner {eff['winner']!r} must both be set "
            "or both be empty (no finish = redo)"
        )
    for side in ("x", "b"):
        if changed & {f"{side}_landed", f"{side}_land_zone"}:
            landed, zone = eff[f"{side}_landed"], eff[f"{side}_land_zone"]
            if landed != (zone == "battle_area"):
                raise CorrectionError(
                    f"{side}_landed={landed!r} contradicts {side}_land_zone={zone!r}: "
                    "landed means the first stable zone is battle_area"
                )


def apply_corrections(
    out_dir: Path, corrections: list[Correction], *, rules: Rules | None = None
) -> list[Correction]:
    """Apply a batch atomically; returns the corrections that actually changed a value.

    Each ``old`` must equal the round's current effective value. Values equal to the
    current effective value are no-ops (not logged, row untouched). Invalid fields,
    values or resulting combinations raise :class:`CorrectionError` before any write.
    """
    out_dir = Path(out_dir)
    rows = read_rounds(out_dir)
    applied: list[Correction] = []
    touched: dict[int, set[str]] = {}
    for c in corrections:
        _check_value(c.field, c.new)
        row = _find(rows, c.video, c.round)
        if c.field in _LAUNCH_FIELDS and not row.get("launch_present"):
            raise CorrectionError(f"{c.field}: {c.video} round {c.round} has no launch check")
        if c.field in _SPIN_FIELDS and not row.get("spin_dir_present"):
            raise CorrectionError(f"{c.field}: {c.video} round {c.round} has no spin result")
        current = current_value(row, c.field)
        if not _same(c.old, current):
            raise CorrectionError(
                f"{c.field}: old value {c.old!r} is not the current value {current!r}"
            )
        if _same(c.new, current):
            continue
        row["reviewer_fix"] = {**(row.get("reviewer_fix") or {}), c.field: c.new}
        touched.setdefault(id(row), set()).add(c.field)
        applied.append(c)
    if not applied:
        return []
    rules = rules if rules is not None else load_rules()
    for row in rows:
        changed = touched.get(id(row))
        if changed is None:
            continue
        _check_consistency(row, changed)
        fix = row["reviewer_fix"]
        if "finish" in fix:
            fix["points"] = finish_points(rules, fix["finish"]) if fix["finish"] else 0
        row["reviewed"] = True
    save_rounds(rows, out_dir, rules)
    with (out_dir / LOG).open("a", encoding="utf-8") as f:
        for c in applied:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    return applied


def apply_correction(out_dir: Path, c: Correction, *, rules: Rules | None = None) -> None:
    """Apply one correction (see :func:`apply_corrections`)."""
    apply_corrections(out_dir, [c], rules=rules)


def confirm_reviewed(
    out_dir: Path, video: str, round: int, *, rules: Rules | None = None
) -> bool:
    """「確認無誤」: mark a real round reviewed without changing any value.

    Returns False when it was already reviewed (nothing written). Not logged: the
    audit log records value changes only.
    """
    out_dir = Path(out_dir)
    rows = read_rounds(out_dir)
    row = _find(rows, video, round)
    if row["reviewed"]:
        return False
    row["reviewed"] = True
    save_rounds(rows, out_dir, rules if rules is not None else load_rules())
    return True


def load_corrections(out_dir: Path) -> list[Correction]:
    """Every logged correction, oldest first."""
    path = Path(out_dir) / LOG
    if not path.exists():
        return []
    return [
        Correction(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
