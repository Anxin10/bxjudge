"""Build the canonical stadium model (config/stadium_model.json) from one manual calibration.

Read only on calib/: the model and the preview are refused inside calib/ and over the
template or its reference image (resolved paths, so links cannot reach them). Usage:
    uv run python scripts/build_stadium_model.py [--template calib/20261014_202713.json]
        [--out config/stadium_model.json] [--preview <jpg>]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from bxjudge import PROJECT_ROOT
from bxjudge.calibration import load_calibration
from bxjudge.stadium_finder import (
    DEFAULT_STADIUM_MODEL,
    build_stadium_model,
    load_stadium_finder_config,
    save_stadium_model,
)


def check_targets(targets: list[Path], *, template: Path, reference: Path | None, calib: Path) -> None:
    """Refuse any write target inside ``calib`` or equal to the template or its reference
    image, all compared after resolving links."""
    protected_dir = calib.expanduser().resolve()
    protected_files = {template.expanduser().resolve()}
    if reference is not None:
        protected_files.add(Path(reference).expanduser().resolve())
    for t in targets:
        real = t.expanduser().resolve()
        if real in protected_files or real == protected_dir or protected_dir in real.parents:
            raise SystemExit(f"{t}: refusing to write over manual calibration data")


def main(argv: list[str] | None = None) -> int:
    cfg = load_stadium_finder_config()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--template", type=Path,
                   default=PROJECT_ROOT / "calib" / f"{cfg.model_template}.json")
    p.add_argument("--out", type=Path, default=DEFAULT_STADIUM_MODEL)
    p.add_argument("--preview", type=Path, default=None)
    a = p.parse_args(argv)
    targets = [a.out] + ([a.preview] if a.preview else [])
    calib = PROJECT_ROOT / "calib"
    check_targets(targets, template=a.template, reference=None, calib=calib)  # before reading
    template = load_calibration(a.template)
    check_targets(targets, template=a.template, reference=Path(template.ref_frame), calib=calib)
    image = cv2.imread(template.ref_frame, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read reference image {template.ref_frame}")
    model = build_stadium_model(template, image, cfg)
    save_stadium_model(model, a.out)
    print(f"model: {len(model.rail)} rail points, sha256 {model.source_sha256[:12]} -> {a.out}")
    if a.preview:
        img = image.copy()
        for x, y in model.rail:
            cv2.circle(img, (round(x), round(y)), 4, (255, 0, 255), -1)
        for name in ("battle_area", "over_left", "over_right", "xtreme"):
            poly = getattr(model.calibration, name)
            cv2.polylines(img, [np.round(np.array(poly)).astype(np.int32)], True, (0, 0, 255), 3)
        a.preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(a.preview), cv2.resize(img, (960, 540)))
        print(f"preview -> {a.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
