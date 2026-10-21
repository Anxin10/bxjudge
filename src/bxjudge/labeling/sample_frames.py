"""Event-aware sampling; one decoded frame and one grayscale frame in memory."""

import math
from pathlib import Path

import cv2
import numpy as np

from bxjudge.video import iter_frames

from .config import load_config


def change_energy(path: Path, long_edge: int | None = None) -> dict[int, float]:
    if long_edge is None:
        long_edge = load_config()["energy_long_edge"]
    if long_edge <= 0:
        raise ValueError("long_edge must be positive")
    energy = {}
    previous = None
    frames = iter_frames(path, long_edge=long_edge, hwaccel="cpu")
    try:
        for index, frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            energy[index] = (
                0.0 if previous is None else float(np.mean(cv2.absdiff(gray, previous)))
            )
            previous = gray
    finally:
        frames.close()
    return energy


def select_frames(
    energy: dict[int, float],
    fps: float,
    n: int,
    uniform_ratio: float | None = None,
    min_gap_s: float | None = None,
) -> list[int]:
    cfg = load_config()
    uniform_ratio = cfg["uniform_ratio"] if uniform_ratio is None else uniform_ratio
    min_gap_s = cfg["min_gap_s"] if min_gap_s is None else min_gap_s
    if (
        not math.isfinite(fps)
        or fps <= 0
        or type(n) is not int
        or n < 0
        or not 0 <= uniform_ratio <= 1
        or min_gap_s < 0
    ):
        raise ValueError("invalid sampling arguments")
    if any(
        type(i) is not int or i < 0 or not math.isfinite(v) for i, v in energy.items()
    ):
        raise ValueError(
            "energy requires nonnegative source indices and finite energies"
        )
    indices = sorted(energy)
    if not indices or n == 0:
        return []
    gap = max(1, math.ceil(fps * min_gap_s))
    target = min(n, len(indices))
    chosen = []

    def add(index):
        if all(abs(index - other) >= gap for other in chosen):
            chosen.append(index)

    # Uniform reservations prevent high-energy clusters from consuming the entire clip.
    uniform_n = round(target * uniform_ratio)
    for pos in np.linspace(0, len(indices) - 1, uniform_n):
        add(indices[round(pos)])
    # Local maxima are preferred to shoulders around a large event.
    peaks = [
        i
        for p, i in enumerate(indices)
        if (p == 0 or energy[i] >= energy[indices[p - 1]])
        and (p == len(indices) - 1 or energy[i] >= energy[indices[p + 1]])
    ]
    ranked = sorted(peaks, key=lambda i: (-energy[i], i))
    for index in ranked + sorted(indices, key=lambda i: (-energy[i], i)):
        if len(chosen) >= target:
            break
        add(index)
    return sorted(chosen)
