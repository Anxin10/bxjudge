"""Read-only frame coverage, including missing samples and confirmation prefixes.

Coverage measures availability of recorded geometry, not correctness of that
geometry. Historical 'lost' states do not identify occlusion, bounds or mismatch.
"""

from collections import Counter


def measure_windows(samples, windows):
    by_frame = {}
    previous = -1
    for sample in samples:
        frame = sample["frame"]
        if type(frame) is not int or frame < 0 or frame <= previous:
            raise ValueError(
                "geometry sample frames must be increasing nonnegative integers"
            )
        previous = frame
        by_frame[frame] = sample
    selected = set()
    for lo, hi in windows:
        if type(lo) is not int or type(hi) is not int or lo < 0 or hi < lo:
            raise ValueError("invalid inclusive source-frame window")
        selected.update(range(lo, hi + 1))
    reasons, intervals, statuses = Counter(), [], Counter()
    valid = 0
    for frame in sorted(selected):
        sample = by_frame.get(frame)
        if sample is None:
            reason = "sample_absent"
            statuses["sample_absent"] += 1
        else:
            statuses[sample["status"]] += 1
            if sample["alignment"] is not None:
                valid += 1
                continue
            reason = sample.get("rejection_reason") or {
                "lost": "alignment_rejected_cause_unrecorded",
                "confirming": "confirming",
                "candidate": "candidate_not_confirmed",
            }.get(sample["status"], "geometry_unavailable_cause_unrecorded")
        reasons[reason] += 1
        if intervals and intervals[-1]["last_frame"] == frame - 1:
            interval = intervals[-1]
            interval["last_frame"] = frame
            interval["frames"] += 1
            interval["reason_counts"][reason] = (
                interval["reason_counts"].get(reason, 0) + 1
            )
        else:
            intervals.append(
                {
                    "first_frame": frame,
                    "last_frame": frame,
                    "frames": 1,
                    "reason_counts": {reason: 1},
                }
            )
    return {
        "total_frames": len(selected),
        "valid_frames": valid,
        "coverage": valid / len(selected) if selected else None,
        "missing_reasons": dict(reasons),
        "status_counts": dict(statuses),
        "missing_intervals": intervals,
    }


def measure_window(samples, first, last):
    return measure_windows(samples, [(first, last)])
