"""Assemble E0 truth from the visually adjudicated keyframe labels (Claude, 2026-11-24; NOT human truth)."""
import json, hashlib
from pathlib import Path
T = Path("/home/<user>/bxjudge/runs/e0-baseline-20261124/truth")
K = json.loads((T / "keyframes.json").read_text())
# One token per detector box, in box order: X / B / U(unknown) / N(not a bey) / D(duplicate box of an already-labelled bey).
# A trailing "+m" means a visible bey had no box (detector miss) on that keyframe.
L = {
 "20261024_201331": ("XB XB BX BX BX BX BX BX BX XB XB XB BX XB BX BX XB BX BX BX "
                     "XB BX BX XB BX XB XB X+m XB BX XB BX BX BX XB BX BX BX BX BX "
                     "XB BX BX BX BX XN+m XN+m XU BX BX BXD BX BX XB XB BX BX XB XB XB "
                     "XB BX BX BX BXD XB BX BX XB XB XB XB XB XB BX BX BX XB BX BX "
                     "XB XB BX BX BX BX BX XB BX BX BX"),
 "20261024_201410": ("XB XB BX XB BX BX XB XB XB XB XB BX XB BX XB XB BX BX BX BX "
                     "BX XB XU XU XB XB XB BX BX XB BX XB XB BX XB UX BX BX BX XB "
                     "XB BX BX XB XB UU BX XU XB BU XB XB XB XB XB XB UX BX XB BX XU XU"),
 "20261024_201444": ("XB BX XB XB XB XB BX BX XBD BX BX XB BX XB BX BX BX XB XB BX "
                     "BX BX XB BX XB BX XB XB XB BX BX XB XB BX BX BX XB XB XB XB "
                     "XB BX XB XB XB XB XB BX BX BX BX BX BX BX BX BX BXD XB BX"),
}
ROUNDS = {  # times in seconds of the source video; intervals bracket the visually uncertain moment
 "20261024_201331": dict(x_look="green", b_look="blue/silver with gold", launch_s=[5.20, 5.30],
                         finish="spin", winner="X", loser_stop_s=[28.80, 29.25], confidence="high", human_verified=True,
                         note="Human-verified: X(green) wins by spin; B(blue/silver) settles stationary around ~28.80-29.25 s while X keeps spinning until pick-up"),
 "20261024_201410": dict(x_look="orange cap, silver ring", b_look="dark red centre, silver ring, blue bit", launch_s=[3.30, 3.40],
                         finish="spin", winner="B", loser_stop_s=[18.80, 18.95], confidence="high", human_verified=True,
                         note="Human-verified: X(orange) loses by spin, tipping over at ~18.85 s; B upright until picked up at ~19.15 s"),
 "20261024_201444": dict(x_look="black/red with orange bit", b_look="blue/purple with silver ring", launch_s=[4.05, 4.20],
                         finish="spin", winner="B", loser_stop_s=[19.00, 19.10], confidence="high", ambiguous=False, human_verified=True,
                         note="Human-verified: B(blue/purple) wins by spin; X(black/red) tips over and stops around ~19.00-19.10 s"),
}
truth = {"schema": "bxjudge.e0-truth.v1", "made_by": "Human-verified round winners & X/B colours (2026-11-24); keyframe box sides assigned by colour appearance",
         "x_definition": "X = the top released by the image-left launcher at the start of the round",
         "keyframe_step_s": 0.25, "detector_for_candidate_boxes": "weights/selected.json (production), conf 0.2, class bey", "videos": {}}
for v, s in L.items():
    toks = s.split(); frames = K[v]["frames"]
    assert len(toks) == len(frames), (v, len(toks), len(frames))
    out = []
    for tok, fr in zip(toks, frames):
        lab, _, miss = tok.partition("+")
        assert len(lab) == len(fr["boxes"]), (v, fr["frame"], tok, len(fr["boxes"]))
        out.append({"frame": fr["frame"], "t": fr["t"], "objects": [{"box_xyxy": b, "side": c} for b, c in zip(fr["boxes"], lab)],
                    "missed_beys": 1 if miss else 0})
    truth["videos"][v + ".mp4"] = {"fps": K[v]["fps"], "coordinate_size": [1920, 1080], "round": ROUNDS[v], "keyframes": out}
p = T / "e0-truth.json"; p.write_text(json.dumps(truth, indent=1, ensure_ascii=False))
from collections import Counter
c = Counter(o["side"] for v in truth["videos"].values() for k in v["keyframes"] for o in k["objects"])
print("wrote", p, "sha256", hashlib.sha256(p.read_bytes()).hexdigest()[:16], dict(c), "missed", sum(k["missed_beys"] for v in truth["videos"].values() for k in v["keyframes"]))
