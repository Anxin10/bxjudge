"""E0 identity keyframes: every 0.25 s inside each round window; candidate bey boxes from the production detector (tracker-independent)."""
import json, cv2, numpy as np, sys
from pathlib import Path
sys.path.insert(0, "/home/<user>/bxjudge/src")
from bxjudge.detect import load_detector
E = Path("/home/<user>/bxjudge/runs/e0-baseline-20261124/truth"); E.mkdir(parents=True, exist_ok=True)
W = Path("/mnt/c/Users/<user>/AppData/Local/Temp/claude/0d05d5ae05dd15db/76cf77f2-3949-4532-8b09-753d801c1ee2/scratchpad/w/kf"); W.mkdir(exist_ok=True)
WIN = {"20261024_201331": (5.5, 28.0), "20261024_201410": (3.6, 19.0), "20261024_201444": (4.4, 19.0)}
sel = json.load(open("/home/<user>/bxjudge/weights/selected.json"))
det = load_detector("rfdetr", sel["weights"], conf=0.2)
out = {}
for v, (t0, t1) in WIN.items():
    cap = cv2.VideoCapture(f"/home/<user>/bx/videos/20261024/{v}.mp4"); fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    t = t0
    while t <= t1 + 1e-6:
        f = int(round(t * fps)); cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read(); t += 0.25
        if not ok: continue
        ds = [d for d in det(im, f) if d.cls == "bey"]
        frames.append({"frame": f, "t": round(f / fps, 4), "boxes": [[round(float(x), 1) for x in d.box] for d in ds], "scores": [round(float(d.score), 3) for d in ds]})
        cv2.imwrite(str(E / f"{v}_{f:05d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    out[v] = {"fps": fps, "frames": frames}
    print(v, len(frames), "keyframes", sum(len(x["boxes"]) for x in frames), "boxes")
(E / "keyframes.json").write_text(json.dumps(out, indent=1))
