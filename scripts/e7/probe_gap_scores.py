import sys, math, json
sys.path.insert(0, "/home/<user>/bxjudge/src")
import cv2, numpy as np
from pathlib import Path
from bxjudge.detect import load_detector
ROOT=Path("/home/<user>/bxjudge")
det=load_detector("rfdetr", next((ROOT/"runs/train-rfdetr-20261023b").rglob("checkpoint_best_ema.pth")), conf=0.01)
VID=ROOT/"runs/e0-baseline-20261124/videos"
# (video, center in 1280-coords, frames)
cases=[("20261024_201331",(810,437),3436,3472)]
for video,(px,py),a,b in cases:
    cap=cv2.VideoCapture(str(VID/f"{video}.mp4")); s=cap.get(cv2.CAP_PROP_FRAME_WIDTH)/1280
    cx,cy=px*s,py*s; cap.set(cv2.CAP_PROP_POS_FRAMES,a)
    for f in range(a,b+1):
        ok,im=cap.read(); H,W=im.shape[:2]
        def near(ds,ox=0,oy=0):
            best=(0.,None)
            for d in ds:
                x0,y0,x1,y1=d.box; mx=ox+(x0+x1)/2; my=oy+(y0+y1)/2
                if math.hypot(mx-cx,my-cy)<120 and d.score>best[0]: best=(d.score,d.cls)
            return best
        R=320; x0=int(max(0,min(W-2*R,cx-R))); y0=int(max(0,min(H-2*R,cy-R)))
        fs=near(det(im,f)); cs=near(det(np.ascontiguousarray(im[y0:y0+2*R,x0:x0+2*R]),f),x0,y0)
        print(f, "full %.2f %s"%(fs[0],fs[1]), " roi %.2f %s"%(cs[0],cs[1]))
        if f in (3440,3458,3466): cv2.imwrite(f"/home/<user>/bxjudge/runs/e7-probe-{video}-{f}.jpg", im[y0:y0+2*R,x0:x0+2*R])
