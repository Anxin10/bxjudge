import sys, math, csv
sys.path.insert(0, "/home/<user>/bxjudge/src")
import cv2, numpy as np
from pathlib import Path
from bxjudge.detect import load_detector
ROOT=Path("/home/<user>/bxjudge")
det=load_detector("rfdetr", next((ROOT/"runs/train-rfdetr-20261023b").rglob("checkpoint_best_ema.pth")), conf=0.01)
VID=ROOT/"runs/e0-baseline-20261124/videos"
RUN=ROOT/"runs/e4-eval-20261125/production"
def rows(video, tid):
    return {int(r["frame"]):r for r in csv.DictReader(open(RUN/f"trajectories/{video}_r1.csv")) if r["track_id"]==tid}
def run(video, tid, a, b, step=1):
    rs=rows(video,tid); cap=cv2.VideoCapture(str(VID/f"{video}.mp4"))
    s=cap.get(cv2.CAP_PROP_FRAME_WIDTH)/1280; cap.set(cv2.CAP_PROP_POS_FRAMES,a)
    last=None; stats=dict(n=0,gap=0,full=0,roi=0,roi_only=0,full_only=0)
    for f in range(a,b+1):
        ok,im=cap.read(); H,W=im.shape[:2]
        r=rs.get(f)
        if r and r["x"]: last=(float(r["x"])*s,float(r["y"])*s)
        if last is None or (f-a)%step: continue
        cx,cy=last
        def best(ds,ox=0,oy=0,k=1.0):
            m=0.
            for d in ds:
                if d.cls!="bey": continue
                x0,y0,x1,y1=d.box; mx=ox+(x0+x1)/2*k; my=oy+(y0+y1)/2*k
                if math.hypot(mx-cx,my-cy)<100: m=max(m,d.score)
            return m
        w,h=1280,720  # same aspect as full frame, 1.5x zoom
        x0=int(max(0,min(W-w,cx-w/2))); y0=int(max(0,min(H-h,cy-h/2)))
        fs=best(det(im,f)); cs=best(det(np.ascontiguousarray(im[y0:y0+h,x0:x0+w]),f),x0,y0)
        observed = bool(r and r["x"])
        stats["n"]+=1
        if not observed: stats["gap"]+=1
        stats["full"]+=fs>=0.3; stats["roi"]+=cs>=0.3
        stats["roi_only"]+=(cs>=0.3 and fs<0.3); stats["full_only"]+=(fs>=0.3 and cs<0.3)
    print(video,tid,a,b,stats)
run("20261024_201444","30",67,380)          # B launch phase near frame edge, many gaps
run("20261024_201331","33",3469,3530)      # stationary B after hand
run("20261024_201331","32",1000,1400,step=4)  # X mid-battle
