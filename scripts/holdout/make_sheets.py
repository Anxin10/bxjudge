"""Holdout review sheets: one overview per video, a frame every STEP seconds, timestamped.
Independent of any pipeline output so the human label is not anchored by predictions.
usage: make_sheets.py VIDEO_LIST OUT_DIR [--step 0.5] [--start S --end E --tag fine]"""
import argparse, csv
from pathlib import Path
import cv2, numpy as np
ap=argparse.ArgumentParser(); ap.add_argument("videos"); ap.add_argument("out")
ap.add_argument("--step",type=float,default=0.5); ap.add_argument("--cols",type=int,default=6)
ap.add_argument("--width",type=int,default=480); ap.add_argument("--only")
ap.add_argument("--start",type=float); ap.add_argument("--end",type=float); ap.add_argument("--tag",default="overview")
a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
for line in open(a.videos):
    p=Path(line.strip())
    if not p.name or (a.only and a.only not in p.name): continue
    cap=cv2.VideoCapture(str(p)); fps=cap.get(cv2.CAP_PROP_FPS); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t0=a.start or 0.0; t1=min(a.end or n/fps, n/fps)
    want=[]; t=t0
    while t<t1-1e-6: want.append(int(round(t*fps))); t+=a.step
    tiles=[]; f=0; wi=0; ok=True
    while wi<len(want) and ok:
        ok=cap.grab()
        if not ok: break
        if f==want[wi]:
            _,im=cap.retrieve(); h=int(im.shape[0]*a.width/im.shape[1])
            im=cv2.resize(im,(a.width,h),interpolation=cv2.INTER_AREA)
            label=f"{f/fps:6.2f}s  f{f}"
            cv2.rectangle(im,(0,0),(a.width,26),(0,0,0),-1)
            cv2.putText(im,label,(6,19),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1,cv2.LINE_AA)
            tiles.append(im); wi+=1
        f+=1
    if not tiles: continue
    h=tiles[0].shape[0]; blank=np.zeros_like(tiles[0])
    while len(tiles)%a.cols: tiles.append(blank)
    rows=[np.hstack(tiles[i:i+a.cols]) for i in range(0,len(tiles),a.cols)]
    # split very tall sheets into pages of 12 rows
    for pi in range(0,len(rows),12):
        page=np.vstack(rows[pi:pi+12]); suffix="" if len(rows)<=12 else f"_p{pi//12+1}"
        cv2.imwrite(str(out/f"{p.stem}_{a.tag}{suffix}.jpg"),page,[cv2.IMWRITE_JPEG_QUALITY,82])
    print(p.name,f"{n/fps:.1f}s",len(want),"tiles")
