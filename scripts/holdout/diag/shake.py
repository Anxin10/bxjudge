import cv2,numpy as np,sys
from pathlib import Path
V=str(Path.home() / "bx/videos/20261024") + "/"
for no,name in [("01","20261025_000157"),("04","20261025_000458"),("13","20261025_001454"),("03","20261025_000239"),("19","20261025_002801")]:
    cap=cv2.VideoCapture(V+name+".mp4"); fps=cap.get(5); n=int(cap.get(7))
    ref=None; shifts=[]
    for t in np.arange(0,n/fps-0.5,0.5):
        cap.set(1,int(t*fps)); ok,im=cap.read()
        if not ok: break
        g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)[700:1050,200:1720].astype(np.float32)
        g=cv2.resize(g,None,fx=0.5,fy=0.5)
        if ref is None: ref=g; continue
        (dx,dy),resp=cv2.phaseCorrelate(ref,g); shifts.append((t,dx*2,dy*2,resp))
    a=np.array([[s[1],s[2]] for s in shifts if s[3]>0.1])
    big=[f"{s[0]:.1f}s({s[1]:+.0f},{s[2]:+.0f})" for s in shifts if abs(s[1])>4 or abs(s[2])>4]
    print(no,name,f"median |shift| {np.median(np.hypot(a[:,0],a[:,1])):.1f}px max {np.hypot(a[:,0],a[:,1]).max():.1f}px  >4px at:",big[:8])
