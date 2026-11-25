"""Zoomed crop sheet + per-step motion energy. usage: zoom.py VIDEO CX CY HALF T0 T1 STEP OUT"""
import sys, cv2, numpy as np
v,cx,cy,hw,t0,t1,step,out=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),float(sys.argv[5]),float(sys.argv[6]),float(sys.argv[7]),sys.argv[8]
cap=cv2.VideoCapture(v); fps=cap.get(cv2.CAP_PROP_FPS)
f0=int(t0*fps); f1=int(t1*fps); cap.set(cv2.CAP_PROP_POS_FRAMES,f0)
tiles=[]; prev=None; every=max(1,int(round(step*fps)))
for f in range(f0,f1+1):
    ok,im=cap.read()
    if not ok: break
    c=im[cy-hw:cy+hw,cx-hw:cx+hw]
    g=cv2.cvtColor(c,cv2.COLOR_BGR2GRAY).astype(np.float32)
    if (f-f0)%every==0:
        e=float(np.abs(g-prev).mean()) if prev is not None else 0
        t=cv2.resize(c,(300,300)); cv2.rectangle(t,(0,0),(300,22),(0,0,0),-1)
        cv2.putText(t,f"{f/fps:.2f}s d={e:.1f}",(4,16),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        tiles.append(t)
    prev=g
while len(tiles)%8: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out,np.vstack([np.hstack(tiles[i:i+8]) for i in range(0,len(tiles),8)]),[cv2.IMWRITE_JPEG_QUALITY,85])
