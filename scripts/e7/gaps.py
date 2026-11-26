import csv, json, sys, glob, os
from pathlib import Path
from pathlib import Path
from pathlib import Path
run=sys.argv[1]
truth=json.load(open(Path(__file__).resolve().parents[2] / "runs/e0-baseline-20261124/truth/e0-truth.json"))
for p in sorted(glob.glob(run+"/trajectories/*.csv")):
    v=os.path.basename(p).rsplit("_r",1)[0]+".mp4"
    fps=truth["videos"][v]["fps"]; rnd=truth["videos"][v]["round"]
    rows=list(csv.DictReader(open(p)))
    print("==",os.path.basename(p),"rows",len(rows),"truth stop",rnd["loser_stop_s"],"winner",rnd["winner"])
    by={}
    for r in rows:
        key=r["side"] or ("t"+r["track_id"])
        by.setdefault(key,[]).append(r)
    for k,rs in by.items():
        fr=[int(r["frame"]) for r in rs]; obs=[int(r["frame"]) for r in rs if r["x"]]
        # gaps between observed frames
        gaps=[]; obs_s=sorted(set(obs))
        for a,b in zip(obs_s,obs_s[1:]):
            if b-a>3: gaps.append((a+1,b-1))
        big=[(a,b,round(a/fps,2),round(b/fps,2)) for a,b in gaps if b-a>=5]
        print(f"  {k}: frames {min(fr)}-{max(fr)} ({min(fr)/fps:.2f}-{max(fr)/fps:.2f}s) observed {len(obs_s)} gaps>=5f: {big[:12]}{' ...' if len(big)>12 else ''}")
