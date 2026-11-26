import json,glob,csv,os
from pathlib import Path
H=str(Path(__file__).resolve().parents[3] / "runs/holdout21-20261125")
truth={r["video"][:-4]:r for r in csv.DictReader(open(f"{H}/truth-working.csv",encoding="utf-8-sig"))}
rounds=json.load(open(f"{H}/combined/rounds.json")); main={}
for r in rounds:
    d=(r["t_end"] or 0)-(r["t_launch"] or 0); v=r["video"][:-4]
    if v not in main or d>main[v][0]: main[v]=(d,r)
for p in sorted(glob.glob(f"{H}/combined/geometry/*.json")):
    v=os.path.basename(p).split("-")[0]; t=truth.get(v,{})
    if t.get("is_round")!="yes": continue
    g=json.load(open(p)); fps=g["fps"]; S={int(s["frame"]):s for s in g["samples"]}
    r=main[v][1]; a=int(r["t_launch"]*fps); b=int(r["t_end"]*fps); ev=float(t["loser_stop_s"].split("-")[0])*fps
    miss=[f for f in range(a,b+1) if f in S and (S[f].get("alignment") is None or S[f].get("status")=="bridged")]
    early=[f for f in miss if f<ev-6*fps]; late=[f for f in miss if ev-6*fps<=f<=ev]; post=[f for f in miss if f>ev]
    flag="calibration_geometry_missing" in r["review_reasons"]
    def span(x): return f"{x[0]/fps:.1f}-{x[-1]/fps:.1f}s" if x else "-"
    print(f"{t['no']} round {r['t_launch']:.1f}-{r['t_end']:.1f}s flag={'Y' if flag else 'n'} missing={len(miss):4d}  early={len(early):4d} [{span(early)}]  late6s={len(late):3d}  post={len(post):4d} [{span(post)}]")
