import json,glob,csv,os
from collections import Counter
from pathlib import Path
H=str(Path(__file__).resolve().parents[3] / "runs/holdout21-20261125")
truth={r["video"][:-4]:r for r in csv.DictReader(open(f"{H}/truth-working.csv",encoding="utf-8-sig"))}
rounds=json.load(open(f"{H}/combined/rounds.json")); main={}
for r in rounds:
    d=(r["t_end"] or 0)-(r["t_launch"] or 0); v=r["video"][:-4]
    if v not in main or d>main[v][0]: main[v]=(d,r)
tot=Counter()
for p in sorted(glob.glob(f"{H}/combined/geometry/*.json")):
    v=os.path.basename(p).split("-")[0]; t=truth.get(v,{})
    if t.get("is_round")!="yes": continue
    g=json.load(open(p)); r=main[v][1]; sides=g["identity"]["anchors"].get("sides",{})
    out=[]
    for s in ("X","B"):
        n=sum(1 for x in sides.values() if x==s)
        unk=f"launch_timing_unknown@{s}" in r["review_reasons"]
        cat=("fragmented" if n>1 else "no_anchor" if n==0 else "release_unobserved") if unk else "ok"
        tot[cat]+=1; out.append(f"{s}:{n}trk {cat}")
    print(t["no"],v,"  ".join(out))
print(dict(tot))
