import json,glob,os,csv
from collections import Counter
H="/home/<user>/bxjudge/runs/holdout21-20261125"
truth={r["video"][:-4]:r for r in csv.DictReader(open(f"{H}/truth-working.csv",encoding="utf-8-sig"))}
tot=Counter()
for p in sorted(glob.glob(f"{H}/combined/geometry/*.json")):
    v=os.path.basename(p).split("-")[0]; t=truth.get(v,{})
    if t.get("is_round")!="yes": continue
    g=json.load(open(p)); fps=g["fps"]; ph=g["phase"]; sides=g["identity"]["anchors"].get("sides",{})
    rel=ph["releases"]; D=ph["diagnostics"]
    rows=[]
    for tid,s in sorted(sides.items(),key=lambda x:int(x[0])):
        d=D.get(tid,{}); it=(d.get("interruption") or {})
        rows.append(f"{s}{tid}:first={d.get('original_first_frame',0)/fps:.2f}s rel={'%.2f'%(rel[tid]/fps) if tid in rel else '-'} body={d.get('body_track_id')} int={it.get('reason')}")
        tot[(tid in rel, it.get("reason"))]+=1
    print(t["no"],v); [print("   ",x) for x in rows]
print(tot)
