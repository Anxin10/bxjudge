import json,glob,csv,os
from collections import Counter
H="/home/<user>/bxjudge/runs/holdout21-20261125"
truth={r["video"][:-4]:r for r in csv.DictReader(open(f"{H}/truth-working.csv",encoding="utf-8-sig"))}
rounds=json.load(open(f"{H}/combined/rounds.json")); main={}
for r in rounds:
    d=(r["t_end"] or 0)-(r["t_launch"] or 0)
    v=r["video"][:-4]
    if v not in main or d>main[v][0]: main[v]=(d,r)
g=json.load(open(sorted(glob.glob(f"{H}/combined/geometry/*000603*"))[0]))
print("sample:",g["samples"][0])
for p in sorted(glob.glob(f"{H}/combined/geometry/*.json")):
    v=os.path.basename(p).split("-")[0]; t=truth.get(v,{})
    if t.get("is_round")!="yes": continue
    g=json.load(open(p)); fps=g["fps"]; S={int(s["frame"]):s for s in g["samples"]}
    r=main[v][1]; t0=r["t_launch"]; ev=float(t["loser_stop_s"].split("-")[0])
    win=[f for f in sorted(S) if max(t0,ev-6)*fps<=f<=ev*fps]
    bad=[f for f in win if S[f].get("status") not in ("tracking","relocated","refound","temporal_verified","conditional_gap")]
    run=best=0; prev=None
    for f in win:
        if f in bad: run=run+1 if prev is not None and prev in bad and f-prev<=2 else 1; best=max(best,run)
        prev=f
    st=Counter(S[f].get("status") for f in bad)
    print(f"{t['no']} {v} launch={t0:.1f} event={ev:.1f} win={len(win)} bad={len(bad)} ({len(bad)/max(1,len(win)):.0%}) longest={best/fps:.2f}s {dict(st)}")
print("no geometry:", [v for v,t in truth.items() if t.get('is_round')=='yes' and not glob.glob(f'{H}/combined/geometry/{v}-*.json')])
