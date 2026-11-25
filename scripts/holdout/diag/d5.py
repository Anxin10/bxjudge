import json,glob,csv,os,re
H="/home/<user>/bxjudge/runs/holdout21-20261125"
truth={r["video"][:-4]:r for r in csv.DictReader(open(f"{H}/truth-working.csv",encoding="utf-8-sig"))}
rounds=json.load(open(f"{H}/combined/rounds.json")); main={}
for r in rounds:
    d=(r["t_end"] or 0)-(r["t_launch"] or 0); v=r["video"][:-4]
    if v not in main or d>main[v][0]: main[v]=(d,r)
first=True
for p in sorted(glob.glob(f"{H}/combined/geometry/*.json")):
    v=os.path.basename(p).split("-")[0]; t=truth.get(v,{})
    if t.get("is_round")!="yes": continue
    g=json.load(open(p)); fps=g["fps"]; r=main[v][1]
    lt=g["launch_timing"]; idn=g["identity"]
    if first:
        print("round_releases sample:", json.dumps(lt["round_releases"])[:600]); print("anchors.evidence:", json.dumps(idn["anchors"]["evidence"])[:600]); first=False
    rel=[x for x in r["review_reasons"] if re.match(r"(launch|identity|two_bey|track_missing|landing)",x)]
    print(f"\n{t['no']} {v} t_launch={r['t_launch']:.1f} sides={idn['anchors'].get('sides')} release_inferred={idn['anchors'].get('physical_release_inferred')}")
    print("   ", rel)
