"""E1c metrics per run: launch clock, setup pairing, geometry-missing, verdicts, identity coverage.
usage: e1c_eval.py SET RUN_DIR... (SET = holdout | idtest)"""
import json,glob,os,csv,sys,re
R="/home/<user>/bxjudge/runs"
L={"20261025_000157":2.1,"20261025_000239":2.7,"20261025_000458":16.9,"20261025_000545":5.7,"20261025_000603":5.37,
   "20261025_001054":5.5,"20261025_001118":6.9,"20261025_001200":7.8,"20261025_001425":11.05,"20261025_001454":10.2,
   "20261025_001522":21.1,"20261025_002731":13.1,"20261025_002801":12.3,"20261025_001330":18.4,
   "20261024_201331":5.25,"20261024_201410":3.35,"20261024_201444":4.1}
if sys.argv[1]=="holdout":
    real={r["video"][:-4] for r in csv.DictReader(open(f"{R}/holdout21-20261125/truth-working.csv",encoding="utf-8-sig")) if r["is_round"]=="yes"}
    truthw={r["video"][:-4]:r["winner"] for r in csv.DictReader(open(f"{R}/holdout21-20261125/truth-working.csv",encoding="utf-8-sig"))}
else:
    t=json.load(open(f"{R}/e0-baseline-20261124/truth/e0-truth.json"))
    real={v[:-4] for v in t["videos"]}; truthw={v[:-4]:d["round"]["winner"] for v,d in t["videos"].items()}
for run in sys.argv[2:]:
    rows=json.load(open(f"{run}/rounds.json")); main={}
    for r in rows:
        v=r["video"][:-4]
        if v not in real: continue
        # main round = the one containing the true launch (or the longest)
        lt=L.get(v); d=(r["t_end"] or 0)-(r["t_launch"] or 0)
        cover = lt is not None and r["t_launch"]-0.6<=lt<=r["t_end"]
        key=(cover,d)
        if v not in main or key>main[v][0]: main[v]=(key,r)
    unk=geo=setup_pair=verd=right=0; errs=[]; cov=[]
    for v,(k,r) in sorted(main.items()):
        rs=r["review_reasons"]
        unk+=sum(f"launch_timing_unknown@{s}" in rs for s in "XB")
        geo+="calibration_geometry_missing" in rs
        if r["winner"] and not r["needs_review"]:
            verd+=1; right+= r["winner"]==truthw.get(v)
        if v in L: errs.append(r["t_launch"]-L[v])
        g=glob.glob(f"{run}/geometry/{v}-*.json")
        if g:
            G=json.load(open(g[0])); rec=G["two_bey_tracking"]["rounds"].get(str(r["round"]))
            if rec and rec["pair_confirmed_at_frame"] is not None and v in L and rec["pair_confirmed_at_frame"]/G["fps"]<L[v]-0.2: setup_pair+=1
        p=f"{run}/trajectories/{v}_r{r['round']}.csv"
        if os.path.exists(p):
            obs=[x for x in csv.DictReader(open(p)) if x["x"] and x["cls"]=="bey"]
            cov.append(sum(x["side"] in ("X","B") for x in obs)/max(1,len(obs)))
    n=len(main); ae=sorted(abs(e) for e in errs)
    print(f"{os.path.basename(run):16} rounds={n} launch_unknown_sides={unk}/{2*n} geometry_missing={geo}/{n} "
          f"paired_before_launch={setup_pair}/{n} |t_launch err| median={ae[len(ae)//2]:.2f}s max={ae[-1]:.2f}s "
          f"verdicts={verd} correct={right} side_cov={sum(cov)/max(1,len(cov)):.3f}")
