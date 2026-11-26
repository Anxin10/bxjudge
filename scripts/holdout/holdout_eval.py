"""Holdout-21 evaluation: verdicts vs truth, identity coverage, blockers.
usage: holdout_eval.py TRUTH_CSV RUN_DIR [RUN_DIR ...]"""
import csv, json, sys, re, glob, os
from collections import Counter
truth={r["video"]:r for r in csv.DictReader(open(sys.argv[1],encoding="utf-8-sig"))}
def main_rounds(run):
    rows=json.load(open(f"{run}/rounds.json")); m={}
    for r in rows:
        d=(r["t_end"] or 0)-(r["t_launch"] or 0)
        if r["video"] not in m or d>m[r["video"]][0]: m[r["video"]]=(d,r)
    return {v:r for v,(d,r) in m.items()}
def side_cov(run, video, rnd):
    p=f"{run}/trajectories/{video[:-4]}_r{rnd}.csv"
    if not os.path.exists(p): return None
    obs=[r for r in csv.DictReader(open(p)) if r["x"] and r["cls"]=="bey"]
    if not obs: return 0.0
    return sum(1 for r in obs if r["side"] in ("X","B"))/len(obs)
out={}
for run in sys.argv[2:]:
    m=main_rounds(run); res=dict(rounds=0,verdicts=0,winner_correct=0,winner_wrong=0,finish_correct=0,stop_err=[],cov=[],reasons=Counter())
    for v,t in truth.items():
        if t["is_round"]!="yes": continue
        r=m.get(v)
        if r is None: continue
        res["rounds"]+=1
        c=side_cov(run,v,r["round"]); res["cov"].append(c or 0.0)
        res["reasons"].update({re.sub(r"@.*","",x) for x in r["review_reasons"] if not x.startswith("video_anomaly")})
        if r["winner"] and not r["needs_review"]:
            res["verdicts"]+=1
            if t["winner"]:
                res["winner_correct" if r["winner"]==t["winner"] else "winner_wrong"]+=1
            if r["finish"]==t["finish"]: res["finish_correct"]+=1
    cov=res["cov"]
    print(f"== {run}")
    print(f"  real rounds {res['rounds']}, verdicts {res['verdicts']}, winner correct {res['winner_correct']}, wrong {res['winner_wrong']}")
    print(f"  identity coverage (observed bey rows with X/B) mean {sum(cov)/max(1,len(cov)):.3f}; rounds with any >0: {sum(1 for c in cov if c>0)}/{len(cov)}")
    print("  top blockers:", ", ".join(f"{k}({n})" for k,n in res["reasons"].most_common(8)))
    out[run]={k:(dict(v) if isinstance(v,Counter) else v) for k,v in res.items()}
json.dump(out,open(__import__("pathlib").Path(__file__).resolve().parents[2] / "runs/holdout21-20261125/holdout-eval.json","w"),ensure_ascii=False,indent=1)
