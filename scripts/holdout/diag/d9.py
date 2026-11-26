import json,csv
from collections import Counter
from pathlib import Path
H=str(Path(__file__).resolve().parents[3] / "runs/holdout21-20261125")
truth={r["video"]:r for r in csv.DictReader(open(f"{H}/truth-working.csv",encoding="utf-8-sig"))}
for run in ("baseline","combined"):
    rows=json.load(open(f"{H}/{run}/rounds.json")); per=Counter(r["video"] for r in rows)
    short=[r for r in rows if (r["t_end"]-r["t_launch"])<2.0]
    nr=[v for v,t in truth.items() if t["is_round"]!="yes"]
    print(run,"rounds",len(rows),"short<2s",len(short),"rounds in non-round videos",sum(per[v] for v in nr),
          "real videos with >1 round",sum(1 for v,t in truth.items() if t["is_round"]=="yes" and per[v]>1))
    rs=Counter(x for r in short for x in r["review_reasons"] if not x.startswith("video_anomaly"))
    print("   short-round reasons:",[k for k,_ in rs.most_common(6)])
