"""Check every quotable number in CLAUDE.md against results.csv AND results_v1.csv.

A figure that fails against v2 but matches v1 is not wrong -- it is a v1 figure
that was never relabelled. That distinction decides whether to fix it or mark it.
"""
import csv, collections, os, sys
import numpy as np

os.chdir(r"C:\Users\user\Desktop\Final Year Project\Dummy Simulation_Modified")

def load(path):
    rows = list(csv.DictReader(open(path, newline="")))
    text = {"run_id", "condition", "fault_type", "map_reporter"}
    for r in rows:
        for k in r:
            if k not in text:
                r[k] = float(r[k])
    return rows

V2, V1 = load("results.csv"), load("results_v1.csv")
CONDS = ["C0", "C1", "C2", "C5", "C3", "C4"]

def idx(rows):
    by = collections.defaultdict(list); sm = collections.defaultdict(dict)
    for r in rows:
        by[r["condition"]].append(r); sm[int(r["seed"])][r["condition"]] = r
    return by, sm

def mean(rs, col):
    return float(np.mean([r[col] for r in rs])) if rs else float("nan")

def per_fault(rows, cond, fault, col):
    by, _ = idx(rows)
    return mean([r for r in by[cond] if r["fault_type"] == fault], col)

def paired(rows, a, b, col, fault=None):
    _, sm = idx(rows)
    d = [sm[s][a][col] - sm[s][b][col] for s in sm
         if (fault is None or sm[s][a]["fault_type"] == fault)]
    d = np.array(d); sd = d.std(ddof=1)
    return d.mean(), (d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else float("nan"))

CHECKS = []
def chk(label, claimed, fn, tol=0.005):
    def safe(R):
        # v1 has 40 columns and no energy breakdown, so some checks simply
        # cannot be evaluated against it. That is not a failure.
        try:
            return fn(R)
        except (KeyError, IndexError):
            return float("nan")
    v2 = safe(V2); v1 = safe(V1)
    def ok(x):
        if x != x or claimed is None: return False
        rel = abs(x - claimed) / max(abs(claimed), 1e-9)
        return abs(x - claimed) <= tol or rel <= 0.005
    CHECKS.append((label, claimed, v2, v1, ok(v2), ok(v1)))

by2, sm2 = idx(V2)

# ---- headline v2 table -------------------------------------------------
head = {"truly": ("points_truly_visited", [39.83,39.55,34.31,36.66,37.10,19.24]),
        "falsely": ("points_falsely_reported", [0.00,0.00,2.21,2.17,2.03,4.86]),
        "success": ("mission_success", [0.86,0.79,0.38,0.59,0.59,0.28]),
        "duration": ("duration_s", [403,423,870,508,490,1355]),
        "J/point": ("energy_per_point_j", [155,163,292,188,186,512])}
for name,(col,vals) in head.items():
    for c,v in zip(CONDS, vals):
        chk(f"headline v2 {name} {c}", v,
            lambda R,c=c,col=col: mean(idx(R)[0][c], col),
            tol=0.6 if name in ("duration","J/point") else 0.006)

# ---- per-fault rows ----------------------------------------------------
pf = [("sensor_degradation","surface_f1",[0.693,0.691,0.696]),
      ("wrong_position","points_truly_visited",[28.14,30.00,29.57]),
      ("comms_loss","coverage_pct",[99.27,99.09,99.09]),
      ("immobilised","points_truly_visited",[29.57,36.86,39.00]),
      ("battery_drain","points_truly_visited",[39.17,39.83,40.00])]
for f,col,vals in pf:
    for c,v in zip(["C2","C5","C3"], vals):
        chk(f"per-fault {f} {col} {c}", v,
            lambda R,c=c,f=f,col=col: per_fault(R,c,f,col), tol=0.006)

# ---- paired C3 vs C2 ---------------------------------------------------
for lbl,col,dv,tv in [("truly","points_truly_visited",2.79,3.65),
                      ("success","mission_success",0.21,2.27),
                      ("duration","duration_s",-380,-2.70),
                      ("J/point","energy_per_point_j",-106,-2.31)]:
    chk(f"paired C3-C2 {lbl} diff", dv, lambda R,col=col: paired(R,"C3","C2",col)[0], tol=0.6)
    chk(f"paired C3-C2 {lbl} t", tv, lambda R,col=col: paired(R,"C3","C2",col)[1], tol=0.02)

# ---- energy composition ------------------------------------------------
cats = [("drive",32.3),("turn",6.1),("sense",44.6),("compute",13.4),("comms",3.5)]
def share(R, cat):
    tot = sum(sum(r[f"energy_{c}_j"] for r in R) for c,_ in cats)
    return 100*sum(r[f"energy_{cat}_j"] for r in R)/tot
for cat,v in cats:
    chk(f"composition {cat} %", v, lambda R,cat=cat: share(R,cat), tol=0.06)

# ---- immobilised energy ------------------------------------------------
chk("immobilised C2 total J", 18632, lambda R: per_fault(R,"C2","immobilised","total_energy_j"), tol=6)
chk("immobilised C3 total J", 6727, lambda R: per_fault(R,"C3","immobilised","total_energy_j"), tol=6)
chk("immobilised C3-C2 J/pt", -474.0, lambda R: paired(R,"C3","C2","energy_per_point_j","immobilised")[0], tol=1.0)
chk("immobilised C3-C2 t", -4.71, lambda R: paired(R,"C3","C2","energy_per_point_j","immobilised")[1], tol=0.02)

# ---- detection ---------------------------------------------------------
det = [("sensor_degradation",4,50.6),("immobilised",7,30.4),("battery_drain",6,11.3),
       ("wrong_position",2,257.5),("comms_loss",1,22.8)]
def ndet(R,f):
    by,_ = idx(R); return sum(r["fault_detected"] for r in by["C3"] if r["fault_type"]==f)
def latmean(R,f):
    by,_ = idx(R)
    v=[r["detection_latency_s"] for r in by["C3"] if r["fault_type"]==f and r["fault_detected"]==1]
    return float(np.mean(v)) if v else float("nan")
for f,n,l in det:
    chk(f"detected {f}", n, lambda R,f=f: ndet(R,f), tol=0.01)
    chk(f"latency {f}", l, lambda R,f=f: latmean(R,f), tol=0.06)

# ---- totals ------------------------------------------------------------
tot = [("C2",1059,995,64),("C5",1126,1063,63),("C3",1127,1076,59)]
for c,b,t,fa in tot:
    chk(f"total believed {c}", b, lambda R,c=c: sum(r["points_believed_visited"] for r in idx(R)[0][c]), tol=0.5)
    chk(f"total truly {c}", t, lambda R,c=c: sum(r["points_truly_visited"] for r in idx(R)[0][c]), tol=0.5)
    chk(f"total falsely {c}", fa, lambda R,c=c: sum(r["points_falsely_reported"] for r in idx(R)[0][c]), tol=0.5)
chk("cells restored C3", 268289, lambda R: sum(r["cells_restored"] for r in idx(R)[0]["C3"]), tol=1)
chk("missions C2", 11, lambda R: sum(r["mission_success"] for r in idx(R)[0]["C2"]), tol=0.01)
chk("missions C5", 17, lambda R: sum(r["mission_success"] for r in idx(R)[0]["C5"]), tol=0.01)
chk("missions C3", 17, lambda R: sum(r["mission_success"] for r in idx(R)[0]["C3"]), tol=0.01)

# ---- deviation detection & contact ------------------------------------
devs = [("C1",85.9),("C2",80.4),("C5",82.1),("C3",84.2),("C4",47.3)]
for c,v in devs:
    chk(f"deviations found {c} %", v,
        lambda R,c=c: 100*mean(idx(R)[0][c],"deviations_detected")/mean(idx(R)[0][c],"deviations_injected"),
        tol=0.06)
for c,v in [("C0",83.7),("C1",84.9),("C2",74.0),("C5",85.6),("C3",86.6)]:
    chk(f"contact {c} %", v, lambda R,c=c: 100*mean(idx(R)[0][c],"contact_fraction"), tol=0.06)
chk("deviations per round", 5.45, lambda R: mean(idx(R)[0]["C1"],"deviations_detected"), tol=0.006)

# ---- wrong_position detail --------------------------------------------
for c,v in [("C2",9.14),("C5",9.00),("C3",8.43)]:
    chk(f"wp falsely {c}", v, lambda R,c=c: per_fault(R,c,"wrong_position","points_falsely_reported"), tol=0.006)
for c,v in [("C2",196),("C5",201),("C3",229)]:
    chk(f"wp J/point {c}", v, lambda R,c=c: per_fault(R,c,"wrong_position","energy_per_point_j"), tol=0.6)

# ---- C0 vs C1 percentages ---------------------------------------------
def pct(R,col):
    by,_=idx(R); return 100*(mean(by["C1"],col)-mean(by["C0"],col))/mean(by["C0"],col)
for lbl,col,v in [("duration",  "duration_s", 4.8), ("energy","total_energy_j",3.3),
                  ("J per point","energy_per_point_j",4.8),
                  ("J per m2","energy_per_m2_j",2.4),
                  ("coverage","coverage_pct",-0.5),
                  ("surface F1","surface_f1",-1.2)]:
    chk(f"C0->C1 {lbl} %", v, lambda R,col=col: pct(R,col), tol=0.06)

# ---- named single runs -------------------------------------------------
chk("seed17 C1 error %", 19.62, lambda R: idx(R)[1][17]["C1"]["observed_error_pct"], tol=0.006)
chk("seed12 C3 error %", 25.23, lambda R: idx(R)[1][12]["C3"]["observed_error_pct"], tol=0.006)
chk("C1 believed mean", 39.34, lambda R: mean(idx(R)[0]["C1"],"points_believed_visited"), tol=0.006)

# ---- report ------------------------------------------------------------
print(f"{'claim':<38s} {'doc':>10s} {'v2':>10s} {'v1':>10s}  verdict")
print("-"*88)
bad_v2 = bad_both = 0
for lbl, claimed, v2, v1, ok2, ok1 in CHECKS:
    if ok2:
        continue
    bad_v2 += 1
    verdict = "matches v1 -- relabel" if ok1 else "MATCHES NEITHER"
    if not ok1: bad_both += 1
    print(f"{lbl:<38s} {claimed:>10.2f} {v2:>10.2f} {v1:>10.2f}  {verdict}")
print("-"*88)
print(f"{len(CHECKS)} claims checked, {len(CHECKS)-bad_v2} reproduce from v2, "
      f"{bad_v2} do not, of which {bad_both} match neither dataset.")
