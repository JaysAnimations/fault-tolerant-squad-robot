"""
The suite, read back. Tables only -- no figures this session.

Validates the dataset first, because a headline computed from a file with a
half-paired seed or a stray -1 averaged into a mean is worse than no
headline at all.
"""
import csv
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\User\Desktop\Final Year Project\Project"
                   r"\Final Year Project\Dummy Simulation_Modified")
import config  # noqa: E402
import run_experiments as rx  # noqa: E402

CSV = (r"C:\Users\User\Desktop\Final Year Project\Project"
       r"\Final Year Project\Dummy Simulation_Modified\results.csv")

rows = list(csv.DictReader(open(CSV, newline="")))
NA = -1


def num(r, k):
    return float(r[k])


# ------------------------------------------------------------------ checks
print("=" * 78)
print("  DATASET VALIDATION")
print("=" * 78)
problems = []

if len(rows) != 174:
    problems.append(f"expected 174 rows, got {len(rows)}")

by_seed = {}
for r in rows:
    by_seed.setdefault(r["seed"], []).append(r["condition"])
unpaired = {s: sorted(c) for s, c in by_seed.items() if len(c) != 6}
if unpaired:
    problems.append(f"half-paired seeds: {unpaired}")

for r in rows:
    for col in rx.COLUMNS:
        if col not in r or r[col] is None or r[col] == "":
            problems.append(f"{r['run_id']}: '{col}' blank")

# Paired seeding: the fault must be identical across the faulted conditions
for seed, _ in by_seed.items():
    mine = [r for r in rows if r["seed"] == seed and r["fault_type"] != "none"]
    if len({r["fault_type"] for r in mine}) > 1:
        problems.append(f"seed {seed}: fault type differs between conditions")
    if len({r["points_total"] for r in rows if r["seed"] == seed}) > 1:
        problems.append(f"seed {seed}: point count differs between conditions")

print(f"  rows            : {len(rows)}")
print(f"  seeds           : {len(by_seed)} (all six conditions each)")
print(f"  seed 14 present : {'YES -- should be excluded' if '14' in by_seed else 'no, correctly excluded'}")
print(f"  problems        : {problems if problems else 'NONE'}")

# --------------------------------------------------------------- headline
print()
print("=" * 78)
print("  HEADLINE -- C2 vs C5 vs C3, the faulted arms")
print("=" * 78)
print(f"  {'':<22s}" + "".join(f"{c:>12s}" for c in ("C2", "C5", "C3")))


def arm(cond):
    return [r for r in rows if r["condition"] == cond]


def total(cond, col):
    return sum(int(num(r, col)) for r in arm(cond))


for label, col in (("believed inspected", "points_believed_visited"),
                   ("truly inspected", "points_truly_visited"),
                   ("falsely reported", "points_falsely_reported"),
                   ("points total", "points_total")):
    print(f"  {label:<22s}" + "".join(f"{total(c, col):>12d}"
                                      for c in ("C2", "C5", "C3")))

print(f"  {'over-report':<22s}"
      + "".join(f"{total(c,'points_believed_visited') - total(c,'points_truly_visited'):>+12d}"
                for c in ("C2", "C5", "C3")))
print(f"  {'missions succeeded':<22s}"
      + "".join(f"{total(c,'mission_success'):>9d}/{len(arm(c)):<2d}"
                for c in ("C2", "C5", "C3")))
for label, col in (("mean duration s", "duration_s"),
                   ("mean energy J", "total_energy_j"),
                   ("mean coverage %", "coverage_pct"),
                   ("mean surface F1", "surface_f1"),
                   ("mean obs error %", "observed_error_pct")):
    print(f"  {label:<22s}"
          + "".join(f"{np.mean([num(r, col) for r in arm(c)]):>12.2f}"
                    for c in ("C2", "C5", "C3")))
for label, col in (("quarantines correct", "quarantines_correct"),
                   ("quarantines wrong", "quarantines_wrong"),
                   ("cells restored", "cells_restored"),
                   ("points reallocated", "points_reallocated")):
    print(f"  {label:<22s}" + "".join(f"{total(c, col):>12d}"
                                      for c in ("C2", "C5", "C3")))

# ------------------------------------------------------- C0 vs C1, C4
print()
print("=" * 78)
print("  C0 vs C1 -- what an out-of-date drawing costs (no faults either way)")
print("=" * 78)
for label, col, fmt in (("points truly visited", "points_truly_visited", "{:>12.2f}"),
                        ("missions succeeded", "mission_success", "{:>12.2f}"),
                        ("duration s", "duration_s", "{:>12.1f}"),
                        ("energy J", "total_energy_j", "{:>12.0f}"),
                        ("energy per m2 J", "energy_per_m2_j", "{:>12.3f}"),
                        ("distance m", "distance_total_m", "{:>12.1f}"),
                        ("coverage %", "coverage_pct", "{:>12.2f}"),
                        ("surface F1", "surface_f1", "{:>12.4f}"),
                        ("false positives", "false_positives", "{:>12.2f}")):
    a = np.mean([num(r, col) for r in arm("C0")])
    b = np.mean([num(r, col) for r in arm("C1")])
    delta = 100 * (b - a) / a if a else float("nan")
    print(f"  {label:<22s}" + fmt.format(a) + fmt.format(b)
          + f"   {delta:>+7.1f} %")
print(f"  {'(columns: C0, C1, change)':<22s}")

print()
print("=" * 78)
print("  C4 -- one robot, faults, no reallocation possible")
print("=" * 78)
c4 = arm("C4")
c3 = arm("C3")
print(f"  truly inspected   C4 {sum(int(num(r,'points_truly_visited')) for r in c4):>4d}"
      f" / {sum(int(num(r,'points_total')) for r in c4):<4d}"
      f"     C3 {sum(int(num(r,'points_truly_visited')) for r in c3):>4d}"
      f" / {sum(int(num(r,'points_total')) for r in c3):<4d}")
print(f"  missions ok       C4 {sum(int(num(r,'mission_success')) for r in c4):>4d}"
      f" / {len(c4):<4d}     C3 {sum(int(num(r,'mission_success')) for r in c3):>4d}"
      f" / {len(c3):<4d}")

# --------------------------------------------------- per fault, per arm
print()
print("=" * 78)
print("  EACH FAULT ON THE METRIC IT DAMAGES -- 29 seeds")
print("=" * 78)
SCORING = {
    "sensor_degradation": [("surface F1", "surface_f1", "hi", "{:.3f}"),
                           ("observed err %", "observed_error_pct", "lo", "{:.2f}")],
    "wrong_position": [("truly inspected", "points_truly_visited", "hi", "{:.2f}"),
                       ("falsely reported", "points_falsely_reported", "lo", "{:.2f}")],
    "comms_loss": [("coverage %", "coverage_pct", "hi", "{:.2f}"),
                   ("surface F1", "surface_f1", "hi", "{:.3f}")],
    "immobilised": [("points visited", "points_visited", "hi", "{:.2f}")],
    "battery_drain": [("points visited", "points_visited", "hi", "{:.2f}"),
                      ("missions ok", "mission_success", "hi", "{:.2f}")],
}
print(f"  {'fault':<20s} {'n':>3s} {'metric':<17s} {'dir':<4s}"
      + "".join(f"{c:>10s}" for c in ("C2", "C5", "C3")) + "  verdict")
held = 0
for fault, metrics in SCORING.items():
    n = len([r for r in arm("C2") if r["fault_type"] == fault])
    for i, (label, col, direction, fmt) in enumerate(metrics):
        vals = {}
        for c in ("C2", "C5", "C3"):
            mine = [r for r in arm(c) if r["fault_type"] == fault]
            vals[c] = float(np.mean([num(r, col) for r in mine])) if mine else float("nan")
        ok = vals["C3"] > vals["C2"] if direction == "hi" else vals["C3"] < vals["C2"]
        if i == 0 and ok:
            held += 1
        print(f"  {(fault if i == 0 else ''):<20s} {(n if i == 0 else ''):>3} "
              f"{label:<17s} {direction:<4s}"
              + "".join(f"{fmt.format(vals[c]):>10s}" for c in ("C2", "C5", "C3"))
              + f"  {'holds' if ok else 'FAILS'}")
print(f"\n  PRIMARY METRIC HOLDS ON {held} OF {len(SCORING)} FAULTS")

# ------------------------------------------------------- detection
print()
print("=" * 78)
print("  DETECTION, on the faulted conditions that detect (C3)")
print("=" * 78)
print(f"  {'fault':<20s} {'n':>4s} {'detected':>10s} {'mean latency s':>15s} "
      f"{'Q ok':>6s} {'Q wrong':>8s}")
for fault in config.FAULT_TYPES:
    mine = [r for r in arm("C3") if r["fault_type"] == fault]
    det = [r for r in mine if int(num(r, "fault_detected")) == 1]
    lat = [num(r, "detection_latency_s") for r in det
           if num(r, "detection_latency_s") != NA]
    print(f"  {fault:<20s} {len(mine):>4d} {len(det):>7d}/{len(mine):<2d} "
          f"{(f'{np.mean(lat):.1f}' if lat else '-'):>15s} "
          f"{sum(int(num(r,'quarantines_correct')) for r in mine):>6d} "
          f"{sum(int(num(r,'quarantines_wrong')) for r in mine):>8d}")

fp0 = sum(int(num(r, "false_positives")) for r in arm("C0"))
fp1 = sum(int(num(r, "false_positives")) for r in arm("C1"))
print(f"\n  false positives on healthy squads: C0 {fp0} across {len(arm('C0'))} "
      f"runs, C1 {fp1} across {len(arm('C1'))} runs")
print(f"  C0 believed {total('C0','points_believed_visited')} vs truly "
      f"{total('C0','points_truly_visited')};  C1 believed "
      f"{total('C1','points_believed_visited')} vs truly "
      f"{total('C1','points_truly_visited')}")
print("=" * 78)
