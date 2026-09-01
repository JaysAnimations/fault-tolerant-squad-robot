"""
Stage 2 verification -- comms loss before and after, on the five affected
seeds.

THE COMPARISON HAS TO CONTROL FOR THE COEFFICIENT CHANGE. results_v1.csv
was produced under the old coefficients, so C3-before cannot be read off
it and compared with C3-after; every joule moved for an unrelated reason.
So all three arms here are run under the NEW coefficients and differ only
in behaviour:

    C2                      no recovery at all (the naive baseline)
    C3 with the flag True   the old behaviour: reallocate an isolated robot
    C3 with the flag False  the new behaviour: observe and log only

C5 is included because v1 suggests it is the arm that actually explains
the energy gap.

Writes nothing to results.csv.
"""
import sys, os, csv, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import run_experiments as rx

SEEDS = [7, 11, 15, 20, 27]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "stage2_commsloss.csv")

ARMS = [
    ("C2", False, "C2 naive"),
    ("C5", False, "C5 timeout realloc, no detection"),
    ("C3", True, "C3 OLD: comms loss reallocates"),
    ("C3", False, "C3 NEW: comms loss observed only"),
]

rows = []
for condition, flag, label in ARMS:
    config.RECOVERY_COMMS_LOSS_REALLOCATE = flag
    for seed in SEEDS:
        t = time.time()
        r = rx.run_one(condition, seed, save_trace=False)
        r["arm"] = label
        r["comms_realloc_flag"] = flag
        rows.append(r)
        print("  %-38s seed %-5d %7.1f J  %6.1f J/pt  truly %2s  "
              "cov %6s  realloc %s   (%.0fs)"
              % (label, seed, float(r["total_energy_j"]),
                 float(r["energy_per_point_j"]), r["points_truly_visited"],
                 r["coverage_pct"], r["points_reallocated"], time.time() - t))

with open(OUT, "w", newline="") as fh:
    # extrasaction="ignore": run_one() attaches a private "_wall_s" key
    # that is not one of the CSV columns, and DictWriter raises on any
    # extra key by default.
    w = csv.DictWriter(fh, fieldnames=["arm", "comms_realloc_flag"]
                       + rx.COLUMNS, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)

# ---- summary --------------------------------------------------------
print("\n" + "=" * 78)
print("MEANS OVER THE FIVE comms_loss SEEDS (new coefficients throughout)")
print("%-38s %10s %9s %8s %8s %8s" %
      ("arm", "energy J", "J/point", "truly", "cov %", "realloc"))
for condition, flag, label in ARMS:
    sub = [r for r in rows if r["arm"] == label]
    n = len(sub)
    print("%-38s %10.1f %9.1f %8.2f %8.2f %8.2f" % (
        label,
        sum(float(r["total_energy_j"]) for r in sub) / n,
        sum(float(r["energy_per_point_j"]) for r in sub) / n,
        sum(float(r["points_truly_visited"]) for r in sub) / n,
        sum(float(r["coverage_pct"]) for r in sub) / n,
        sum(float(r["points_reallocated"]) for r in sub) / n))

def mean(label, col):
    sub = [r for r in rows if r["arm"] == label]
    return sum(float(r[col]) for r in sub) / len(sub)

c2 = mean("C2 naive", "energy_per_point_j")
old = mean("C3 OLD: comms loss reallocates", "energy_per_point_j")
new = mean("C3 NEW: comms loss observed only", "energy_per_point_j")
c5 = mean("C5 timeout realloc, no detection", "energy_per_point_j")
print("\nC3 old vs C2 : %+.1f J/point (%+.1f %%)" % (old - c2, 100 * (old - c2) / c2))
print("C3 new vs C2 : %+.1f J/point (%+.1f %%)" % (new - c2, 100 * (new - c2) / c2))
print("C3 new vs C5 : %+.1f J/point" % (new - c5))
print("change due to the Stage 2 flag alone: %+.1f J/point" % (new - old))
print("\ntruly inspected -- C2 %.2f  C5 %.2f  C3 old %.2f  C3 new %.2f"
      % (mean("C2 naive", "points_truly_visited"),
         mean("C5 timeout realloc, no detection", "points_truly_visited"),
         mean("C3 OLD: comms loss reallocates", "points_truly_visited"),
         mean("C3 NEW: comms loss observed only", "points_truly_visited")))
print("=" * 78)
print("wrote", OUT)
