"""
sweep_byzantine_roc.py
======================
Stage 4b -- the operating curve for the Byzantine detector.

`BYZANTINE_RATIO = 1.4` was chosen in Session 8 from three seeds, and
Session 13's M3 fix changed what the ratio is computed over without
revisiting it. The current operating point (0.2 % false flags, 2/7
detection) has therefore never been compared against the alternatives it
was chosen over. Either a better point exists or it does not, and both
answers are worth having.

TWO ARMS, BECAUSE A ROC NEEDS BOTH AXES:
  * C1 -- deviations, no faults. Every quarantine here is false by
    construction. This is the gate Session 13 re-pointed from C0, because
    C0 has no deviations and so cannot exhibit the mechanism at all.
  * C3 on the wrong_position seeds -- every detection here is a true one.

THE DEFAULT IS NOT CHANGED BY THIS SCRIPT. It reports whether a better
operating point exists and leaves the decision, explicitly, to be made and
recorded rather than slipped in. Asserts the config default at the end.

Usage:
    py sweep_byzantine_roc.py
"""

import csv
import os
import time

import numpy as np

import config
import run_experiments as rx
from faults import schedule_for_seed

# SCOPE REDUCED IN SESSION 14, DELIBERATELY AND ON THE RECORD. The
# intended sweep was five ratios over seventeen seeds -- 85 runs. Wall
# times on this machine degraded roughly tenfold mid-session under CPU
# contention (one 584 s mission took 7142 s of wall clock), so the grid was
# cut rather than the session left unfinished. Three points is a thin curve
# and is reported as such; it is still three more than the zero points the
# current operating point was chosen from.
#
# 1.15 is close to "any disagreement convicts"; 2.0 is "only a gross
# outlier convicts". 1.4 is the current default and IS NOT RUN HERE -- the
# suite already measured it, so those rows are read from results.csv
# instead of being paid for twice.
RATIOS_TO_RUN = [1.15, 2.0]
DEFAULT_RATIO_FROM_SUITE = 1.4

# Eight healthy seeds. The Session 11/13 false-positive gate used ten; the
# full 29 would treble the cost for a third-decimal change in a rate
# already near zero.
N_HEALTHY = 8
SUITE = "results.csv"
OUT = "sweep_byzantine_roc.csv"


def wrong_position_seeds():
    return [s for s in config.EXPERIMENT_SEEDS
            if rx.deployable(s)[0]
            and schedule_for_seed(s, config.EXPERIMENT_FAULT_ROBOT)[2]
            == "wrong_position"]


def healthy_seeds(n):
    return [s for s in config.EXPERIMENT_SEEDS if rx.deployable(s)[0]][:n]


def main():
    original = config.BYZANTINE_RATIO
    faulty = wrong_position_seeds()
    healthy = healthy_seeds(N_HEALTHY)

    print("BYZANTINE THRESHOLD ROC")
    print(f"  ratios run  : {RATIOS_TO_RUN}")
    print(f"  ratio {DEFAULT_RATIO_FROM_SUITE} read from {SUITE} "
          f"(config default, already measured by the suite)")
    print(f"  C3 faulty   : {faulty}")
    print(f"  C1 healthy  : {healthy}")
    print(f"  {len(RATIOS_TO_RUN) * (len(faulty) + len(healthy))} runs\n")

    existing = set()
    if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        with open(OUT, newline="") as fh:
            for r in csv.DictReader(fh):
                existing.add((r["byzantine_ratio"], r["run_id"]))

    fresh = not (os.path.exists(OUT) and os.path.getsize(OUT) > 0)
    fh = open(OUT, "a", newline="")
    # extrasaction="ignore": run_one() attaches a private "_wall_s" key.
    writer = csv.DictWriter(fh, fieldnames=["byzantine_ratio"] + rx.COLUMNS,
                            extrasaction="ignore")
    if fresh:
        writer.writeheader()
        fh.flush()

    started = time.time()
    for ratio in RATIOS_TO_RUN:
        config.BYZANTINE_RATIO = ratio
        for condition, seeds in (("C3", faulty), ("C1", healthy)):
            for seed in seeds:
                if (f"{ratio}", f"{condition}_s{seed}") in existing:
                    continue
                t = time.time()
                row = rx.run_one(condition, seed, save_trace=False)
                row["byzantine_ratio"] = ratio
                writer.writerow(row)
                fh.flush()
                print(f"  ratio {ratio:<5} {condition}_s{seed:<5} "
                      f"det {row['fault_detected']:>2}  "
                      f"Qok {row['quarantines_correct']} "
                      f"Qbad {row['quarantines_wrong']}  "
                      f"fp {row['false_positives']}  "
                      f"truly {row['points_truly_visited']:>2} "
                      f"({time.time() - t:.0f}s)")
    fh.close()
    config.BYZANTINE_RATIO = original

    with open(OUT, newline="") as f:
        rows = list(csv.DictReader(f))

    # The default-ratio point comes from the suite rather than from a
    # repeat run. Restricted to the SAME seeds as the swept arms, so the
    # three points on the curve are comparable rather than nearly so.
    if os.path.exists(SUITE):
        with open(SUITE, newline="") as f:
            for r in csv.DictReader(f):
                seed = int(r["seed"])
                if ((r["condition"] == "C3" and seed in faulty)
                        or (r["condition"] == "C1" and seed in healthy)):
                    r["byzantine_ratio"] = DEFAULT_RATIO_FROM_SUITE
                    rows.append(r)
    else:
        print(f"  (note: {SUITE} absent, default-ratio point not shown)")

    all_ratios = sorted({float(r["byzantine_ratio"]) for r in rows})

    print(f"\ncompleted in {(time.time() - started) / 60:.1f} min")
    print("\n" + "=" * 88)
    print("ROC -- detection against false quarantine")
    print("%7s | %-28s | %-34s" % ("ratio", "C3 wrong_position (n=%d)" % len(faulty),
                                   "C1 healthy (n=%d)" % len(healthy)))
    print("%7s | %9s %9s %7s | %9s %11s %9s" %
          ("", "detected", "correct Q", "truly", "false Q", "runs with Q", "believed"))
    print("-" * 88)
    for ratio in all_ratios:
        f3 = [r for r in rows
              if float(r["byzantine_ratio"]) == ratio and r["condition"] == "C3"]
        f1 = [r for r in rows
              if float(r["byzantine_ratio"]) == ratio and r["condition"] == "C1"]
        if not f3 or not f1:
            continue
        det = sum(int(r["fault_detected"]) == 1 for r in f3)
        qok = sum(int(r["quarantines_correct"]) for r in f3)
        truly = np.mean([float(r["points_truly_visited"]) for r in f3])
        qbad = sum(int(r["quarantines_wrong"]) for r in f1)
        runs_with = sum(int(r["quarantines_wrong"]) > 0 for r in f1)
        believed = np.mean([float(r["points_believed_visited"]) for r in f1])
        mark = ("  <-- current default, from the suite"
                if ratio == DEFAULT_RATIO_FROM_SUITE else "")
        print("%7.2f | %6d/%-3d %9d %7.2f | %9d %8d/%-3d %9.2f%s" %
              (ratio, det, len(f3), qok, truly,
               qbad, runs_with, len(f1), believed, mark))

    print("\nLower ratio = quicker to convict. The question is whether any")
    print("row buys detection on the left without paying for it on the right.")
    print("=" * 88)

    # Checked against the source rather than by reloading the module, for
    # the same reason as in sweep_displacement.py.
    assert config.BYZANTINE_RATIO == original, "ratio left modified"
    source = open("config.py", encoding="utf-8", errors="replace").read()
    assert f"BYZANTINE_RATIO = {original}" in source, (
        "config.py no longer declares the original BYZANTINE_RATIO -- this "
        "script reports an operating point, it does not choose one")
    print(f"config.py default verified unchanged on disk: "
          f"BYZANTINE_RATIO = {original}")


if __name__ == "__main__":
    main()
