"""
sweep_displacement.py
=====================
Stage 4a -- how far does a robot have to be displaced before the squad can
tell?

THIS IS CHARACTERISATION, NOT TUNING, AND THE DISTINCTION IS THE POINT.
The 174-run suite detected `wrong_position` in 2 of 7 runs at 6 m, which
reads like a broken detector. But 6 m is one arbitrary point on a curve
nobody has measured, and a detector that fails at 6 m may be perfectly
serviceable at 12 m. Measuring the curve turns "our detector does not
work" into a stated operating envelope.

**`FAULT_POSE_OFFSET_M` IS NOT CHANGED BY THIS SCRIPT.** The prohibition
from Session 8 stands: raising the injected displacement to flatter the
detector would be tuning the fault to fit the instrument, and it is the
one move that would genuinely damage the work. The offset is overridden
inside this process only, for the duration of the sweep, and the script
asserts at the end that config.py still reads 6.0 m.

WHY 24 m IS THE PHYSICALLY INTERESTING POINT. 6.0 m is the pipe-rack
column spacing, so the default fault models a robot matching its scan to
bay N+1 instead of bay N. 12 m and 24 m are two and four bays. A robot
mistaking bay 3 for bay 7 is a MORE realistic aliasing failure than a
6 m slip, not a less realistic one -- the facility was built with eleven
identical columns precisely so that this failure has something to alias
against.

Also reports the odometry error a HEALTHY robot accumulates over a full
mission, because that is the noise floor the detector works against. If
the knee of the curve sits near that value, that is the explanation, and
it is the sentence Chapter 4 needs.

Usage:
    py sweep_displacement.py
"""

import csv
import os
import sys
import time

import numpy as np

import config
import run_experiments as rx

# The seven seeds the suite injects wrong_position on. Taken from the
# schedule rather than hard-coded, so this cannot drift out of step with
# the suite.
from faults import schedule_for_seed

DISPLACEMENTS_M = [2.0, 4.0, 6.0, 12.0, 24.0]
OUT = "sweep_displacement.csv"


def wrong_position_seeds():
    seeds = []
    for s in config.EXPERIMENT_SEEDS:
        if not rx.deployable(s)[0]:
            continue
        if schedule_for_seed(s, config.EXPERIMENT_FAULT_ROBOT)[2] == "wrong_position":
            seeds.append(s)
    return seeds


def healthy_odometry_drift():
    """
    Mean and worst final odometry error of a healthy robot over a full
    mission, read from the preserved v1 traces.

    NO RE-RUN NEEDED. The traces already store both poses per sample --
    true (x, y) and believed (bx, by) -- so the drift is recoverable from
    a dataset that already exists. C1 is the right condition: deviations,
    no faults, which is a healthy squad in a realistic facility.

    Odometry drift is analysis-only, exactly like `pose_error_m()`. No
    robot can see it, and nothing in the simulation consults it.
    """
    finals, peaks = [], []
    trace_dir = "traces_v1" if os.path.isdir("traces_v1") else config.TRACE_DIR
    for name in sorted(os.listdir(trace_dir)):
        if not name.startswith("C1_"):
            continue
        z = np.load(os.path.join(trace_dir, name))
        err = np.hypot(z["x"] - z["bx"], z["y"] - z["by"])
        for rid in np.unique(z["id"]):
            e = err[z["id"] == rid]
            if len(e):
                finals.append(float(e[-1]))
                peaks.append(float(e.max()))
    return finals, peaks, trace_dir


def main():
    seeds = wrong_position_seeds()
    original = config.FAULT_POSE_OFFSET_M
    print("DISPLACEMENT SENSITIVITY SWEEP -- C3 only, wrong_position seeds")
    print(f"seeds: {seeds}  (n={len(seeds)})")
    print(f"displacements: {DISPLACEMENTS_M} m")
    print(f"config default, unchanged: FAULT_POSE_OFFSET_M = {original}\n")

    # Resumable like the suite, but keyed on (displacement, run_id) rather
    # than run_id alone -- the same run_id appears once per displacement.
    existing = set()
    if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        with open(OUT, newline="") as fh:
            for r in csv.DictReader(fh):
                existing.add((r["displacement_m"], r["run_id"]))

    fresh = not (os.path.exists(OUT) and os.path.getsize(OUT) > 0)
    fh = open(OUT, "a", newline="")
    # extrasaction="ignore": run_one() attaches a private "_wall_s" key.
    writer = csv.DictWriter(fh, fieldnames=["displacement_m"] + rx.COLUMNS,
                            extrasaction="ignore")
    if fresh:
        writer.writeheader()
        fh.flush()

    started = time.time()
    rows = []
    for d in DISPLACEMENTS_M:
        # Override for this process only. The believed pose is displaced
        # along x, matching the rack-bay geometry the fault models.
        config.FAULT_POSE_OFFSET_M = (d, 0.0)
        for seed in seeds:
            key = (f"{d}", f"C3_s{seed}")
            if key in existing:
                continue
            t = time.time()
            row = rx.run_one("C3", seed, save_trace=False)
            row["displacement_m"] = d
            writer.writerow(row)
            fh.flush()
            rows.append(row)
            print(f"  {d:>5.1f} m  C3_s{seed:<5} "
                  f"det {row['fault_detected']}  "
                  f"lat {row['detection_latency_s']:>7}  "
                  f"truly {row['points_truly_visited']:>2}  "
                  f"false {row['points_falsely_reported']:>2}  "
                  f"Qok {row['quarantines_correct']} "
                  f"Qbad {row['quarantines_wrong']}  ({time.time() - t:.0f}s)")
    fh.close()
    config.FAULT_POSE_OFFSET_M = original

    # Re-read everything, so a resumed run reports the whole sweep.
    with open(OUT, newline="") as f:
        allrows = list(csv.DictReader(f))

    print(f"\ncompleted in {(time.time() - started) / 60:.1f} min")
    print("\n" + "=" * 86)
    print("DETECTION AGAINST DISPLACEMENT -- C3, n=%d seeds each" % len(seeds))
    print("%9s %10s %11s %9s %9s %10s %10s" %
          ("offset m", "detected", "mean lat s", "false Q", "correct Q",
           "truly", "falsely"))
    print("-" * 86)
    for d in DISPLACEMENTS_M:
        sub = [r for r in allrows if float(r["displacement_m"]) == d]
        if not sub:
            continue
        n = len(sub)
        det = sum(int(r["fault_detected"]) == 1 for r in sub)
        lats = [float(r["detection_latency_s"]) for r in sub
                if float(r["detection_latency_s"]) >= 0]
        print("%9.1f %6d/%-3d %11s %9.2f %9.2f %10.2f %10.2f" % (
            d, det, n,
            ("%.1f" % np.mean(lats)) if lats else "-",
            sum(int(r["quarantines_wrong"]) for r in sub) / n,
            sum(int(r["quarantines_correct"]) for r in sub) / n,
            sum(float(r["points_truly_visited"]) for r in sub) / n,
            sum(float(r["points_falsely_reported"]) for r in sub) / n))

    # --- the noise floor the curve has to clear ----------------------
    finals, peaks, trace_dir = healthy_odometry_drift()
    print("\nHEALTHY ODOMETRY DRIFT (C1, from %s, %d robot-missions)"
          % (trace_dir, len(finals)))
    print("  mean final pose error : %.2f m" % np.mean(finals))
    print("  median                : %.2f m" % np.median(finals))
    print("  90th percentile       : %.2f m" % np.percentile(finals, 90))
    print("  worst                 : %.2f m" % np.max(finals))
    print("  mean PEAK during a mission : %.2f m" % np.mean(peaks))
    print("  worst peak                 : %.2f m" % np.max(peaks))

    print("\n" + "=" * 86)
    print("READ THE KNEE AGAINST THE DRIFT. A displacement comparable to")
    print("the squad's own positional uncertainty is not separable by map")
    print("comparison, because the detector's whole signal is that one")
    print("robot disagrees with the others more than they disagree with")
    print("each other.")
    print("=" * 86)

    # THE PROHIBITION, CHECKED RATHER THAN PROMISED. The in-process value
    # is restored above; this confirms the FILE on disk still declares 6 m,
    # which is the thing Session 8 forbade changing. Checked by reading the
    # source rather than by reloading the module: a reload re-executes
    # config.py under every other module that already holds a reference to
    # it, which is a lot of risk for an assertion.
    assert config.FAULT_POSE_OFFSET_M == original, "offset left modified"
    source = open("config.py", encoding="utf-8", errors="replace").read()
    assert "FAULT_POSE_OFFSET_M = (6.0, 0.0)" in source, (
        "config.py no longer declares FAULT_POSE_OFFSET_M = (6.0, 0.0) -- "
        "the default must stay at 6 m whatever this sweep shows")
    print("\nconfig.py default verified unchanged on disk: "
          "FAULT_POSE_OFFSET_M = (6.0, 0.0)")


if __name__ == "__main__":
    main()
