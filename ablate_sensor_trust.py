"""
ablate_sensor_trust.py
======================
Stage 4c -- does down-weighting a degraded robot's map buy anything?

THE CASE FOR ASKING. The suite measured surface F1 of 0.696 for C3 against
0.693 for C2 on sensor_degradation -- a difference of 0.003 on four seeds.
That mechanism is also what caused the M1 defect, the largest single bug of
Session 12: because the ledger stored a source's weighted TOTAL and
replaced it on every merge, the first packet after a trust drop
retroactively stripped three quarters of everything that robot had ever
contributed, including all its correct mapping from before the sensor
degraded. Session 13 fixed the arithmetic. It did not re-ask whether the
mechanism earns its place.

THE ABLATION. Run the sensor_degradation seeds with RECOVERY_SENSOR_TRUST
at its current 0.25 and at 1.0. At 1.0 the down-weight is a no-op: the
robot is still detected and still reported, its data is simply counted at
full value. Everything else is identical.

RECOMMENDS, DOES NOT CHANGE. The default is left alone whatever this
shows; a change to it is a decision to be recorded, not a side effect of
measuring. Asserts the config default at the end.

Usage:
    py ablate_sensor_trust.py
"""

import csv
import os
import time

import numpy as np

import config
import run_experiments as rx
from faults import schedule_for_seed

TRUSTS = [0.25, 1.0]       # 0.25 is the current default; 1.0 is the ablation
OUT = "ablate_sensor_trust.csv"


def sensor_seeds():
    return [s for s in config.EXPERIMENT_SEEDS
            if rx.deployable(s)[0]
            and schedule_for_seed(s, config.EXPERIMENT_FAULT_ROBOT)[2]
            == "sensor_degradation"]


def main():
    original = config.RECOVERY_SENSOR_TRUST
    seeds = sensor_seeds()
    print("SENSOR TRUST ABLATION -- C3, sensor_degradation seeds")
    print(f"  seeds  : {seeds}  (n={len(seeds)})")
    print(f"  trusts : {TRUSTS}   (config default {original})\n")

    fresh = not (os.path.exists(OUT) and os.path.getsize(OUT) > 0)
    fh = open(OUT, "a", newline="")
    # extrasaction="ignore": run_one() attaches a private "_wall_s" key.
    writer = csv.DictWriter(fh, fieldnames=["sensor_trust"] + rx.COLUMNS,
                            extrasaction="ignore")
    if fresh:
        writer.writeheader()
        fh.flush()

    rows = []
    started = time.time()
    for trust in TRUSTS:
        config.RECOVERY_SENSOR_TRUST = trust
        for seed in seeds:
            t = time.time()
            row = rx.run_one("C3", seed, save_trace=False)
            row["sensor_trust"] = trust
            writer.writerow(row)
            fh.flush()
            rows.append(row)
            print(f"  trust {trust:<5} C3_s{seed:<5} "
                  f"F1 {float(row['surface_f1']):.4f}  "
                  f"cov {float(row['coverage_pct']):6.2f}  "
                  f"err {float(row['observed_error_pct']):6.3f}  "
                  f"truly {row['points_truly_visited']:>2} "
                  f"({time.time() - t:.0f}s)")
    fh.close()
    config.RECOVERY_SENSOR_TRUST = original

    print(f"\ncompleted in {(time.time() - started) / 60:.1f} min")
    print("\n" + "=" * 74)
    print("MEANS OVER %d sensor_degradation SEEDS" % len(seeds))
    print("%14s %9s %9s %11s %9s" %
          ("trust", "surface F1", "cov %", "obs err %", "truly"))
    print("-" * 74)
    for trust in TRUSTS:
        sub = [r for r in rows if r["sensor_trust"] == trust]
        if not sub:
            continue
        label = f"{trust}" + (" (default)" if trust == original
                              else " (ablated)")
        print("%14s %9.4f %9.2f %11.3f %9.2f" % (
            label,
            np.mean([float(r["surface_f1"]) for r in sub]),
            np.mean([float(r["coverage_pct"]) for r in sub]),
            np.mean([float(r["observed_error_pct"]) for r in sub]),
            np.mean([float(r["points_truly_visited"]) for r in sub])))

    # Paired per seed, since the design is paired.
    print("\nPAIRED PER SEED (0.25 minus 1.0)")
    for seed in seeds:
        a = next(r for r in rows if r["sensor_trust"] == 0.25
                 and int(r["seed"]) == seed)
        b = next(r for r in rows if r["sensor_trust"] == 1.0
                 and int(r["seed"]) == seed)
        print("  seed %-5d dF1 %+.4f  dcov %+6.2f  derr %+7.3f" % (
            seed,
            float(a["surface_f1"]) - float(b["surface_f1"]),
            float(a["coverage_pct"]) - float(b["coverage_pct"]),
            float(a["observed_error_pct"]) - float(b["observed_error_pct"])))
    print("=" * 74)

    import importlib
    importlib.reload(config)
    assert config.RECOVERY_SENSOR_TRUST == original, "config default changed"
    print(f"config.py default verified unchanged: "
          f"RECOVERY_SENSOR_TRUST = {config.RECOVERY_SENSOR_TRUST}")


if __name__ == "__main__":
    main()
