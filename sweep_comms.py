"""
sweep_comms.py
==============
Deriving the radio range instead of choosing it.

Run:  python sweep_comms.py

WHY THIS SWEEP EXISTS
---------------------
Step 4 measured that at COMMS_RANGE_M = 25 m, healthy robots spend most of
a mission unable to hear each other. Step 5 quarantines a robot only when
TWO robots corroborate an accusation -- and two robots that are rarely in
contact cannot corroborate anything. So the radio range is not a detail of
the comms model: it decides whether the fault-tolerance mechanism this
project is graded on can function at all.

Raising it is not engineering the problem away. It is making the thing
under test testable, and the honest way to pick the value is to measure
what each range buys and say so.

WHAT IS MEASURED, PER RANGE
---------------------------
  contact       fraction of the mission a robot can hear at least one peer
  detection     per fault, across three seeds
  false alarms  healthy robots wrongly accused
  duration      how long the round takes
  energy        what the squad spends

The contact fraction is the one that matters for Step 5; the rest are the
cost of buying it. A longer radio is not free -- in a real plant it means
more power, a bigger antenna, or a repeater.

This run takes a while. It is 4 ranges x 3 seeds x 6 conditions, and it is
a Chapter 4 figure rather than something to run casually.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import demo_squad
from demo_detect import score_run, VICTIM, WHEN

SEEDS = [42, 7, 2024]


def run_one(seed, fault, range_m):
    """One mission at a given radio range."""
    faults = [] if fault is None else [(VICTIM, WHEN, fault)]
    out = demo_squad.run(seed, verbose=False, faults=faults)
    row = score_run(out, fault)
    row["seed"] = seed
    row["range_m"] = range_m
    return row


def sweep(verbose=True):
    rows = []
    original = config.COMMS_RANGE_M
    try:
        for range_m in config.COMMS_RANGE_SWEEP_M:
            # Setting the config value IS the experiment. Radio and the
            # detectors both read it at call time, so this is the whole
            # change -- no logic is touched to move between conditions.
            config.COMMS_RANGE_M = range_m
            for seed in SEEDS:
                for fault in [None] + list(config.FAULT_TYPES):
                    row = run_one(seed, fault, range_m)
                    rows.append(row)
                    if verbose:
                        det = ("-" if fault is None
                               else ("YES" if row["detected"] else "no"))
                        print(f"  {range_m:4.0f} m  seed {seed:<5d} "
                              f"{row['fault']:<20s} detected {det:>3s}  "
                              f"contact {row['contact']*100:5.1f} %  "
                              f"fp {len(row['false_positives'])}")
    finally:
        config.COMMS_RANGE_M = original
    return rows


def summarise(rows):
    """Collapse to one line per range."""
    out = []
    for range_m in config.COMMS_RANGE_SWEEP_M:
        mine = [r for r in rows if r["range_m"] == range_m]
        controls = [r for r in mine if r["fault"].startswith("none")]
        faulty = [r for r in mine if not r["fault"].startswith("none")]

        per_fault = {}
        for fault in config.FAULT_TYPES:
            hits = [r for r in faulty if r["fault"] == fault and r["detected"]]
            per_fault[fault] = len(hits)

        out.append({
            "range_m": range_m,
            "contact": float(np.mean([r["contact"] for r in controls])),
            "detected": sum(per_fault.values()),
            "possible": len(config.FAULT_TYPES) * len(SEEDS),
            "per_fault": per_fault,
            "control_fp": sum(len(r["false_positives"]) for r in controls),
            "total_fp": sum(len(r["false_positives"]) for r in mine),
            "duration": float(np.mean([r["steps"] * config.DT_S
                                       for r in controls])),
            "energy": float(np.mean([r["energy"] for r in controls])),
            "visited": float(np.mean([r["visited"] for r in controls])),
        })
    return out


def main():
    print("\nSweeping radio range. 4 ranges x 3 seeds x 6 conditions.\n")
    rows = sweep()
    table = summarise(rows)

    print("\n" + "=" * 92)
    print("  COMMS RANGE SWEEP -- what each radio range buys")
    print("=" * 92)
    print(f"  {'range':>7s} {'in contact':>11s} {'detected':>10s} "
          f"{'control FP':>11s} {'all FP':>7s} {'duration':>10s} "
          f"{'energy':>10s} {'points':>8s}")
    print("-" * 92)
    for t in table:
        print(f"  {t['range_m']:5.0f} m {t['contact']*100:10.1f} % "
              f"{t['detected']:>4d}/{t['possible']:<5d} "
              f"{t['control_fp']:>11d} {t['total_fp']:>7d} "
              f"{t['duration']:>9.0f}s {t['energy']:>10.0f} "
              f"{t['visited']:>8.1f}")

    print("-" * 92)
    print("  DETECTION BY FAULT (out of 3 seeds)")
    header = "  " + " " * 22 + "".join(f"{t['range_m']:>8.0f} m"
                                       for t in table)
    print(header)
    for fault in config.FAULT_TYPES:
        line = f"  {fault:<22s}"
        for t in table:
            line += f"{t['per_fault'][fault]:>7d}/3"
        print(line)
    print("=" * 92 + "\n")

    _figure(table)
    return rows, table


def _figure(table):
    ranges = [t["range_m"] for t in table]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.plot(ranges, [t["contact"] * 100 for t in table], "o-", lw=2,
            color="#1D4E89")
    ax.set_xlabel("radio range (m)")
    ax.set_ylabel("mission spent in contact (%)")
    ax.set_title("Can two robots corroborate?\n"
                 "Quarantine needs two accusers")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ranges, [100 * t["detected"] / t["possible"] for t in table],
            "o-", lw=2, color="#1D9E75", label="detection rate")
    ax.plot(ranges, [t["total_fp"] for t in table], "s--", lw=2,
            color="#C1442E", label="false accusations")
    ax.set_xlabel("radio range (m)")
    ax.set_ylabel("% detected  /  count")
    ax.set_title("Detection and false alarms")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(ranges, [t["duration"] for t in table], "o-", lw=2,
            color="#BA7517", label="duration (s)")
    ax2 = ax.twinx()
    ax2.plot(ranges, [t["energy"] for t in table], "s--", lw=2,
             color="#7A5EA8", label="energy (J)")
    ax.set_xlabel("radio range (m)")
    ax.set_ylabel("mission duration (s)")
    ax2.set_ylabel("squad energy (J)")
    ax.set_title("Cost of the round\n(healthy runs)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("sweep_comms_result.png", dpi=125)
    print("Saved sweep_comms_result.png")


if __name__ == "__main__":
    main()
