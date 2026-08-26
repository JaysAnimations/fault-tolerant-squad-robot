"""
demo_recover.py
===============
STEP 5: does acting on what the squad noticed actually help?

Run:  python demo_recover.py

Every fault, three seeds, run twice: once with recovery enabled and once
without. The pair is identical in every other respect -- same points, same
deviations, same noise, same fault at the same instant -- so any
difference is the recovery mechanism and nothing else. That is condition
C2 against condition C3 in miniature, and it is the comparison the whole
project is built to make.

WHAT IS MEASURED
----------------
  recovery latency   fault injected -> first robot acts on it
  cells restored     map cells rollback() put back, across the squad
  points reallocated points finished after recovery by a robot other than
                     the faulty one
  mission outcome    points visited, and whether the round succeeded

WHAT "RECOVERY" DOES AND DOES NOT MEAN HERE
-------------------------------------------
Only a displaced robot is quarantined and rolled back. A degraded one is
down-weighted, and a robot that is isolated, immobilised or dying keeps
its map entirely -- the squad simply takes over its work. See recovery.py
for why each fault gets the response it does.

Nothing is acted on unless two robots independently agree. A robot with a
dead radio accuses everybody, and on one robot's word it would quarantine
the healthy majority.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import demo_squad

SEEDS = [42, 7, 2024]
VICTIM = config.FAULT_DEMO_ROBOT
WHEN = config.FAULT_DEMO_STEP


# The three arms, and what separates them. C2 is the genuinely naive
# baseline -- a claim is a deed, so a robot that dies holding one takes
# that point with it. C5 keeps only the auction's own timeout recovery.
# C3 adds detection and quarantine on top.
ARMS = {
    "C2": dict(reallocation=False, detect=False, recover=False),
    "C5": dict(reallocation=True, detect=False, recover=False),
    "C3": dict(reallocation=True, detect=True, recover=True),
}


def measure(seed, fault, arm):
    out = demo_squad.run(seed, verbose=False,
                         faults=[(VICTIM, WHEN, fault)], **ARMS[arm])
    facility, squad, points, deviations, history, trace, stats = out
    m = demo_squad.squad_metrics(facility, squad, points)

    actions = stats["recovery"]
    first = min((a["step"] for a in actions), default=None)

    # Who did the squad actually act against? On a good run this is the
    # robot that was broken; it is worth reporting when it is not.
    suspects = sorted({a["suspect"] for a in actions})
    correct = (suspects == [VICTIM]) if suspects else None

    # Work finished after the squad acted, by somebody other than the
    # robot it acted against. This is the reallocation, counted.
    released = {}
    for a in actions:
        for index in a["claims_released"]:
            released.setdefault(index, a["step"])
    by_index = {p.index: p for p in points}
    reallocated = sum(
        1 for index, when in released.items()
        if (by_index.get(index) is not None
            and by_index[index].visited
            and by_index[index].visit_step is not None
            and by_index[index].visit_step > when
            and by_index[index].visited_by not in suspects))

    coverage, f1, obs_err = demo_squad.squad_map_metrics(facility, squad)

    return {
        "seed": seed, "fault": fault, "arm": arm,
        "visited": m["visited"], "total": m["total"],
        "believed": m["believed"], "truly": m["truly"],
        "invalidated": m["invalidated"],
        "success": m["success"],
        "coverage": coverage, "f1": f1, "obs_err": obs_err,
        "duplicated": m["redundancy"] * 100.0,
        "duration": stats["steps"] * config.DT_S,
        "energy": m["energy"],
        "latency_s": ((first - WHEN) * config.DT_S
                      if first is not None else None),
        "cells_restored": sum(a["cells_restored"] for a in actions),
        "reallocated": reallocated,
        "actions": len(actions),
        "suspects": suspects,
        "correct_target": correct,
    }


# Which metric each fault actually damages. Reporting one aggregate across
# all five averages the signal away: only two of them touch coverage at
# all, so a "points visited" table asks four of the five faults a question
# they were never going to answer.
SCORING = {
    "sensor_degradation": [("surface F1", "f1", "hi", "{:.3f}"),
                           ("observed err %", "obs_err", "lo", "{:.2f}")],
    "wrong_position":     [("truly inspected", "truly", "hi", "{:.0f}"),
                           ("believed", "believed", "-", "{:.0f}"),
                           ("observed err %", "obs_err", "lo", "{:.2f}")],
    "comms_loss":         [("duplicated %", "duplicated", "lo", "{:.1f}"),
                           ("coverage %", "coverage", "hi", "{:.2f}")],
    "immobilised":        [("points visited", "visited", "hi", "{:.0f}"),
                           ("missions ok", "success", "hi", "{:.0f}")],
    "battery_drain":      [("points visited", "visited", "hi", "{:.0f}"),
                           ("missions ok", "success", "hi", "{:.0f}")],
}


def _per_fault_scoring(rows):
    """Each fault reported on the metric that fault damages."""
    print("-" * 96)
    print("  EACH FAULT ON THE METRIC IT DAMAGES")
    print("  (hi = higher is better, lo = lower is better;")
    print("   'verdict' asks only whether C2 is worse than C3, as it should be)")
    print("-" * 96)
    print(f"  {'fault':<20s} {'metric':<17s} {'dir':<4s} "
          f"{'C2':>9s} {'C5':>9s} {'C3':>9s}  verdict")
    for fault, metrics in SCORING.items():
        for i, (label, key, direction, fmt) in enumerate(metrics):
            vals = {}
            for arm in ARMS:
                mine = [r for r in rows
                        if r["fault"] == fault and r["arm"] == arm]
                vals[arm] = float(np.mean([float(r[key]) for r in mine])) \
                    if mine else float("nan")
            if direction == "hi":
                ok = vals["C3"] > vals["C2"]
            elif direction == "lo":
                ok = vals["C3"] < vals["C2"]
            else:
                ok = None
            verdict = "-" if ok is None else ("as expected" if ok
                                              else "NOT as expected")
            name = fault if i == 0 else ""
            print(f"  {name:<20s} {label:<17s} {direction:<4s} "
                  + "".join(f"{fmt.format(vals[a]):>9s}" for a in ARMS)
                  + f"  {verdict}")


def main():
    rows = []
    for seed in SEEDS:
        for fault in config.FAULT_TYPES:
            for arm in ARMS:
                row = measure(seed, fault, arm)
                rows.append(row)
                print(f"  seed {seed:<5d} {fault:<20s} {arm}  "
                      f"{row['visited']:>2d}/{row['total']:<3d} "
                      f"{row['duration']:>6.0f}s  "
                      f"acted {row['actions']}", flush=True)

    print("\n" + "=" * 96)
    print("  STEP 5 -- RECOVERY, re-measured with a genuinely naive C2")
    print("=" * 96)
    print("  C2 = claims permanent, nothing reallocated")
    print("  C5 = claims lapse and are re-auctioned, nobody detects anything")
    print("  C3 = detection and quarantine on top")
    print("-" * 96)
    print(f"  {'seed':<6s} {'fault':<20s} {'C2':>7s} {'C5':>7s} {'C3':>7s} "
          f"{'latency':>9s} {'cells back':>11s} {'realloc':>8s} "
          f"{'target':>8s}")
    print("-" * 96)
    for seed in SEEDS:
        for fault in config.FAULT_TYPES:
            got = {a: next(r for r in rows if r["seed"] == seed
                           and r["fault"] == fault and r["arm"] == a)
                   for a in ARMS}
            c3 = got["C3"]
            lat = (f"{c3['latency_s']:.0f} s"
                   if c3["latency_s"] is not None else "-")
            tgt = ("-" if c3["correct_target"] is None
                   else ("right" if c3["correct_target"] else "WRONG"))
            print(f"  {seed:<6d} {fault:<20s} "
                  + "".join(f"{got[a]['visited']:>3d}/{got[a]['total']:<3d}"
                            for a in ARMS)
                  + f" {lat:>9s} {c3['cells_restored']:>11,d} "
                  f"{c3['reallocated']:>8d} {tgt:>8s}")

    _per_fault_scoring(rows)

    print("-" * 96)
    for arm in ARMS:
        mine = [r for r in rows if r["arm"] == arm]
        print(f"  OVERALL {arm}   believed "
              f"{sum(r['believed'] for r in mine)}/"
              f"{sum(r['total'] for r in mine)}   truly "
              f"{sum(r['truly'] for r in mine)}/"
              f"{sum(r['total'] for r in mine)}   "
              f"success {sum(1 for r in mine if r['success'])}/{len(mine)}   "
              f"dur {np.mean([r['duration'] for r in mine]):.0f} s   "
              f"energy {np.mean([r['energy'] for r in mine]):.0f} J   "
              f"re-inspected {sum(r['invalidated'] for r in mine)}")
    print("  're-inspected' is work thrown away by quarantine and done "
          "again --")
    print("  the price of not being able to tell a faulty robot's good "
          "work from its bad.")
    wrong = [r for r in rows if r["arm"] == "C3"
             and r["correct_target"] is False]
    if wrong:
        print(f"  WRONG ROBOT quarantined in {len(wrong)} run(s): "
              + ", ".join(f"seed {r['seed']}/{r['fault']} -> {r['suspects']}"
                          for r in wrong))
    print("=" * 96 + "\n")

    _figure(rows)
    return rows


def _figure(rows):
    faults = list(config.FAULT_TYPES)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    width = 0.27
    idx = np.arange(len(faults))
    colours = {"C2": "#C1442E", "C5": "#BA7517", "C3": "#1D9E75"}
    for k, arm in enumerate(ARMS):
        vals = [sum(r["visited"] for r in rows
                    if r["fault"] == f and r["arm"] == arm) for f in faults]
        ax.bar(idx + k * width, vals, width, label=arm, color=colours[arm])
    ax.set_xticks(idx + width)
    ax.set_xticklabels([f.replace("_", "\n") for f in faults], fontsize=8)
    ax.set_ylabel("inspection points visited (3 seeds)")
    ax.set_title("Mission outcome: naive, timeout-only, full")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    restored = [np.mean([r["cells_restored"] for r in rows
                         if r["fault"] == f and r["arm"] == "C3"])
                for f in faults]
    ax.bar(idx, restored, color="#1D4E89")
    ax.set_xticks(idx)
    ax.set_xticklabels([f.replace("_", "\n") for f in faults], fontsize=8)
    ax.set_ylabel("map cells restored by rollback")
    ax.set_title("Only a displaced robot's map is erased\n"
                 "the other four faults leave it trustworthy")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig("demo_recover_result.png", dpi=125)
    print("Saved demo_recover_result.png")


if __name__ == "__main__":
    main()
