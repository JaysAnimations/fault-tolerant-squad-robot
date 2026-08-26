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


def measure(seed, fault, recover):
    out = demo_squad.run(seed, verbose=False, recover=recover,
                         faults=[(VICTIM, WHEN, fault)])
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
    reallocated = 0
    if first is not None:
        reallocated = sum(1 for p in points
                          if p.visited and p.visit_step is not None
                          and p.visit_step > first
                          and p.visited_by not in suspects)

    return {
        "seed": seed, "fault": fault, "recover": recover,
        "visited": m["visited"], "total": m["total"],
        "success": m["success"],
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


def main():
    rows = []
    for seed in SEEDS:
        for fault in config.FAULT_TYPES:
            for recover in (False, True):
                row = measure(seed, fault, recover)
                rows.append(row)
                print(f"  seed {seed:<5d} {fault:<20s} "
                      f"recovery {'on ' if recover else 'off'}  "
                      f"{row['visited']:>2d}/{row['total']:<3d} "
                      f"{row['duration']:>6.0f}s  "
                      f"acted {row['actions']}")

    print("\n" + "=" * 96)
    print("  STEP 5 -- RECOVERY. Same fault, same seed, acted on or not.")
    print("=" * 96)
    print(f"  {'seed':<6s} {'fault':<20s} {'points off':>11s} "
          f"{'points on':>10s} {'latency':>9s} {'cells back':>11s} "
          f"{'realloc':>8s} {'target':>8s}")
    print("-" * 96)
    for seed in SEEDS:
        for fault in config.FAULT_TYPES:
            off = next(r for r in rows if r["seed"] == seed
                       and r["fault"] == fault and not r["recover"])
            on = next(r for r in rows if r["seed"] == seed
                      and r["fault"] == fault and r["recover"])
            lat = (f"{on['latency_s']:.0f} s"
                   if on["latency_s"] is not None else "-")
            tgt = ("-" if on["correct_target"] is None
                   else ("right" if on["correct_target"] else "WRONG"))
            print(f"  {seed:<6d} {fault:<20s} "
                  f"{off['visited']:>5d}/{off['total']:<5d} "
                  f"{on['visited']:>4d}/{on['total']:<5d} "
                  f"{lat:>9s} {on['cells_restored']:>11,d} "
                  f"{on['reallocated']:>8d} {tgt:>8s}")

    print("-" * 96)
    print("  BY FAULT (totals across three seeds)")
    print(f"  {'fault':<20s} {'points off':>11s} {'points on':>10s} "
          f"{'success off':>12s} {'success on':>11s} {'mean latency':>13s}")
    for fault in config.FAULT_TYPES:
        off = [r for r in rows if r["fault"] == fault and not r["recover"]]
        on = [r for r in rows if r["fault"] == fault and r["recover"]]
        lats = [r["latency_s"] for r in on if r["latency_s"] is not None]
        print(f"  {fault:<20s} "
              f"{sum(r['visited'] for r in off):>5d}/"
              f"{sum(r['total'] for r in off):<5d} "
              f"{sum(r['visited'] for r in on):>4d}/"
              f"{sum(r['total'] for r in on):<5d} "
              f"{sum(1 for r in off if r['success']):>8d}/3 "
              f"{sum(1 for r in on if r['success']):>8d}/3 "
              f"{(f'{np.mean(lats):.0f} s' if lats else '-'):>13s}")

    off_all = [r for r in rows if not r["recover"]]
    on_all = [r for r in rows if r["recover"]]
    print("-" * 96)
    print(f"  OVERALL   points  {sum(r['visited'] for r in off_all)}/"
          f"{sum(r['total'] for r in off_all)} without recovery"
          f"   ->   {sum(r['visited'] for r in on_all)}/"
          f"{sum(r['total'] for r in on_all)} with")
    print(f"            missions succeeded  "
          f"{sum(1 for r in off_all if r['success'])}/{len(off_all)}"
          f"   ->   {sum(1 for r in on_all if r['success'])}/{len(on_all)}")
    wrong = [r for r in on_all if r["correct_target"] is False]
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
    width = 0.38
    off = [sum(r["visited"] for r in rows
               if r["fault"] == f and not r["recover"]) for f in faults]
    on = [sum(r["visited"] for r in rows
              if r["fault"] == f and r["recover"]) for f in faults]
    idx = np.arange(len(faults))
    ax.bar(idx, off, width, label="recovery off (C2)", color="#C1442E")
    ax.bar(idx + width, on, width, label="recovery on (C3)", color="#1D9E75")
    ax.set_xticks(idx + width / 2)
    ax.set_xticklabels([f.replace("_", "\n") for f in faults], fontsize=8)
    ax.set_ylabel("inspection points visited (3 seeds)")
    ax.set_title("Mission outcome, with and without recovery")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    restored = [np.mean([r["cells_restored"] for r in rows
                         if r["fault"] == f and r["recover"]])
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
