"""
demo_faults.py
==============
STEP 3: break one robot, five different ways, and watch the mission suffer.

Run:  python demo_faults.py

Every run below is the SAME mission on the SAME seed -- same inspection
points, same deviations, same sensor noise, same start positions. The only
thing that changes is which fault is injected into robot 1 at t = 120 s.
That is what makes the comparison a comparison: any difference in the
result is caused by the fault and by nothing else.

WHAT THIS DEMONSTRATES, AND WHAT IT DELIBERATELY DOES NOT
---------------------------------------------------------
It demonstrates that each fault does real damage. It does NOT detect
anything: no robot notices, nobody is quarantined, no work is reallocated.
The squad carries on as though everything were fine, which is exactly the
point -- this run is the "before" picture that Step 4 and Step 5 are
measured against, and it is condition C2 in miniature.

If you want to know whether a fault is working, look for its signature:

  sensor_degradation  the faulty robot's own map gets worse. It reports
                      obstacles that are not there and misses ones that
                      are, so its disagreement with the others climbs.
  wrong_position      its map is fine and in the wrong place. Its
                      disagreement with the others is the largest of any
                      fault, while nothing about the robot looks unwell.
  comms_loss          it stops contributing. Deliveries collapse, its work
                      is never reconciled, and the squad duplicates it.
  immobilised         it stops covering ground. Its points are never
                      reallocated, so the mission ends with them unvisited.
  battery_drain       it dies part-way through, taking its assignment with
                      it.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import demo_squad

# The robot to break and when. Both live in config so that a run can be
# reproduced from the parameter file alone.
VICTIM = config.FAULT_DEMO_ROBOT
WHEN = config.FAULT_DEMO_STEP


def _observed_error(member, facility):
    """
    How wrong this robot's map is, over the cells IT ACTUALLY SAW.

    WHY NOT grid.error_rate(). That scores every decided cell, and with the
    documented layout seeded into the map every one of the 440,000 cells is
    decided and nearly all of them are right. A badly degraded sensor moved
    it from 0.78 % to 0.88 % -- a real effect, drowned in a denominator
    made almost entirely of the prior. Restricting the score to cells this
    robot observed for itself measures what the sensor did, which is the
    thing the fault damaged.

    Analysis only. A robot that could compute this would not need Step 4.
    """
    own = member.grid.own_observations() != 0.0
    cls = member.grid.classified()
    decided = own & (cls >= 0)
    if not decided.any():
        return 0.0
    truth = (facility.grid == 1).astype(np.int8)
    return float((cls[decided] != truth[decided]).sum() / decided.sum()) * 100


def _disagreement_cells(squad):
    """
    Number of cells where two robots CONFIDENTLY contradict each other,
    counted rather than averaged.

    Same dilution problem as above: disagreement_with() divides by every
    decided cell, so a squad that agrees on the prior everywhere reports
    fractions of a percent whatever happens. The absolute count is what
    moves. This is the raw signal Step 4's Byzantine detector will work
    from, reported here and acted on by nothing.
    """
    total = 0
    for i in range(len(squad)):
        for j in range(i + 1, len(squad)):
            a = squad[i].grid.classified()
            b = squad[j].grid.classified()
            both = (a >= 0) & (b >= 0)
            total += int((a[both] != b[both]).sum())
    return total


def measure(label, faults):
    # DETECTION AND RECOVERY BOTH OFF, EXPLICITLY. This file is the Step 3
    # "before" picture and its whole claim is that nobody notices. Once
    # Step 5 landed, RECOVERY_ENABLED defaulted to True and these runs
    # silently started quarantining robots -- the degraded robot's map
    # error jumped from 0.82 % to 26 % because the squad was down-weighting
    # and rolling back underneath a demo that says it does neither.
    # A demo whose behaviour drifts away from its own docstring is worse
    # than no demo.
    out = demo_squad.run(config.DEFAULT_SEED, verbose=False, faults=faults,
                         detect=False, recover=False)
    facility, squad, points, deviations, history, trace, stats = out
    m = demo_squad.squad_metrics(facility, squad, points)
    victim = squad[VICTIM]

    return {
        "label": label,
        "visited": m["visited"],
        "total": m["total"],
        "duration": stats["steps"] * config.DT_S,
        "energy": m["energy"],
        "victim_points": sum(1 for p in points if p.visited_by == VICTIM),
        "victim_distance": victim.robot.distance_travelled_m,
        "victim_battery": victim.robot.battery_fraction * 100,
        "victim_status": victim.robot.status(),
        "victim_map_error": _observed_error(victim, facility),
        "disagreement": _disagreement_cells(squad),
        "deliveries": stats["radio"]["delivered"],
        "detected": sum(1 for d in deviations if d.detected),
        "ndev": len(deviations),
        "squad": squad,
        "facility": facility,
    }


def main():
    runs = [measure("no fault (control)", [])]
    for name in config.FAULT_TYPES:
        runs.append(measure(name, [(VICTIM, WHEN, name)]))

    base = runs[0]

    print("\n" + "=" * 78)
    print(f"  STEP 3 -- FIVE FAULTS, ONE MISSION, NO DETECTION")
    print(f"  seed {config.DEFAULT_SEED}, robot {VICTIM} broken at "
          f"t = {WHEN * config.DT_S:.0f} s")
    print("=" * 78)
    print(f"  {'fault':<20s} {'points':>7s} {'dur_s':>7s} {'energy_J':>9s} "
          f"{'r1 pts':>7s} {'r1 dist':>8s} {'r1 batt':>8s} {'r1 err':>7s} "
          f"{'disagree':>9s} {'msgs':>6s}")
    print("-" * 78)
    for r in runs:
        print(f"  {r['label']:<20s} {r['visited']:>4d}/{r['total']:<2d} "
              f"{r['duration']:>7.0f} {r['energy']:>9.0f} "
              f"{r['victim_points']:>7d} {r['victim_distance']:>8.1f} "
              f"{r['victim_battery']:>7.1f}% {r['victim_map_error']:>6.2f}% "
              f"{r['disagreement']:>9d} {r['deliveries']:>6d}")

    print("-" * 78)
    print("  CHANGE AGAINST THE CONTROL")
    for r in runs[1:]:
        dp = r["visited"] - base["visited"]
        dt = r["duration"] - base["duration"]
        de = r["energy"] - base["energy"]
        dd = r["disagreement"] - base["disagreement"]
        print(f"    {r['label']:<20s} points {dp:+3d}   duration {dt:+7.0f} s   "
              f"energy {de:+8.0f} J   disagreeing cells {dd:+7d}")

    print("-" * 78)
    print("  Robot 1's final status, per run:")
    for r in runs:
        print(f"    {r['label']:<20s} {r['victim_status']}")
    print("-" * 78)
    print("  No robot detected any of this. The squad kept bidding, kept")
    print("  merging the faulty maps, and kept reporting success. Making it")
    print("  notice is Step 4; making it recover is Step 5.")
    print("=" * 78 + "\n")

    _figure(runs)


def _figure(runs):
    """One row per fault: the broken robot's own map, and the damage."""
    n = len(runs)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(16, 8.5))
    axes = axes.ravel()
    facility = runs[0]["facility"]
    ext = [0, facility.width_m, 0, facility.height_m]

    for ax, r in zip(axes, runs):
        grid = r["squad"][VICTIM].grid
        cls = grid.classified().astype(float)
        cls[cls == -1] = 0.5
        ax.imshow(1 - cls, cmap="gray", origin="lower", extent=ext,
                  vmin=0, vmax=1)
        ax.plot(r["squad"][VICTIM].trail_x, r["squad"][VICTIM].trail_y,
                lw=1.0, color="#C1442E", alpha=0.9)
        ax.set_title(f"{r['label']}\nobserved-cell error "
                     f"{r['victim_map_error']:.2f} %, "
                     f"{r['disagreement']:,} cells in dispute", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(runs):]:
        ax.axis("off")

    fig.suptitle(f"Robot {VICTIM}'s own map after each fault "
                 f"(seed {config.DEFAULT_SEED}, broken at "
                 f"t = {WHEN * config.DT_S:.0f} s)", fontsize=11)
    fig.tight_layout()
    fig.savefig("demo_faults_result.png", dpi=120)
    print("Saved demo_faults_result.png")


if __name__ == "__main__":
    main()
