"""
check_false_positives.py
========================
The gate: how often does a HEALTHY squad accuse or quarantine one of its
own?

    python check_false_positives.py

Runs condition C0 -- perfect drawings, no deviations, no faults, full
fault tolerance -- across ten seeds. Nothing is broken in any of these
runs, so every accusation is false by construction and every quarantine
is the system damaging itself.

WHY TEN SEEDS AND NOT THREE
---------------------------
Because three was how the Byzantine detector came to be overfitted. Tuned
on seeds 42, 7 and 2024 it scored 3 of 3 detections with zero false
positives; measured more widely it quarantined the wrong robot in 5 runs
of 15 and cost a healthy squad 14 completed inspections. Three seeds is an
anecdote. The thresholds it produced were fitted to noise.

WHAT COUNTS
-----------
  accusations   any robot naming any peer. All false here.
  quarantines   accusations acted on destructively -- rollback plus, for
                wrong_position, invalidating that robot's completions.
                These are what actually cost the mission.
  believed vs truly
                a healthy squad should report what it actually did. A gap
                means the squad threw away good work.

The pass mark is a quarantine rate under 10 % of runs. Above that, nothing
measured downstream means anything, because the dominant effect in every
condition is the fault tolerance attacking healthy robots.
"""

import numpy as np

import config
import demo_squad

SEEDS = (1, 2, 3, 4, 5, 7, 11, 13, 42, 2024)


def main(seeds=SEEDS):
    print(f"\nC0 -- healthy squad, {len(seeds)} seeds. "
          "Every accusation here is false.\n")
    print(f"  {'seed':>6s} {'believed':>9s} {'truly':>7s} {'total':>6s} "
          f"{'accus':>6s} {'quaran':>7s} {'invalid':>8s} {'dur_s':>7s}  who")
    print("-" * 86)

    rows = []
    for seed in seeds:
        out = demo_squad.run(seed, verbose=False, with_deviations=False)
        facility, squad, points, deviations, history, trace, stats = out
        m = demo_squad.squad_metrics(facility, squad, points)

        accusations = sum(len(f) for f in stats["detections"].values())
        quarantines = [a for a in stats["recovery"] if a["rolled_back"]]
        who = sorted({a["suspect"] for a in quarantines})

        row = {"seed": seed, "believed": m["believed"], "truly": m["truly"],
               "total": m["total"], "accusations": accusations,
               "quarantines": len(quarantines),
               "invalidated": m["invalidated"],
               "duration": stats["steps"] * config.DT_S}
        rows.append(row)
        print(f"  {seed:>6d} {row['believed']:>9d} {row['truly']:>7d} "
              f"{row['total']:>6d} {accusations:>6d} {len(quarantines):>7d} "
              f"{row['invalidated']:>8d} {row['duration']:>7.0f}  "
              f"{who if who else ''}")

    n = len(rows)
    with_accusation = sum(1 for r in rows if r["accusations"] > 0)
    with_quarantine = sum(1 for r in rows if r["quarantines"] > 0)
    believed = sum(r["believed"] for r in rows)
    truly = sum(r["truly"] for r in rows)
    total = sum(r["total"] for r in rows)

    print("-" * 86)
    print(f"  runs with a false accusation : {with_accusation}/{n} "
          f"({100 * with_accusation / n:.0f} %)")
    print(f"  runs with a false QUARANTINE : {with_quarantine}/{n} "
          f"({100 * with_quarantine / n:.0f} %)   <-- the gate")
    print(f"  completions thrown away      : {sum(r['invalidated'] for r in rows)}")
    print(f"  believed {believed}/{total}   truly {truly}/{total}   "
          f"gap {truly - believed:+d}")

    rate = 100.0 * with_quarantine / n
    verdict = "PASS" if rate < 10.0 else "FAIL"
    print(f"\n  FALSE-QUARANTINE RATE {rate:.0f} %  -> {verdict} "
          f"(gate is under 10 %)\n")
    return rows, rate


if __name__ == "__main__":
    main()
