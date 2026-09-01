"""
run_two_faults.py
=================
Stage 4d -- what happens when two of the three robots are faulty at once.

THIS IS A LIMIT FINDING, NOT A BUG, and it should be reported as one. The
architecture rests on majority voting: a robot is quarantined when two
others agree about it. With three robots and two of them faulty, one
healthy robot cannot outvote two, so the mechanism is defeated BY
DEFINITION rather than by any defect in the implementation. Chapter 5 gets
to say: *the architecture tolerates one concurrent fault in three; two
defeats the majority vote, which is a property of the voting scheme rather
than of this implementation.*

Measuring it is worth doing anyway, because "defeated by definition" has
several possible shapes and they are not equally bad:
  * nothing is detected and the mission degrades quietly, or
  * the two faulty robots corroborate each other against the healthy one,
    which is far worse -- the squad would quarantine the only robot still
    telling the truth.
Session 13's M2 fix (an accuser under accusation does not count toward a
quorum) was built for a related case, so which of these happens is an
empirical question.

DESIGN. Robot 1 takes the seed's own scheduled fault, exactly as in the
suite, so this arm stays comparable with it. Robot 2 takes a second,
DIFFERENT fault drawn deterministically from the seed. Robot 0 is left
healthy and is therefore the operator-map reporter.

C3 only: the naive arms have no voting to defeat.

Usage:
    py run_two_faults.py
"""

import csv
import os
import time

import numpy as np

import config
import demo_squad
import run_experiments as rx
from faults import schedule_for_seed

N_SEEDS = 10
SECOND_ROBOT = 2
OUT = "results_two_faults.csv"


def second_fault(seed, first_name):
    """
    A second fault for robot 2, drawn from the seed so the arm is
    deterministic and reproducible like everything else in the project.

    Deliberately a DIFFERENT fault from robot 1's: two robots with the
    same fault at the same moment is a narrower case, and the interesting
    question is whether two differently-broken robots can still defeat the
    vote.

    Its own RNG stream, so drawing it cannot disturb the existing fault
    schedule -- seeds must keep the fault type and timing the suite gave
    them or this arm is not comparable with the suite.
    """
    rng = np.random.default_rng([seed, config.RNG_STREAM_FAULTS, 1729])
    others = [f for f in config.FAULT_TYPES if f != first_name]
    name = str(rng.choice(others))
    fraction = float(rng.uniform(config.FAULT_TIMING_MIN_FRACTION,
                                 config.FAULT_TIMING_MAX_FRACTION))
    step = int(fraction * config.EXPECTED_MISSION_STEPS)
    return SECOND_ROBOT, step, name


def main():
    seeds = [s for s in config.EXPERIMENT_SEEDS
             if rx.deployable(s)[0]][:N_SEEDS]

    print("TWO CONCURRENT FAULTS -- C3 only, %d seeds" % len(seeds))
    print("robot 1: the seed's own fault (as in the suite)")
    print("robot 2: a second, different fault drawn from the seed")
    print("robot 0: healthy\n")

    fresh = not (os.path.exists(OUT) and os.path.getsize(OUT) > 0)
    done = rx.already_done(OUT)
    fh = open(OUT, "a", newline="")
    # extrasaction="ignore": build_row() attaches a private "_wall_s" key.
    writer = csv.DictWriter(
        fh, fieldnames=rx.COLUMNS + ["fault2_robot", "fault2_step",
                                     "fault2_type"],
        extrasaction="ignore")
    if fresh:
        writer.writeheader()
        fh.flush()

    rows = []
    started = time.time()
    for seed in seeds:
        if f"C3_s{seed}" in done:
            continue
        r1, step1, name1 = schedule_for_seed(seed, config.EXPERIMENT_FAULT_ROBOT)
        f2 = second_fault(seed, name1)

        t = time.time()
        out = demo_squad.run(seed, verbose=False, with_deviations=True,
                             reallocation=True, detect=True, recover=True,
                             squad_size=3,
                             faults=[(r1, step1, name1), f2])
        row = rx.build_row("C3", seed, out, time.time() - t)
        row["fault2_robot"], row["fault2_step"], row["fault2_type"] = f2
        writer.writerow(row)
        fh.flush()
        rows.append(row)
        print("  C3_s%-5d r1=%-19s r2=%-19s truly %2s  succ %s  "
              "Qok %s Qbad %s  (%.0fs)"
              % (seed, name1, f2[2], row["points_truly_visited"],
                 row["mission_success"], row["quarantines_correct"],
                 row["quarantines_wrong"], time.time() - t))
    fh.close()

    if not rows:
        print("Nothing to do.")
        return

    print(f"\ncompleted in {(time.time() - started) / 60:.1f} min")
    print("\n" + "=" * 76)
    print("TWO FAULTS AT ONCE -- means over %d seeds" % len(rows))
    print("  truly inspected      : %.2f / %.2f" % (
        np.mean([float(r["points_truly_visited"]) for r in rows]),
        np.mean([float(r["points_total"]) for r in rows])))
    print("  falsely reported     : %.2f" %
          np.mean([float(r["points_falsely_reported"]) for r in rows]))
    print("  missions succeeded   : %d / %d" % (
        sum(int(r["mission_success"]) for r in rows), len(rows)))
    print("  detected (robot 1)   : %d / %d" % (
        sum(int(r["fault_detected"]) == 1 for r in rows), len(rows)))
    print("  correct quarantines  : %d" %
          sum(int(r["quarantines_correct"]) for r in rows))
    print("  WRONG quarantines    : %d" %
          sum(int(r["quarantines_wrong"]) for r in rows))
    print("  energy per point     : %.1f J" %
          np.mean([float(r["energy_per_point_j"]) for r in rows]))
    print("\nCompare against the single-fault C3 rows in results.csv.")
    print("A rise in WRONG quarantines is the dangerous outcome: it would")
    print("mean two faulty robots corroborating each other against the one")
    print("healthy robot still telling the truth.")
    print("=" * 76)


if __name__ == "__main__":
    main()
