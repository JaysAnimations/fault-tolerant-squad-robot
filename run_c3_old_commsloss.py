"""
run_c3_old_commsloss.py
=======================
The control arm that separates Session 14's two changes.

WHY THIS EXISTS. Session 14 changed two things: the energy coefficients
(Stage 1b) and the comms-loss response (Stage 2). Reported together their
effects are unattributable, so they have to be separated.

The Session 14 brief proposed doing that arithmetically -- recompute
results_v1.csv under the new coefficients by scaling each energy category.
THAT IS NOT COMPUTABLE. results_v1.csv was produced before the five energy
columns existed; it has one total and no breakdown, and the traces store
only total energy_j as well. The per-category split of the v1 runs is
simply not recorded anywhere.

The substitute is exact rather than approximate, and cheaper than a second
full suite. `recovery.consider()` is reached only when the condition sets
recover=True, and C3 is the only condition that does. So:

  * The 145 non-C3 rows are BEHAVIOURALLY IDENTICAL between v1 and v2 --
    same seeds, same code path, deterministic. Any difference between them
    is the coefficient change and nothing else. That comparison is free.

  * The 29 C3 rows are the only ones the Stage 2 change can touch. Re-run
    them here with RECOVERY_COMMS_LOSS_REALLOCATE = True (the old
    behaviour) under the NEW coefficients, and the difference against v2's
    C3 rows is the behaviour change and nothing else.

That gives the three-way separation the brief asked for:

    v1            old behaviour, old coefficients   (results_v1.csv)
    C3-old arm    old behaviour, new coefficients   (this file, C3 only)
    v2            new behaviour, new coefficients   (results.csv)

29 runs rather than a second 174-run suite.

Usage:
    py run_c3_old_commsloss.py          # writes results_c3_old.csv
Resumable in the same way as the main suite: rows already present are
skipped.
"""

import csv
import os
import sys
import time

import config
import run_experiments as rx

OUT = "results_c3_old.csv"


def main():
    # THE WHOLE POINT OF THE ARM: put the comms-loss response back to what
    # it was before Stage 2, while leaving the new coefficients in place.
    config.RECOVERY_COMMS_LOSS_REALLOCATE = True

    # Same exclusion the suite applies, so this arm stays paired with it:
    # seed 14 sites a deviation on a start pose and cannot deploy.
    # deployable() returns (ok, blocked_pose); a bare tuple is always
    # truthy, so the [0] matters.
    seeds = [s for s in config.EXPERIMENT_SEEDS if rx.deployable(s)[0]]
    done = rx.already_done(OUT)
    todo = [s for s in seeds if f"C3_s{s}" not in done]

    print(f"C3 control arm -- old comms-loss behaviour, new coefficients")
    print(f"RECOVERY_COMMS_LOSS_REALLOCATE = "
          f"{config.RECOVERY_COMMS_LOSS_REALLOCATE}")
    print(f"{len(seeds)} seeds, {len(done)} already done, {len(todo)} to go\n")
    if not todo:
        print("Nothing to do.")
        return

    exists = os.path.exists(OUT) and os.path.getsize(OUT) > 0
    fh = open(OUT, "a", newline="")
    # extrasaction="ignore" for the same reason run_experiments.open_writer
    # uses it: run_one() attaches a private "_wall_s" key that is not a CSV
    # column, and DictWriter raises on any extra key by default.
    writer = csv.DictWriter(fh, fieldnames=rx.COLUMNS,
                            extrasaction="ignore")
    if not exists:
        writer.writeheader()
        fh.flush()

    started = time.time()
    for i, seed in enumerate(todo, 1):
        t = time.time()
        # Traces are not saved: this arm is a control for one comparison,
        # not part of the deliverable dataset, and 29 more .npz files in
        # traces/ would be indistinguishable from the real suite's.
        row = rx.run_one("C3", seed, save_trace=False)
        writer.writerow(row)
        fh.flush()
        elapsed = time.time() - started
        rate = elapsed / i
        print(f"  [{i:>2}/{len(todo)}] C3_s{seed:<5} "
              f"{float(row['total_energy_j']):8.1f} J  "
              f"{float(row['energy_per_point_j']):6.1f} J/pt  "
              f"truly {row['points_truly_visited']:>2}  "
              f"realloc {row['points_reallocated']:>2}  "
              f"({time.time() - t:.0f}s, ETA "
              f"{(len(todo) - i) * rate / 60:.0f} min)")

    fh.close()
    print(f"\nDone in {(time.time() - started) / 60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
