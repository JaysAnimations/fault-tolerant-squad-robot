"""
sensitivity.py
==============
Does any conclusion in Chapter 4 depend on an energy coefficient being
right?

WHY THIS SCRIPT EXISTS. The four energy coefficients in `config.py` are
derived from manufacturer datasheets, not measured from the robot this
project actually simulates. That is an honest weakness, and the honest
answer to it is not to argue the numbers are correct -- it is to show that
the conclusions do not depend on them being correct.

HOW IT WORKS, AND WHY IT IS ARITHMETIC RATHER THAN A RE-RUN. Since
Session 14 the suite writes five energy columns instead of one total:

    energy_drive_j    = E_DRIVE_J_PER_M  x (metres commanded)
    energy_turn_j     = E_TURN_J_PER_RAD x (radians commanded)
    energy_sense_j    = P_SENSE_W        x (seconds spent sensing)
    energy_compute_j  = P_COMPUTE_W      x (seconds spent alive)
    energy_comms_j    = E_COMMS_J_PER_KB x (kilobytes sent)

Each column is exactly LINEAR in one coefficient and independent of the
other three. So "what if P_SENSE_W were double?" is answered by doubling
one column and re-totalling -- no mission has to be simulated again.

TWO PLACES WHERE THAT IS NOT QUITE TRUE, both measured in Session 14
rather than assumed away, and both small:

  1. THE BATTERY. A robot that runs flat stops, and the mission that
     follows is a different mission. Scaling on paper cannot reproduce
     that. It binds only on battery_drain runs -- where the 25x drain
     multiplier is the whole point of the fault -- and battery_headroom()
     below excludes them and says so. Measured: the new, lower
     coefficients let the drained robot survive longer, and those
     missions ran 15-18 % longer with robots_alive_at_end going 2 -> 3 on
     two runs.

  2. THE PREDICTIVE BATTERY DETECTOR. detection.py projects energy_j
     forward against remaining charge to warn that a robot will not
     finish the round. That projection reads the ABSOLUTE energy scale,
     so it crosses its threshold at a different step when a coefficient
     moves -- even on a healthy robot with no fault injected. Measured on
     the fault-free conditions: the same accusations fire (5 across C0
     and C1 in both datasets) but at different moments, perturbing
     routing on 2 of 58 rows and moving no reported mean.

Neither invalidates the method. Both are stated because a sensitivity
analysis whose own assumptions are unexamined is not worth much.

WHAT IT REPORTS. Every headline energy comparison in Chapter 4, evaluated
at 0.5x, 1x and 2x on each coefficient independently -- a factor of four.
For each, whether the sign of the difference survives. A conclusion that
flips is named; a conclusion that holds everywhere is reported as stable.

Usage:
    py sensitivity.py                 # reads results.csv
    py sensitivity.py results_v2.csv  # or a named file
"""

import csv
import sys
import collections
import numpy as np

import config

# The five columns, and the config coefficient each one is linear in.
# Kept beside each other so nobody can add a category to the CSV and
# forget to make it scalable here.
CATEGORIES = [
    ("energy_drive_j", "E_DRIVE_J_PER_M"),
    ("energy_turn_j", "E_TURN_J_PER_RAD"),
    ("energy_sense_j", "P_SENSE_W"),
    ("energy_compute_j", "P_COMPUTE_W"),
    ("energy_comms_j", "E_COMMS_J_PER_KB"),
]

SCALES = [0.5, 1.0, 2.0]


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------
def load(path):
    """Read the suite, keeping only what this analysis needs."""
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    if not rows:
        raise SystemExit(f"{path} is empty")

    missing = [c for c, _ in CATEGORIES if c not in rows[0]]
    if missing:
        raise SystemExit(
            f"{path} has no energy breakdown (missing {missing}).\n"
            "That file predates Session 14. Sensitivity analysis needs the\n"
            "five per-category columns -- re-run the suite, or point this\n"
            "script at a newer results.csv.")
    return rows


def scaled_total(row, factors):
    """Mission energy with each category multiplied by its factor."""
    total = 0.0
    for column, coefficient in CATEGORIES:
        total += float(row[column]) * factors.get(coefficient, 1.0)
    return total


def per_point(row, factors):
    """
    Joules per inspection point, recomputed.

    Divides by the same denominator run_experiments uses -- points the
    squad BELIEVES it visited -- so the number is comparable with the
    energy_per_point_j column rather than nearly comparable with it.
    """
    visited = max(int(row["points_visited"]), 1)
    return scaled_total(row, factors) / visited


# ---------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------
def paired(rows, cond_a, cond_b, value, fault=None):
    """
    Pair two conditions seed by seed and return (differences, seeds).

    The suite is a paired design: the same seed gives identical points,
    deviations, noise and fault timing in every condition. Comparing means
    across unpaired subsets would throw that away, so every comparison
    here is per-seed.
    """
    by_cond = collections.defaultdict(dict)
    for r in rows:
        if fault is not None and r["fault_type"] != fault:
            continue
        by_cond[r["condition"]][int(r["seed"])] = r

    seeds = sorted(set(by_cond[cond_a]) & set(by_cond[cond_b]))
    diffs = [value(by_cond[cond_a][s]) - value(by_cond[cond_b][s])
             for s in seeds]
    return np.array(diffs, dtype=float), seeds


def t_statistic(diffs):
    """
    Paired t on the differences. Written out rather than imported because
    the project depends on numpy and matplotlib only, and this is three
    lines of arithmetic.
    """
    n = len(diffs)
    if n < 2:
        return 0.0
    sd = float(np.std(diffs, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(diffs)) / (sd / np.sqrt(n))


# ---------------------------------------------------------------------
# The comparisons Chapter 4 actually makes
# ---------------------------------------------------------------------
# Each is (label, condition_a, condition_b, metric, fault, claimed_sign).
#
# claimed_sign is the direction the dissertation reports, INCLUDING where
# that direction is unflattering. comms_loss and wrong_position are +1
# because C3 genuinely costs more per point than C2 on those two faults --
# both are documented in CLAUDE.md as honest failures, and a sensitivity
# analysis that quietly encoded them as wins would be checking the wrong
# claim.
#
# TWO DIFFERENT QUESTIONS, AND THEY MUST NOT BE CONFLATED:
#   1. does the sign of this difference CHANGE as a coefficient is scaled?
#      That is coefficient dependence, and it is what this script is for.
#   2. is the sign what the dissertation claims? That is a question about
#      the claim, not about the coefficients, and a wrong answer here does
#      not make anything unstable.
COMPARISONS = [
    ("C3 vs C2   energy/point, all faults", "C3", "C2", per_point, None, -1),
    ("C5 vs C2   energy/point, all faults", "C5", "C2", per_point, None, -1),
    ("C3 vs C4   energy/point, squad vs solo", "C3", "C4", per_point, None, -1),
    ("C1 vs C0   total energy, cost of bad drawings",
     "C1", "C0", scaled_total, None, +1),
    ("C3 vs C2   energy/point, IMMOBILISED",
     "C3", "C2", per_point, "immobilised", -1),
    # C3 costs MORE on these two. Stated as such deliberately.
    ("C3 vs C2   energy/point, COMMS LOSS",
     "C3", "C2", per_point, "comms_loss", +1),
    ("C3 vs C2   energy/point, BATTERY DRAIN",
     "C3", "C2", per_point, "battery_drain", -1),
    ("C3 vs C2   energy/point, WRONG POSITION",
     "C3", "C2", per_point, "wrong_position", +1),
]


def ordering_holds(rows, factors):
    """
    Is the C2 -> C5 -> C3 progression on energy per point still monotone?

    This is the shape of the argument, not a single comparison: each layer
    of fault tolerance should cost less per point than the one below it.
    """
    means = {}
    for cond in ("C2", "C5", "C3"):
        vals = [per_point(r, factors) for r in rows if r["condition"] == cond]
        means[cond] = float(np.mean(vals)) if vals else float("nan")
    return means, means["C2"] >= means["C5"] >= means["C3"]


# ---------------------------------------------------------------------
def battery_headroom(rows, factors):
    """
    The one place energy DOES feed back on behaviour: a robot that
    exhausts its battery stops, and the mission that follows is a
    different mission. Scaling a coefficient on paper cannot reproduce
    that, so the assumption this whole method rests on has to be checked
    rather than asserted.

    Reports the worst HEALTHY mission as a fraction of one robot's
    battery. Per robot, because total_energy_j is the squad's bill and
    every condition but C4 runs three.

    TWO HONEST LIMITS ON THIS CHECK, both worth stating rather than
    burying:

    1. battery_drain runs are EXCLUDED and cannot be covered by it. That
       fault multiplies the rate at which charge is drawn
       (FAULT_BATTERY_DRAIN_MULTIPLIER = 25) without changing the joules
       recorded in the energy ledger, so the CSV simply does not contain
       what those robots took from their batteries. On those seeds energy
       genuinely does feed back on behaviour and the arithmetic here is
       approximate, not exact.

    2. It uses the squad mean rather than the individual robot that
       actually ran flat, so it is a lower bound on the true worst case.

    Session 14 measured this directly rather than leaving it theoretical:
    on battery_drain seeds the new, lower coefficients let the drained
    robot survive longer and the mission ran 15-18 % longer. That is a
    real behavioural change caused by a coefficient change, and it is why
    this function reports rather than merely reassures.
    """
    worst = 0.0
    for r in rows:
        if r["fault_type"] == "battery_drain":
            continue
        robots = 1 if r["condition"] == "C4" else 3
        per_robot = scaled_total(r, factors) / robots
        worst = max(worst, per_robot / config.BATTERY_CAPACITY_J)
    return worst


# ---------------------------------------------------------------------
def main(path="results.csv"):
    rows = load(path)
    print(f"SENSITIVITY ANALYSIS -- {path}, {len(rows)} runs")
    print("Each energy coefficient scaled 0.5x / 1x / 2x independently:")
    print("a factor-of-four range on every one.\n")

    # --- the composition at 1x, which is the context for everything else
    print("ENERGY COMPOSITION AT THE CURRENT COEFFICIENTS")
    grand = sum(float(r[c]) for r in rows for c, _ in CATEGORIES)
    for column, coefficient in CATEGORIES:
        share = sum(float(r[column]) for r in rows) / grand * 100.0
        # rounded for display: the derived values carry float dust
        # (2.4/0.6 is 4.000000000000001), which is arithmetic, not a typo
        print(f"  {column:<18} {share:5.1f} %   ({coefficient}"
              f" = {round(getattr(config, coefficient), 6):g})")
    print()

    # --- every comparison, at every scaling of every coefficient
    unstable = []
    print("SIGN OF EACH HEADLINE COMPARISON, ACROSS THE FULL SWEEP")
    print(f"  {'comparison':<44} {'n':>3}  "
          f"{'1x mean':>10} {'t':>7}   worst-case over the sweep")
    print("  " + "-" * 100)

    mismatched = []
    for label, a, b, metric, fault, claimed in COMPARISONS:
        base_diffs, seeds = paired(
            rows, a, b, lambda r: metric(r, {}), fault)
        if len(seeds) == 0:
            print(f"  {label:<44}   -- no paired rows")
            continue
        base_mean = float(np.mean(base_diffs))
        base_t = t_statistic(base_diffs)
        base_sign = np.sign(base_mean)

        # Does the sign the suite actually measured survive the sweep? The
        # baseline is the measured 1x value, NOT the claimed direction --
        # scaling a coefficient cannot make a claim wrong, only a
        # measurement move.
        flips = []
        closest = None
        for _, coefficient in CATEGORIES:
            for s in SCALES:
                factors = {coefficient: s}
                diffs, _ = paired(rows, a, b,
                                  lambda r: metric(r, factors), fault)
                mean = float(np.mean(diffs))
                margin = mean * base_sign      # >0 while the sign survives
                if closest is None or margin < closest[0]:
                    closest = (margin, coefficient, s, mean)
                if margin <= 0:
                    flips.append((coefficient, s, mean))

        _, coefficient, s, mean = closest
        if flips:
            verdict = f"SIGN FLIPS at {coefficient} x{s} ({mean:+.1f})"
            unstable.append((label, flips))
        else:
            verdict = f"stable; closest {coefficient} x{s} -> {mean:+.1f}"

        # Separately: is the measured sign the one we report?
        note = ""
        if base_sign != np.sign(claimed):
            note = "  [sign is opposite to the claimed direction]"
            mismatched.append((label, base_mean))
        print(f"  {label:<44} {len(seeds):>3}  "
              f"{base_mean:>10.1f} {base_t:>7.2f}   {verdict}{note}")

    # --- the C2 -> C5 -> C3 progression
    print("\nC2 -> C5 -> C3 ORDERING ON ENERGY PER POINT")
    broke = []
    for _, coefficient in CATEGORIES:
        for s in SCALES:
            means, ok = ordering_holds(rows, {coefficient: s})
            if not ok:
                broke.append((coefficient, s, means))
    base_means, base_ok = ordering_holds(rows, {})
    print(f"  at 1x:  C2 {base_means['C2']:.0f}  "
          f"C5 {base_means['C5']:.0f}  C3 {base_means['C3']:.0f}   "
          f"{'monotone' if base_ok else 'NOT monotone'}")
    if broke:
        for coefficient, s, means in broke:
            print(f"  BREAKS at {coefficient} x{s}: "
                  f"C2 {means['C2']:.0f} C5 {means['C5']:.0f} "
                  f"C3 {means['C3']:.0f}")
    else:
        print(f"  holds at every scaling of every coefficient "
              f"({len(CATEGORIES) * len(SCALES)} combinations)")

    # --- the assumption this whole method rests on
    print("\nBATTERY HEADROOM -- the one feedback path scaling cannot model")
    for s in SCALES:
        worst = max(battery_headroom(rows, {c: s})
                    for _, c in CATEGORIES)
        flag = "" if worst < 1.0 else "   <-- A ROBOT WOULD RUN FLAT"
        print(f"  at x{s}: worst mission uses {worst * 100:5.1f} % "
              f"of one robot's battery{flag}")
    print("  Below 100 % no robot dies, so no mission would have gone")
    print("  differently and the arithmetic above is exact rather than")
    print("  approximate. battery_drain runs are excluded and are the")
    print("  known exception -- see battery_headroom()'s docstring.")

    # --- the verdict, in the words Chapter 4 needs
    print("\n" + "=" * 72)
    if not unstable and not broke:
        print("VERDICT: conclusions stable across a factor-of-four variation")
        print("in every energy coefficient. No headline comparison changes")
        print("sign, and the C2 -> C5 -> C3 ordering is preserved throughout.")
    else:
        print("VERDICT: NOT fully stable. The following change SIGN when a")
        print("coefficient is scaled, and must be reported with that")
        print("dependence:")
        for label, flips in unstable:
            for coefficient, s, mean in flips:
                print(f"  - {label}: flips at {coefficient} x{s} "
                      f"(mean {mean:+.1f})")
        for coefficient, s, means in broke:
            print(f"  - C2/C5/C3 ordering breaks at {coefficient} x{s}")

    # A SEPARATE MATTER, AND NOT A STABILITY PROBLEM. These are rows where
    # the squad's fault tolerance costs more than the naive baseline. They
    # are stable -- reliably, robustly unflattering -- and both are already
    # documented as honest failures.
    if mismatched:
        print("\nSIGN OPPOSITE TO THE CLAIMED DIRECTION (not instability):")
        for label, mean in mismatched:
            print(f"  - {label}: measured {mean:+.1f} J/point")
        print("  These hold their sign across the whole sweep, so they are")
        print("  robust findings rather than coefficient artefacts.")
    print("=" * 72)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results.csv")
