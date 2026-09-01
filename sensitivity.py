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
one column and re-totalling -- no mission has to be simulated again. The
physical behaviour is unchanged by construction, because none of these
coefficients feeds back into what a robot decides to do. The one place
energy does feed back is the battery: a robot that runs flat stops. That
caveat is stated in the report this script prints rather than hidden,
because at a mission cost of a few per cent of the battery it does not
bind, and the script checks that it does not.

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
# Each is (label, condition_a, condition_b, metric, fault, expected_sign).
# expected_sign is what the dissertation claims: -1 means "a spends less
# than b". A run at some scale factor that produces the opposite sign is a
# conclusion that depends on a coefficient, and this script exists to find
# exactly those.
COMPARISONS = [
    ("C3 vs C2   energy/point, all faults", "C3", "C2", per_point, None, -1),
    ("C5 vs C2   energy/point, all faults", "C5", "C2", per_point, None, -1),
    ("C3 vs C4   energy/point, squad vs solo", "C3", "C4", per_point, None, -1),
    ("C1 vs C0   total energy, cost of bad drawings",
     "C1", "C0", scaled_total, None, +1),
    ("C3 vs C2   energy/point, IMMOBILISED",
     "C3", "C2", per_point, "immobilised", -1),
    ("C3 vs C2   energy/point, COMMS LOSS",
     "C3", "C2", per_point, "comms_loss", -1),
    ("C3 vs C2   energy/point, BATTERY DRAIN",
     "C3", "C2", per_point, "battery_drain", -1),
    ("C3 vs C2   energy/point, WRONG POSITION",
     "C3", "C2", per_point, "wrong_position", -1),
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
    exhausts its battery stops. Scaling a coefficient in analysis cannot
    reproduce that, so the assumption has to be checked rather than
    asserted. Reports the worst mission as a fraction of one robot's
    battery, at this scaling.

    Reported per robot: total_energy_j is the squad's bill, and the
    conditions run three robots except C4.
    """
    worst = 0.0
    for r in rows:
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
        print(f"  {column:<18} {share:5.1f} %   ({coefficient}"
              f" = {getattr(config, coefficient)})")
    print()

    # --- every comparison, at every scaling of every coefficient
    unstable = []
    print("SIGN OF EACH HEADLINE COMPARISON, ACROSS THE FULL SWEEP")
    print(f"  {'comparison':<44} {'n':>3}  "
          f"{'1x mean':>10} {'t':>7}   worst-case over the sweep")
    print("  " + "-" * 100)

    for label, a, b, metric, fault, want in COMPARISONS:
        base_diffs, seeds = paired(
            rows, a, b, lambda r: metric(r, {}), fault)
        if len(seeds) == 0:
            print(f"  {label:<44}   -- no paired rows")
            continue
        base_mean = float(np.mean(base_diffs))
        base_t = t_statistic(base_diffs)

        # Sweep: one coefficient moved at a time, the rest at 1x.
        flips = []
        worst_margin = None
        for _, coefficient in CATEGORIES:
            for s in SCALES:
                factors = {coefficient: s}
                diffs, _ = paired(rows, a, b,
                                  lambda r: metric(r, factors), fault)
                mean = float(np.mean(diffs))
                # margin > 0 means the claimed sign still holds
                margin = mean * want
                if worst_margin is None or margin < worst_margin[0]:
                    worst_margin = (margin, coefficient, s, mean)
                if margin <= 0:
                    flips.append((coefficient, s, mean))

        margin, coefficient, s, mean = worst_margin
        if flips:
            verdict = f"FLIPS at {coefficient} x{s} (mean {mean:+.1f})"
            unstable.append((label, flips))
        else:
            verdict = (f"holds; closest {coefficient} x{s} "
                       f"-> {mean:+.1f}")
        print(f"  {label:<44} {len(seeds):>3}  "
              f"{base_mean:>10.1f} {base_t:>7.2f}   {verdict}")

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
    print("  approximate.")

    # --- the verdict, in the words Chapter 4 needs
    print("\n" + "=" * 72)
    if not unstable and not broke:
        print("VERDICT: conclusions stable across a factor-of-four variation")
        print("in every energy coefficient. No headline comparison changes")
        print("sign, and the C2 -> C5 -> C3 ordering is preserved throughout.")
    else:
        print("VERDICT: NOT fully stable. The following depend on a")
        print("coefficient and must be reported with that dependence:")
        for label, flips in unstable:
            for coefficient, s, mean in flips:
                print(f"  - {label}: flips at {coefficient} x{s} "
                      f"(mean {mean:+.1f})")
        for coefficient, s, means in broke:
            print(f"  - C2/C5/C3 ordering breaks at {coefficient} x{s}")
    print("=" * 72)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results.csv")
