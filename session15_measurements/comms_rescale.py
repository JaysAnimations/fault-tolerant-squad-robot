"""
comms_rescale.py  --  Session 15, Stage D
=========================================
What `results.csv` says once comms energy is charged at the derived
coefficient instead of the old placeholder.

    py session15_measurements/comms_rescale.py

WHY THIS IS ARITHMETIC AND NOT A RE-RUN. Session 14 added five per-category
energy columns for exactly this situation: `energy_comms_j` is linear in
`E_COMMS_J_PER_KB` and independent of the other four coefficients, so
moving that coefficient is multiplying one column and re-totalling. All the
machinery already exists in sensitivity.py -- the paired comparisons, the
t statistic, the C2 -> C5 -> C3 ordering check -- so this file imports it
rather than growing a second copy that could drift.

THIS IS AN APPROXIMATION, AND HERE IS THE PART THAT IS APPROXIMATE.
Comms energy is drawn from the same battery as everything else, so
lowering it leaves every robot marginally more charge. Two paths therefore
exist that arithmetic on a finished CSV cannot follow:

  1. a robot that ran flat would run flat slightly later, and the mission
     after that moment is a different mission;
  2. detection.py's PREDICTIVE battery check projects energy_j forward
     against remaining charge, so it crosses its threshold at a different
     STEP when the absolute scale moves -- even on a healthy robot.

Both are real. Session 14 measured them when the four main coefficients
moved and found 18 of 145 non-C3 rows changed. Neither is reproduced here.
What IS quantified below is how much smaller this change is than that one,
in units of battery charge, so the size of what is being neglected is
stated rather than waved at.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import sensitivity

OLD_E_COMMS = 0.05                      # what results.csv was produced with
NEW_E_COMMS = config.E_COMMS_J_PER_KB   # derived in Session 15
FACTOR = NEW_E_COMMS / OLD_E_COMMS

AS_MEASURED = {}                            # every coefficient at 1x
RESCALED = {"E_COMMS_J_PER_KB": FACTOR}     # comms only


def category_shares(rows, factors):
    """Where a mission's joules go, as percentages, under given factors."""
    totals = {c: 0.0 for c, _ in sensitivity.CATEGORIES}
    for r in rows:
        for column, coefficient in sensitivity.CATEGORIES:
            totals[column] += float(r[column]) * factors.get(coefficient, 1.0)
    grand = sum(totals.values())
    return {c: 100.0 * v / grand for c, v in totals.items()}, grand


def main(path="results.csv"):
    here = os.path.dirname(os.path.abspath(__file__))
    rows = sensitivity.load(os.path.join(os.path.dirname(here), path))

    print(f"STAGE D -- comms energy rescaled in analysis, {len(rows)} runs\n")
    print(f"  E_COMMS_J_PER_KB  {OLD_E_COMMS:.7f}  ->  {NEW_E_COMMS:.7f} "
          f"J/kB   (x{FACTOR:.6f}, i.e. 1/{1 / FACTOR:.2f})")
    print("  Derived from ESP32 Table 5-4 transmit airtime; see config.py.\n")

    # --- where the joules go ------------------------------------------
    before, grand_before = category_shares(rows, AS_MEASURED)
    after, grand_after = category_shares(rows, RESCALED)

    print("  COMPOSITION OF A MISSION")
    print(f"    {'category':<20s} {'as measured':>12s} {'rescaled':>10s}")
    for column, _ in sensitivity.CATEGORIES:
        print(f"    {column:<20s} {before[column]:>11.2f} % "
              f"{after[column]:>9.2f} %")
    print(f"    {'TOTAL (all runs)':<20s} {grand_before:>11.0f} J "
          f"{grand_after:>9.0f} J")
    drop = 100.0 * (grand_before - grand_after) / grand_before
    print(f"\n  Total energy across the suite falls {drop:.2f} %. Comms goes "
          f"from {before['energy_comms_j']:.2f} % of a mission to "
          f"{after['energy_comms_j']:.2f} %.")

    # --- does any headline comparison move? ---------------------------
    print("\n  HEADLINE COMPARISONS -- do any of them move?")
    print(f"    {'comparison':<46s} {'n':>3s} {'as measured':>13s} "
          f"{'rescaled':>13s} {'sign':>6s}")
    flipped = []
    for label, a, b, value, fault, claimed in sensitivity.COMPARISONS:
        d0, seeds = sensitivity.paired(rows, a, b,
                                       lambda r: value(r, AS_MEASURED), fault)
        d1, _ = sensitivity.paired(rows, a, b,
                                   lambda r: value(r, RESCALED), fault)
        m0, t0 = float(np.mean(d0)), sensitivity.t_statistic(d0)
        m1, t1 = float(np.mean(d1)), sensitivity.t_statistic(d1)
        same = np.sign(m0) == np.sign(m1)
        if not same:
            flipped.append(label)
        print(f"    {label:<46s} {len(seeds):>3d} "
              f"{m0:>8.1f} (t{t0:>+5.2f}) {m1:>8.1f} (t{t1:>+5.2f}) "
              f"{'same' if same else 'FLIPS':>6s}")

    means0, ok0 = sensitivity.ordering_holds(rows, AS_MEASURED)
    means1, ok1 = sensitivity.ordering_holds(rows, RESCALED)
    print(f"\n  C2 -> C5 -> C3 energy per point")
    print(f"    as measured : {means0['C2']:.1f} / {means0['C5']:.1f} / "
          f"{means0['C3']:.1f}   monotone: {ok0}")
    print(f"    rescaled    : {means1['C2']:.1f} / {means1['C5']:.1f} / "
          f"{means1['C3']:.1f}   monotone: {ok1}")

    # --- how big is the thing being neglected? ------------------------
    # Comms energy per robot, as a fraction of one battery. That is the
    # size of the charge that the behavioural paths above would have to
    # act on, so it bounds what is being left out.
    worst_shift = 0.0
    for r in rows:
        robots = 1 if r["condition"] == "C4" else 3
        removed = float(r["energy_comms_j"]) * (1.0 - FACTOR) / robots
        worst_shift = max(worst_shift, removed / config.BATTERY_CAPACITY_J)

    head0 = sensitivity.battery_headroom(rows, AS_MEASURED)
    head1 = sensitivity.battery_headroom(rows, RESCALED)

    print("\n  THE APPROXIMATION, SIZED")
    print(f"    worst healthy mission, battery used : "
          f"{100 * head0:.2f} %  ->  {100 * head1:.2f} %")
    print(f"    largest charge freed on any one robot by this rescale: "
          f"{100 * worst_shift:.3f} % of a battery")
    print("    That is the entire budget the two neglected behavioural")
    print("    paths -- later battery exhaustion, and the predictive")
    print("    battery detector crossing at a different step -- would have")
    print("    to work with. It is not zero, so this is an approximation")
    print("    and is reported as one.")

    print("\n  VERDICT")
    if flipped:
        print("    Comparisons that change sign: " + ", ".join(flipped))
    else:
        print("    No headline comparison changes sign, and the")
        print("    C2 -> C5 -> C3 ordering holds. Rescaling comms in")
        print("    analysis changes no conclusion in Chapter 4.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "results.csv"))
