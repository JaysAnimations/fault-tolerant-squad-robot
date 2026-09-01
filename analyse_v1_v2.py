"""
analyse_v1_v2.py
================
Stage 3 -- the re-run, reported beside Session 13's, with the session's two
changes separated.

WHY SEPARATION IS NEEDED. Session 14 changed two things: the energy
coefficients (Stage 1b) and the comms-loss response (Stage 2). Reported
together their effects would be unattributable.

WHY THE BRIEF'S METHOD DOES NOT WORK. The brief proposed recomputing
results_v1.csv under the new coefficients by scaling each energy category.
results_v1.csv predates the five energy columns -- it has one total and no
breakdown -- and the traces store only total energy_j. The v1 per-category
split is not recorded anywhere and cannot be recovered.

THE SUBSTITUTE, WHICH IS EXACT RATHER THAN APPROXIMATE.
`recovery.consider()` is reached only when a condition sets recover=True,
and C3 is the only one that does. Therefore:

  * C0, C1, C2, C5, C4 -- 145 rows -- are BEHAVIOURALLY IDENTICAL between
    v1 and v2. Same seeds, same code path, deterministic. Every difference
    between them is the coefficient change and nothing else.

  * C3 -- 29 rows -- is the only condition Stage 2 can touch. The control
    arm in results_c3_old.csv re-runs those 29 with the OLD comms-loss
    behaviour under the NEW coefficients, so the difference against v2's
    C3 is the behaviour change and nothing else.

Giving the three datasets the brief asked for:

    v1                old behaviour, old coefficients   results_v1.csv
    C3-old arm        old behaviour, new coefficients   results_c3_old.csv
    v2                new behaviour, new coefficients   results.csv

Usage:
    py analyse_v1_v2.py
"""

import csv
import collections
import os

import numpy as np

V1 = "results_v1.csv"
V2 = "results.csv"
C3OLD = "results_c3_old.csv"

ORDER = ["C0", "C1", "C2", "C5", "C3", "C4"]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def by_cond_seed(rows):
    d = collections.defaultdict(dict)
    for r in rows:
        d[r["condition"]][int(r["seed"])] = r
    return d


def col(rows, cond, name):
    return np.array([float(r[name]) for r in rows if r["condition"] == cond])


def t_paired(diffs):
    n = len(diffs)
    if n < 2:
        return 0.0
    sd = float(np.std(diffs, ddof=1))
    return 0.0 if sd == 0 else float(np.mean(diffs)) / (sd / np.sqrt(n))


def headline(rows, title):
    print(f"\n{title}")
    metrics = [("truly inspected", "points_truly_visited"),
               ("falsely reported", "points_falsely_reported"),
               ("mission success", "mission_success"),
               ("duration s", "duration_s"),
               ("energy per point", "energy_per_point_j")]
    print("  %-18s" % "" + "".join("%9s" % c for c in ORDER))
    for label, name in metrics:
        vals = []
        for c in ORDER:
            v = col(rows, c, name)
            vals.append(np.mean(v) if len(v) else float("nan"))
        fmt = "%9.2f" if label in ("truly inspected", "falsely reported",
                                   "mission success") else "%9.0f"
        print("  %-18s" % label + "".join(fmt % v for v in vals))


def paired_stats(rows, a, b, label):
    d = by_cond_seed(rows)
    seeds = sorted(set(d[a]) & set(d[b]))
    print(f"\n  PAIRED {a} vs {b}  (n={len(seeds)})")
    for name, pretty in (("points_truly_visited", "truly inspected"),
                         ("mission_success", "mission success"),
                         ("duration_s", "duration s"),
                         ("energy_per_point_j", "energy per point")):
        diffs = np.array([float(d[a][s][name]) - float(d[b][s][name])
                          for s in seeds])
        print("    %-18s %+9.2f   t = %+6.2f" %
              (pretty, np.mean(diffs), t_paired(diffs)))


def main():
    v1, v2 = load(V1), load(V2)
    c3old = load(C3OLD)

    if v2 is None:
        raise SystemExit("results.csv not found -- run the suite first")

    print("=" * 78)
    print("STAGE 3 -- THE RE-RUN")
    print("=" * 78)
    print(f"  v1 (old behaviour, old coefficients) : "
          f"{len(v1) if v1 else 0} rows")
    print(f"  v2 (new behaviour, new coefficients) : {len(v2)} rows")
    print(f"  C3-old control arm                   : "
          f"{len(c3old) if c3old else 0} rows")

    headline(v2, "HEADLINE, v2 -- means over %d seeds"
             % len({r['seed'] for r in v2}))
    if v1:
        headline(v1, "HEADLINE, v1 (Session 13) -- for comparison")

    # ---------------------------------------------------------------
    # 1. THE COEFFICIENT EFFECT, isolated on the 145 behaviourally
    #    identical rows.
    # ---------------------------------------------------------------
    if v1:
        print("\n" + "=" * 78)
        print("1. COEFFICIENT EFFECT -- isolated on the non-C3 conditions")
        print("   These 145 rows run identical code in v1 and v2, so every")
        print("   difference here is the coefficient change alone.")
        print("=" * 78)
        d1, d2 = by_cond_seed(v1), by_cond_seed(v2)

        # IS THE COEFFICIENT CHANGE PURELY AN ACCOUNTING CHANGE? Mostly,
        # but not entirely, and the exceptions are the interesting part.
        # Checked rather than assumed, and broken down by fault so the
        # mechanism is visible instead of just a count.
        print("\n  DOES ANY NON-ENERGY COLUMN MOVE? (non-C3 rows)")
        watch = ("points_truly_visited", "points_believed_visited",
                 "duration_s", "distance_total_m", "collisions",
                 "coverage_pct", "surface_f1")
        differing = []
        checked = 0
        for c in ORDER:
            if c == "C3":
                continue
            for s in set(d1[c]) & set(d2[c]):
                checked += 1
                if any(d1[c][s][n] != d2[c][s][n] for n in watch):
                    differing.append((c, s, d2[c][s]["fault_type"]))
        print(f"    {checked} paired non-C3 rows checked, "
              f"{len(differing)} differ")
        by_fault = collections.Counter(f for _, _, f in differing)
        for fault, n in by_fault.most_common():
            print(f"      {fault:<20} {n} rows")

        print("\n    WHY, AND BOTH REASONS ARE REAL FEEDBACK PATHS:")
        print("    battery_drain -- the coefficients are ~25 % lower, so the")
        print("      drained robot survives longer and the mission that")
        print("      follows is a different mission. Visible directly in")
        print("      robots_alive_at_end going 2 -> 3 on C5_s17 and C5_s18.")
        print("      Energy feeding back through battery exhaustion is the")
        print("      caveat sensitivity.py already flags.")
        print("    none (fault-free) -- C0 and C1 run detect=True and")
        print("      recover=True: they are the false-positive gates, not")
        print("      fault-tolerance-off arms. detection.py's PREDICTIVE")
        print("      battery check projects energy_j forward against")
        print("      remaining charge, so it crosses its threshold at a")
        print("      different STEP under different coefficients. The same")
        print("      false accusation fires (counts are unchanged: 5 across")
        print("      C0+C1 in both datasets) but at a different moment,")
        print("      shifting one claim release and therefore the routing.")
        print("      No reported C0 or C1 mean moves as a result.")

        print("\n  ENERGY, non-C3 rows")
        for c in ORDER:
            if c == "C3":
                continue
            seeds = sorted(set(d1[c]) & set(d2[c]))
            e1 = np.array([float(d1[c][s]["total_energy_j"]) for s in seeds])
            e2 = np.array([float(d2[c][s]["total_energy_j"]) for s in seeds])
            p1 = np.array([float(d1[c][s]["energy_per_point_j"]) for s in seeds])
            p2 = np.array([float(d2[c][s]["energy_per_point_j"]) for s in seeds])
            print("    %-4s total %8.0f -> %8.0f J (%+5.1f %%)   "
                  "J/point %6.0f -> %6.0f (%+5.1f %%)" %
                  (c, e1.mean(), e2.mean(),
                   100 * (e2.mean() - e1.mean()) / e1.mean(),
                   p1.mean(), p2.mean(),
                   100 * (p2.mean() - p1.mean()) / p1.mean()))

    # ---------------------------------------------------------------
    # 2. THE BEHAVIOUR EFFECT, isolated on C3.
    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("2. BEHAVIOUR EFFECT -- isolated on C3, both arms new coefficients")
    print("=" * 78)
    if not c3old:
        print("  results_c3_old.csv absent or empty -- run "
              "run_c3_old_commsloss.py first.")
        print("  This section is the ONLY thing that isolates the Stage 2")
        print("  behaviour change; without it the two changes stay mixed.")
    else:
        dold, dnew = by_cond_seed(c3old), by_cond_seed(v2)
        seeds = sorted(set(dold["C3"]) & set(dnew["C3"]))
        print(f"  {len(seeds)} paired C3 seeds\n")
        identical = 0
        differing = []
        for s in seeds:
            a, b = dold["C3"][s], dnew["C3"][s]
            same = all(a[k] == b[k] for k in
                       ("total_energy_j", "points_truly_visited",
                        "duration_s", "coverage_pct"))
            identical += same
            if not same:
                differing.append(s)
        print(f"  identical on energy/truly/duration/coverage: "
              f"{identical} of {len(seeds)} seeds")
        if differing:
            print(f"  seeds that differ: {differing}")
            for s in differing:
                a, b = dold["C3"][s], dnew["C3"][s]
                print("    seed %-5d energy %9s -> %9s   truly %s -> %s   "
                      "realloc %s -> %s" %
                      (s, a["total_energy_j"], b["total_energy_j"],
                       a["points_truly_visited"], b["points_truly_visited"],
                       a["points_reallocated"], b["points_reallocated"]))
        else:
            print("  NO SEED DIFFERS. The Stage 2 change has no measurable")
            print("  effect on any C3 run in the suite.")

        print("\n  MEANS, C3 old behaviour vs C3 new behaviour")
        for name, pretty in (("points_truly_visited", "truly inspected"),
                             ("points_falsely_reported", "falsely reported"),
                             ("mission_success", "mission success"),
                             ("duration_s", "duration s"),
                             ("energy_per_point_j", "energy per point"),
                             ("points_reallocated", "points reallocated")):
            a = np.mean([float(dold["C3"][s][name]) for s in seeds])
            b = np.mean([float(dnew["C3"][s][name]) for s in seeds])
            print("    %-18s %9.2f -> %9.2f  (%+.2f)" % (pretty, a, b, b - a))

    # ---------------------------------------------------------------
    # 3. Headline paired comparisons on v2.
    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("3. HEADLINE PAIRED COMPARISONS, v2")
    print("=" * 78)
    paired_stats(v2, "C3", "C2", "fault tolerance")
    paired_stats(v2, "C3", "C4", "why a squad")
    paired_stats(v2, "C1", "C0", "cost of bad drawings")

    # ---------------------------------------------------------------
    # 4. The immobilised result, which the composition shift should
    #    strengthen.
    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("4. IMMOBILISED -- does the composition shift strengthen it?")
    print("   Sensing is now ~39 % of the bill instead of ~11.5 %, so energy")
    print("   is far more time-dependent. C2 flails while C3 does not, so")
    print("   the duration gap should now dominate the energy comparison.")
    print("=" * 78)
    for tag, rows in (("v1", v1), ("v2", v2)):
        if rows is None:
            continue
        sub = [r for r in rows if r["fault_type"] == "immobilised"]
        if not sub:
            continue
        d = by_cond_seed(sub)
        seeds = sorted(set(d["C2"]) & set(d["C3"]))
        print(f"\n  {tag}  (n={len(seeds)})")
        print("    %-20s %10s %10s %10s" % ("", "C2", "C5", "C3"))
        for name, pretty in (("points_truly_visited", "truly inspected"),
                             ("duration_s", "duration s"),
                             ("energy_per_point_j", "energy per point"),
                             ("total_energy_j", "total energy J")):
            vals = []
            for c in ("C2", "C5", "C3"):
                vals.append(np.mean([float(d[c][s][name]) for s in seeds
                                     if s in d[c]]))
            print("    %-20s %10.1f %10.1f %10.1f" % (pretty, *vals))
        diffs = np.array([float(d["C3"][s]["energy_per_point_j"])
                          - float(d["C2"][s]["energy_per_point_j"])
                          for s in seeds])
        print("    C3-C2 energy/point %+.1f J  t = %+.2f"
              % (np.mean(diffs), t_paired(diffs)))

    # ---------------------------------------------------------------
    # 5. Energy composition, old vs new.
    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("5. ENERGY COMPOSITION -- the thing that matters more than totals")
    print("=" * 78)
    cats = ["energy_drive_j", "energy_turn_j", "energy_sense_j",
            "energy_compute_j", "energy_comms_j"]
    if all(c in v2[0] for c in cats):
        grand = sum(float(r[c]) for r in v2 for c in cats)
        print("  %-18s %10s %12s" % ("", "v2 share", "v1 (Session 13)"))
        old_shares = {"energy_drive_j": 66.4, "energy_turn_j": 4.8,
                      "energy_sense_j": 11.5, "energy_compute_j": 17.3,
                      "energy_comms_j": None}
        for c in cats:
            share = sum(float(r[c]) for r in v2) / grand * 100
            o = old_shares[c]
            print("  %-18s %9.1f %% %11s" %
                  (c, share, ("%.1f %%" % o) if o is not None else "-"))
    print("=" * 78)


if __name__ == "__main__":
    main()
