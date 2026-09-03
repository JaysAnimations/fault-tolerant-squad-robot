"""
cadence_probe.py  --  Session 15, Stage B
=========================================
Does halving the Byzantine detector's cadence buy detection, and what does
it cost at the false-positive gate?

    py session15_measurements/cadence_probe.py

THE ARGUMENT BEING TESTED
-------------------------
Session 14's displacement sweep found no knee: detection saturates at 43 %
even at 24 m, which is 133x the measured 0.18 m healthy drift. The overlap
probe then traced the cause to shared evidence rather than threshold --
displacing a robot further shrinks the triple-overlap region 42 % and cuts
the number of occasions the detector can run at all by 61 % (183 checks at
2 m down to 72 at 24 m).

That points at OPPORTUNITY as the binding constraint. check_wrong_position
is throttled to every 100 steps -- a Session 13 performance fix, not a
detection decision -- and BYZANTINE_MIN_CHECKS = 5 requires five
consecutive checks, so confirmation takes 500 steps. Halving the cadence
doubles the opportunities and halves the confirmation time. False
quarantines are 0.00 at every displacement in Stage 4a and 0/10 on the C1
gate, so there is headroom to spend.

WHY EACH CADENCE IS RE-RUN RATHER THAN READ OUT OF results.csv
--------------------------------------------------------------
results.csv was produced with E_COMMS_J_PER_KB = 0.05. Session 15 derived
that coefficient from ESP32 radio airtime and it is now 0.0057344. Session
14 established that an energy coefficient is NOT purely accounting: it
feeds back through battery exhaustion, and detection.py's predictive
battery check projects energy_j forward against remaining charge, so it
crosses threshold at a different STEP when the scale moves. 18 of 145 rows
shifted the last time a coefficient changed.

So reading the cadence-100 arm out of results.csv would compare cadence 50
under the new coefficient against cadence 100 under the old one, and any
difference would be unattributable. Both arms are therefore re-run under
identical config, differing in cadence and nothing else. The cadence-100
arm is compared against results.csv separately, which measures the
coefficient's footprint on these seeds for free.

NOTHING IS WRITTEN TO config.py. The cadence is overridden in this
process only, and the file's SHA-256 is checked at the end.
"""
import csv
import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import run_experiments

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PY = os.path.join(os.path.dirname(HERE), "config.py")
OUT_CSV = os.path.join(HERE, "cadence_probe.csv")

def wrong_position_seeds():
    """
    The seeds whose injected fault is wrong_position, read out of
    results.csv rather than typed in.

    Which fault a seed draws is a property of the seed, so a hand-copied
    list is a second copy of that fact and would stop matching silently if
    the schedule ever moved. Reading the dataset also picks up the seed-14
    exclusion for free -- it is absent from the file, so it cannot creep
    back in here. Asserted against the expected set so a change is loud.
    """
    path = os.path.join(os.path.dirname(HERE), config.RESULTS_CSV)
    with open(path, newline="") as fh:
        seeds = sorted({int(r["seed"]) for r in csv.DictReader(fh)
                        if r["fault_type"] == "wrong_position"})
    assert tuple(seeds) == (4, 6, 10, 12, 21, 22, 42), seeds
    return tuple(seeds)

# The ten seeds check_false_positives.py has used since Session 13. Kept
# identical so this gate is comparable with the one already reported.
C1_GATE_SEEDS = (1, 2, 3, 4, 5, 7, 11, 13, 42, 2024)

CADENCES = (100, 50)


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def run_arm(condition, seeds, cadence):
    """One condition across some seeds at one detector cadence."""
    original = config.BYZANTINE_CHECK_EVERY_N_STEPS
    config.BYZANTINE_CHECK_EVERY_N_STEPS = cadence
    rows = []
    try:
        for seed in seeds:
            started = time.time()
            # save_trace=False: this probe must not write into traces/,
            # which holds the suite's 174 deliverable traces.
            row = run_experiments.run_one(condition, seed, save_trace=False)
            row["cadence"] = cadence
            row["wall_s"] = round(time.time() - started, 1)
            rows.append(row)
            print(f"    {condition}_s{seed:<5d} cadence {cadence:>3d}  "
                  f"detected {row['fault_detected']:>2}  "
                  f"lat {row['detection_latency_s']:>6}  "
                  f"Qok {row['quarantines_correct']}  "
                  f"Qbad {row['quarantines_wrong']}  "
                  f"truly {row['points_truly_visited']:>2}  "
                  f"falsely {row['points_falsely_reported']:>2}  "
                  f"({row['wall_s']:.0f}s)", flush=True)
    finally:
        config.BYZANTINE_CHECK_EVERY_N_STEPS = original
    return rows


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def summarise_detection(rows, label):
    """The six quantities Stage B asks for, plus wall clock."""
    n = len(rows)
    detected = [r for r in rows if r["fault_detected"] == 1]
    # Latency is averaged over the runs that detected. Averaging -1 in for
    # the misses would report a latency no run ever had.
    return {
        "label": label,
        "n": n,
        "detected": len(detected),
        "rate": len(detected) / n,
        "latency": mean(r["detection_latency_s"] for r in detected),
        "q_correct": sum(r["quarantines_correct"] for r in rows),
        "q_wrong": sum(r["quarantines_wrong"] for r in rows),
        "truly": mean(r["points_truly_visited"] for r in rows),
        "falsely": mean(r["points_falsely_reported"] for r in rows),
        "wall": mean(r["wall_s"] for r in rows),
        "duration": mean(r["duration_s"] for r in rows),
    }


def summarise_gate(rows, label):
    """C1: nothing is broken, so every accusation and quarantine is false."""
    n = len(rows)
    return {
        "label": label,
        "n": n,
        "runs_with_quarantine": sum(1 for r in rows
                                    if r["quarantines_wrong"] > 0),
        "quarantines": sum(r["quarantines_wrong"] for r in rows),
        "accusations": sum(r["false_positives"] for r in rows),
        "runs_with_accusation": sum(1 for r in rows if r["false_positives"] > 0),
        "invalidated": sum(r["points_invalidated"] for r in rows),
        "believed": sum(r["points_believed_visited"] for r in rows),
        "truly": sum(r["points_truly_visited"] for r in rows),
        "total": sum(r["points_total"] for r in rows),
        "wall": mean(r["wall_s"] for r in rows),
    }


def stored_rows(condition, seeds):
    """The same runs as they stand in results.csv (cadence 100, old comms)."""
    path = os.path.join(os.path.dirname(HERE), config.RESULTS_CSV)
    want = {f"{condition}_s{s}" for s in seeds}
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["run_id"] in want:
                out.append({k: (int(v) if k in ("fault_detected",
                                                "quarantines_correct",
                                                "quarantines_wrong",
                                                "points_truly_visited",
                                                "points_falsely_reported",
                                                "points_believed_visited",
                                                "points_invalidated",
                                                "points_total",
                                                "false_positives")
                                else float(v) if k in ("detection_latency_s",
                                                       "duration_s")
                                else v)
                            for k, v in r.items()})
    for r in out:
        r["wall_s"] = float("nan")
    return out


def main():
    seeds_wp = wrong_position_seeds()
    before = sha256(CONFIG_PY)
    print("\nSTAGE B -- the cadence probe")
    print(f"config.py SHA-256 before : {before}")
    print(f"BYZANTINE_CHECK_EVERY_N_STEPS on disk = "
          f"{config.BYZANTINE_CHECK_EVERY_N_STEPS}")
    print(f"E_COMMS_J_PER_KB = {config.E_COMMS_J_PER_KB:.7f} "
          f"(results.csv was produced at 0.05)")

    all_rows = []

    # --- B1: detection on the seven wrong_position seeds ---------------
    print(f"\nB1 -- C3 on the {len(seeds_wp)} wrong_position "
          f"seeds {seeds_wp}")
    b1 = {}
    for cadence in CADENCES:
        print(f"\n  cadence {cadence}:")
        b1[cadence] = run_arm("C3", seeds_wp, cadence)
        all_rows += b1[cadence]

    # --- B2: the false-positive gate -----------------------------------
    print(f"\nB2 -- C1 false-positive gate, {len(C1_GATE_SEEDS)} seeds "
          f"{C1_GATE_SEEDS}")
    b2 = {}
    for cadence in CADENCES:
        print(f"\n  cadence {cadence}:")
        b2[cadence] = run_arm("C1", C1_GATE_SEEDS, cadence)
        all_rows += b2[cadence]

    # --- every row kept, so the tables below are auditable -------------
    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["cadence", "wall_s"] + run_experiments.COLUMNS,
            extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    # =================================================================
    # B1 table
    # =================================================================
    print("\n" + "=" * 78)
    print("B1 -- DETECTION, C3, wrong_position seeds, n = 7")
    print("=" * 78)
    arms = [summarise_detection(stored_rows("C3", seeds_wp),
                                "results.csv (cad 100, old comms)")]
    arms += [summarise_detection(b1[c], f"re-run cadence {c}")
             for c in CADENCES]

    print(f"  {'arm':<34s} {'det':>7s} {'rate':>6s} {'lat s':>8s} "
          f"{'Qok':>4s} {'Qbad':>5s} {'truly':>7s} {'falsely':>8s} "
          f"{'wall s':>7s}")
    for a in arms:
        rate = f"{100 * a['rate']:.0f} %"
        lat = "--" if a["detected"] == 0 else f"{a['latency']:.1f}"
        wall = "--" if a["wall"] != a["wall"] else f"{a['wall']:.0f}"
        print(f"  {a['label']:<34s} {a['detected']:>4d}/{a['n']:<2d} "
              f"{rate:>6s} {lat:>8s} {a['q_correct']:>4d} {a['q_wrong']:>5d} "
              f"{a['truly']:>7.2f} {a['falsely']:>8.2f} {wall:>7s}")

    # Per-seed, because a rate out of seven hides which seeds moved.
    print(f"\n  per seed:  {'seed':>6s} {'cad100 det/lat':>16s} "
          f"{'cad50 det/lat':>16s} {'truly 100->50':>15s} "
          f"{'falsely 100->50':>17s}")
    for i, seed in enumerate(seeds_wp):
        a, b = b1[100][i], b1[50][i]
        print(f"             {seed:>6d} "
              f"{a['fault_detected']:>8d}/{a['detection_latency_s']:<7} "
              f"{b['fault_detected']:>8d}/{b['detection_latency_s']:<7} "
              f"{a['points_truly_visited']:>7d} ->{b['points_truly_visited']:>4d} "
              f"{a['points_falsely_reported']:>9d} ->"
              f"{b['points_falsely_reported']:>4d}")

    # =================================================================
    # The coefficient's footprint, measured rather than assumed
    # =================================================================
    print("\n" + "-" * 78)
    print("Is the cadence-100 re-run identical to results.csv? (i.e. did the")
    print("new E_COMMS_J_PER_KB change any behaviour on these seeds?)")
    stored = {r["run_id"]: r for r in stored_rows("C3", seeds_wp)}
    behavioural = ["points_truly_visited", "points_believed_visited",
                   "points_falsely_reported", "duration_s", "fault_detected",
                   "detection_latency_s", "quarantines_correct",
                   "quarantines_wrong", "cells_restored", "collisions"]
    moved = []
    for row in b1[100]:
        old = stored[row["run_id"]]
        for column in behavioural:
            if float(old[column]) != float(row[column]):
                moved.append((row["run_id"], column,
                              old[column], row[column]))
    if moved:
        print(f"  {len(moved)} behavioural differences -- the coefficient is "
              "NOT purely accounting here:")
        for run_id, column, was, now in moved:
            print(f"    {run_id:<10s} {column:<26s} {was} -> {now}")
    else:
        print("  NO behavioural column moved on any of the seven seeds.")
        print("  The comms coefficient is pure accounting here, so the")
        print("  cadence-100 arm and results.csv are the same experiment.")

    # =================================================================
    # B2 table
    # =================================================================
    print("\n" + "=" * 78)
    print("B2 -- THE FALSE-POSITIVE GATE, C1, healthy squad with "
          "deviations, n = 10")
    print("=" * 78)
    gates = [summarise_gate(stored_rows("C1", C1_GATE_SEEDS),
                            "results.csv (cad 100, old comms)")]
    gates += [summarise_gate(b2[c], f"re-run cadence {c}") for c in CADENCES]

    print(f"  {'arm':<34s} {'false Q runs':>13s} {'false Q':>8s} "
          f"{'accusations':>12s} {'invalidated':>12s} {'believed':>9s} "
          f"{'truly':>7s}")
    for g in gates:
        print(f"  {g['label']:<34s} {g['runs_with_quarantine']:>9d}/{g['n']:<3d} "
              f"{g['quarantines']:>8d} {g['accusations']:>12d} "
              f"{g['invalidated']:>12d} {g['believed']:>9d} {g['truly']:>7d}")

    gate50 = gates[-1]
    rate = 100.0 * gate50["runs_with_quarantine"] / gate50["n"]
    print(f"\n  FALSE-QUARANTINE RATE AT CADENCE 50: {rate:.0f} % "
          f"({gate50['runs_with_quarantine']}/{gate50['n']} runs)")
    if gate50["runs_with_quarantine"] == 0:
        print("  The gate holds at zero. The cadence change is not a trade.")
    else:
        print("  THE GATE HAS MOVED ABOVE ZERO. The cadence change is a")
        print("  TRADE, not an improvement, and must be reported as one.")

    # =================================================================
    print("\n" + "=" * 78)
    after = sha256(CONFIG_PY)
    source = open(CONFIG_PY, encoding="utf-8", errors="replace").read()
    on_disk_100 = "BYZANTINE_CHECK_EVERY_N_STEPS = 100" in source
    print(f"config.py SHA-256 after  : {after}")
    print(f"  unchanged on disk           : {before == after}")
    print(f"  still reads '= 100' on disk : {on_disk_100}")
    print(f"  live value back at default  : "
          f"{config.BYZANTINE_CHECK_EVERY_N_STEPS == 100}")
    print(f"\nEvery row written to {OUT_CSV}")
    return 0 if (before == after and on_disk_100) else 1


if __name__ == "__main__":
    sys.exit(main())
