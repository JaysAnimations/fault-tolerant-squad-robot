"""
run_experiments.py
==================
STEP 6: run the whole thing and write one row per mission.

    python run_experiments.py --validate    one seed, all six conditions
    python run_experiments.py               the full 180-run suite
    python run_experiments.py --seeds 7 42  just those seeds

Six conditions x 30 seeds = 180 rows in results.csv. Every figure in
Chapter 4 comes out of that file and nothing in the report is produced by
hand.

PAIRED SEEDING
--------------
Seed 7 gives the same inspection points, the same deviations, the same
sensor noise and the same fault at the same step in every condition it
appears in. Only the condition differs. A difference between two rows
sharing a seed is caused by the condition and by nothing else -- that is
where the statistical power comes from, and reseeding per condition would
throw it away.

RESUMABLE, BECAUSE IT HAS TO BE
-------------------------------
Each row is written and flushed the moment its run finishes, and on
restart every run_id already in results.csv is skipped. A suite this long
that had to start over after an interruption would cost an evening, and
the failure mode is not hypothetical -- it is a laptop lid closing.

TRACES ARE SAVED FOR EVERY RUN
------------------------------
~24 KB each, a few megabytes for the suite. Without them the demonstration
video can only be made by running all 180 missions again, and by then the
code will have moved on and the numbers will not match the figures.

NOT-APPLICABLE VALUES ARE -1, NEVER BLANK AND NEVER NaN
-------------------------------------------------------
C0 and C1 have no fault, so detection latency is not zero, it is
meaningless. Writing 0 would be a lie that averages silently; writing NaN
would poison every later aggregation. -1 is out of range for every column
that uses it and has to be filtered on purpose.
"""

import csv
import os
import sys
import time

import numpy as np

import config
import demo_squad
from demo_detect import score_run


# ---------------------------------------------------------------------
# The six conditions
# ---------------------------------------------------------------------
# Written out rather than derived, because the whole argument of Chapter 4
# is what each one has and has not got. Reading down the columns should
# make the progression C2 -> C5 -> C3 obvious.
#
#   deviations      are the drawings out of date?
#   fault           is a robot broken?
#   reallocation   does an unfinished claim ever return to the pool?
#   detect          does anybody notice a broken robot?
#   recover         does anybody act on it?
CONDITIONS = {
    "C0": dict(deviations=False, fault=False, reallocation=True,
               detect=True, recover=True, robots=3,
               note="perfect drawings, no faults -- theoretical best case"),
    "C1": dict(deviations=True, fault=False, reallocation=True,
               detect=True, recover=True, robots=3,
               note="out-of-date drawings, no faults"),
    "C2": dict(deviations=True, fault=True, reallocation=False,
               detect=False, recover=False, robots=3,
               note="naive: claims are permanent, nothing is reallocated"),
    "C5": dict(deviations=True, fault=True, reallocation=True,
               detect=False, recover=False, robots=3,
               note="ablation: claims lapse and are re-auctioned, but "
                    "nobody detects anything"),
    "C3": dict(deviations=True, fault=True, reallocation=True,
               detect=True, recover=True, robots=3,
               note="the proposed system"),
    "C4": dict(deviations=True, fault=True, reallocation=False,
               detect=False, recover=False, robots=1,
               note="one robot -- reallocation is not possible at all"),
}

CONDITION_ORDER = ["C0", "C1", "C2", "C5", "C3", "C4"]

COLUMNS = [
    "run_id", "condition", "seed",
    "fault_type", "fault_robot", "fault_step",
    # BOTH counts, always. `points_believed_visited` is what the squad
    # thinks it inspected; `points_truly_visited` is what it actually did,
    # scored on true positions. The gap is the wrong-position fault's real
    # damage and it is a headline result, not bookkeeping.
    "points_believed_visited", "points_truly_visited", "points_invalidated",
    "points_visited", "points_total", "points_unreachable", "mission_success",
    "duration_s", "total_energy_j", "energy_per_point_j", "energy_per_m2_j",
    "distance_total_m", "collisions", "robots_alive_at_end",
    "coverage_pct", "surface_f1", "observed_error_pct",
    "deviations_injected", "deviations_detected", "mean_detection_time_s",
    "fault_detected", "detection_latency_s", "false_positives",
    "quarantines_correct", "quarantines_wrong",
    "cells_restored", "points_reallocated",
    "duplicated_coverage_pct", "contact_fraction",
]

NA = -1          # "does not apply to this condition" -- never 0, never NaN


def fault_for_seed(seed):
    """
    Which fault this seed gets, drawn from its own stream.

    Its own stream so that adding a condition, or changing how many random
    numbers a mission consumes, cannot change which fault seed 7 gets. The
    fault has to be a property of the seed alone or the pairing breaks.
    """
    rng = np.random.default_rng([seed, config.RNG_STREAM_FAULTS])
    return str(rng.choice(config.FAULT_TYPES))


def build_row(condition, seed, out, wall_s):
    """Turn one finished mission into one CSV row."""
    facility, squad, points, deviations, history, trace, stats = out
    spec = CONDITIONS[condition]

    m = demo_squad.squad_metrics(facility, squad, points)
    coverage, f1, obs_err = demo_squad.squad_map_metrics(facility, squad)

    duration_s = stats["steps"] * config.DT_S
    visited = m["visited"]
    area = m["area_seen_m2"]

    # --- the fault, and whether anybody noticed ----------------------
    if spec["fault"]:
        fault_type = fault_for_seed(seed)
        fault_robot = (config.EXPERIMENT_FAULT_ROBOT if spec["robots"] > 1
                       else 0)
        fault_step = config.EXPERIMENT_FAULT_STEP
        scored = score_run(out, fault_type)
        detected = 1 if scored["detected"] else 0
        latency = (scored["latency_s"] if scored["latency_s"] is not None
                   else NA)
        false_pos = len(scored["false_positives"])
    else:
        fault_type, fault_robot, fault_step = "none", NA, NA
        detected, latency = NA, NA
        # A healthy squad cannot have a false NEGATIVE, but it can very
        # much have a false positive, and on the fault-free conditions
        # every accusation is one by definition. This is the cleanest
        # measurement of false-positive rate in the whole suite.
        false_pos = sum(len(f) for f in stats["detections"].values())

    # --- deviations ---------------------------------------------------
    found = [d for d in deviations if d.detected]
    det_times = [d.detected_step * config.DT_S for d in found
                 if d.detected_step is not None]

    # --- recovery -----------------------------------------------------
    actions = stats["recovery"]
    quarantines = [a for a in actions if a["rolled_back"]]
    correct = sum(1 for a in quarantines if a["suspect"] == fault_robot)
    wrong = len(quarantines) - correct
    # WORK ACTUALLY TAKEN OFF THE SUSPECT, not work that merely happened
    # afterwards. The first version counted every point finished after the
    # first recovery action by anyone other than the suspect, which on a
    # fault-free condition reported 20 points "reallocated" because one
    # spurious accusation early in the round swept up the entire remaining
    # mission. A claim that was released and then completed by somebody
    # else is the only thing that has actually been reallocated.
    suspects = {a["suspect"] for a in actions}
    released = {}
    for a in actions:
        for index in a["claims_released"]:
            released.setdefault(index, a["step"])
    by_index = {p.index: p for p in points}
    reallocated = 0
    for index, when in released.items():
        p = by_index.get(index)
        if (p is not None and p.visited and p.visit_step is not None
                and p.visit_step > when and p.visited_by not in suspects):
            reallocated += 1

    return {
        "run_id": f"{condition}_s{seed}",
        "condition": condition,
        "seed": seed,
        "fault_type": fault_type,
        "fault_robot": fault_robot,
        "fault_step": fault_step,
        "points_believed_visited": m["believed"],
        "points_truly_visited": m["truly"],
        "points_invalidated": m["invalidated"],
        "points_visited": visited,
        "points_total": m["total"],
        "points_unreachable": m["unreachable"],
        "mission_success": 1 if m["success"] else 0,
        "duration_s": round(duration_s, 1),
        "total_energy_j": round(m["energy"], 1),
        "energy_per_point_j": round(m["energy"] / max(visited, 1), 1),
        "energy_per_m2_j": round(m["energy"] / max(area, 1e-6), 3),
        "distance_total_m": round(m["distance"], 1),
        "collisions": sum(s.robot.collisions for s in squad),
        "robots_alive_at_end": sum(1 for s in squad if s.robot.alive),
        "coverage_pct": round(coverage, 2),
        "surface_f1": round(f1, 4),
        "observed_error_pct": round(obs_err, 3),
        "deviations_injected": len(deviations),
        "deviations_detected": len(found),
        "mean_detection_time_s": (round(float(np.mean(det_times)), 1)
                                  if det_times else NA),
        "fault_detected": detected,
        "detection_latency_s": (round(latency, 1) if latency != NA else NA),
        "false_positives": false_pos,
        "quarantines_correct": correct,
        "quarantines_wrong": wrong,
        "cells_restored": sum(a["cells_restored"] for a in actions),
        "points_reallocated": reallocated,
        "duplicated_coverage_pct": round(m["redundancy"] * 100.0, 2),
        "contact_fraction": round(
            demo_squad.contact_fraction(trace, len(squad)), 4),
        "_wall_s": round(wall_s, 1),
    }


def run_one(condition, seed, save_trace=True):
    """One mission, plus its row."""
    spec = CONDITIONS[condition]
    faults = []
    if spec["fault"]:
        robot = config.EXPERIMENT_FAULT_ROBOT if spec["robots"] > 1 else 0
        faults = [(robot, config.EXPERIMENT_FAULT_STEP, fault_for_seed(seed))]

    started = time.time()
    out = demo_squad.run(seed, verbose=False,
                         with_deviations=spec["deviations"],
                         reallocation=spec["reallocation"],
                         detect=spec["detect"],
                         recover=spec["recover"],
                         squad_size=spec["robots"],
                         faults=faults)
    wall = time.time() - started

    row = build_row(condition, seed, out, wall)

    if save_trace:
        os.makedirs(config.TRACE_DIR, exist_ok=True)
        out[5].save(os.path.join(config.TRACE_DIR, row["run_id"] + ".npz"))
    return row


# ---------------------------------------------------------------------
# The CSV, written as we go
# ---------------------------------------------------------------------
def already_done(path):
    """run_ids already in results.csv, so a restart skips them."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as fh:
        return {r["run_id"] for r in csv.DictReader(fh) if r.get("run_id")}


def open_writer(path):
    """Append if the file exists, otherwise create it with a header."""
    fresh = not os.path.exists(path)
    fh = open(path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
    if fresh:
        writer.writeheader()
        fh.flush()
    return fh, writer


def validate(seed=7):
    """
    One seed across all six conditions, checked hard before anything long
    is launched.

    Checks the things that would otherwise be discovered three hours into
    the suite: every column present, nothing blank, nothing NaN, the right
    number of rows, and the paired-seeding property actually holding.
    """
    print(f"\nVALIDATION -- seed {seed}, all six conditions\n")
    rows = []
    for condition in CONDITION_ORDER:
        started = time.time()
        row = run_one(condition, seed, save_trace=False)
        rows.append(row)
        print(f"  {condition}  {CONDITIONS[condition]['note'][:44]:<44s} "
              f"{row['points_visited']:>2d}/{row['points_total']:<2d} "
              f"{row['duration_s']:>6.0f}s  ({row['_wall_s']:.0f}s wall)")

    problems = []
    if len(rows) != len(CONDITION_ORDER):
        problems.append(f"expected {len(CONDITION_ORDER)} rows, got {len(rows)}")

    for row in rows:
        for column in COLUMNS:
            if column not in row:
                problems.append(f"{row['run_id']}: column '{column}' missing")
                continue
            value = row[column]
            if value is None or value == "":
                problems.append(f"{row['run_id']}: '{column}' is blank")
            elif isinstance(value, float) and np.isnan(value):
                problems.append(f"{row['run_id']}: '{column}' is NaN")

    # Paired seeding: the conditions that inject a fault must all inject
    # the SAME one, or the comparison is between different scenarios.
    faulted = [r for r in rows if r["fault_type"] != "none"]
    if len({r["fault_type"] for r in faulted}) > 1:
        problems.append("fault type differs between conditions on one seed")
    if len({r["points_total"] for r in rows}) > 1:
        problems.append("point count differs between conditions on one seed")

    print()
    print(f"  {'run_id':<10s} {'fault':<20s} {'pts':>7s} {'succ':>5s} "
          f"{'dur_s':>7s} {'J/pt':>7s} {'cov%':>6s} {'F1':>6s} "
          f"{'det':>4s} {'lat':>6s} {'fp':>3s} {'Qok':>4s} {'Qbad':>5s} "
          f"{'cells':>8s} {'realloc':>8s} {'contact':>8s}")
    for r in rows:
        print(f"  {r['run_id']:<10s} {r['fault_type']:<20s} "
              f"{r['points_visited']:>3d}/{r['points_total']:<3d} "
              f"{r['mission_success']:>5d} {r['duration_s']:>7.0f} "
              f"{r['energy_per_point_j']:>7.0f} {r['coverage_pct']:>6.1f} "
              f"{r['surface_f1']:>6.3f} {r['fault_detected']:>4d} "
              f"{r['detection_latency_s']:>6.0f} {r['false_positives']:>3d} "
              f"{r['quarantines_correct']:>4d} {r['quarantines_wrong']:>5d} "
              f"{r['cells_restored']:>8d} {r['points_reallocated']:>8d} "
              f"{r['contact_fraction']:>8.3f}")

    print()
    if problems:
        print("  VALIDATION FAILED:")
        for p in problems:
            print("    -", p)
        return False, rows

    wall = sum(r["_wall_s"] for r in rows)
    per_run = wall / len(rows)
    total = per_run * len(CONDITION_ORDER) * len(config.EXPERIMENT_SEEDS)
    print(f"  All {len(COLUMNS)} columns populated on all {len(rows)} rows. "
          "No blanks, no NaNs.")
    print(f"  Seed {seed} took {wall:.0f} s for six conditions "
          f"({per_run:.0f} s per run).")
    print(f"  ESTIMATE for {len(CONDITION_ORDER)} x "
          f"{len(config.EXPERIMENT_SEEDS)} = "
          f"{len(CONDITION_ORDER) * len(config.EXPERIMENT_SEEDS)} runs: "
          f"{total / 60:.0f} min ({total / 3600:.1f} h)")
    print("  -- seed 7 is a fast seed; the awkward ones (2024) run longer,")
    print("     so treat this as a lower bound.")
    return True, rows


def main(seeds=None):
    seeds = list(config.EXPERIMENT_SEEDS if seeds is None else seeds)
    done = already_done(config.RESULTS_CSV)
    todo = [(c, s) for s in seeds for c in CONDITION_ORDER
            if f"{c}_s{s}" not in done]

    print(f"\n{len(CONDITION_ORDER)} conditions x {len(seeds)} seeds = "
          f"{len(CONDITION_ORDER) * len(seeds)} runs")
    print(f"{len(done)} already in {config.RESULTS_CSV}, {len(todo)} to go\n")
    if not todo:
        print("Nothing to do.")
        return

    fh, writer = open_writer(config.RESULTS_CSV)
    started = time.time()
    try:
        for i, (condition, seed) in enumerate(todo, 1):
            row = run_one(condition, seed)
            writer.writerow(row)
            fh.flush()          # a row on disk survives a closed lid
            elapsed = time.time() - started
            rate = elapsed / i
            print(f"  [{i:>3d}/{len(todo)}] {row['run_id']:<10s} "
                  f"{row['points_visited']:>2d}/{row['points_total']:<2d} "
                  f"{row['duration_s']:>6.0f}s sim  {row['_wall_s']:>5.0f}s "
                  f"wall  | eta {(rate * (len(todo) - i)) / 60:.0f} min",
                  flush=True)
    finally:
        fh.close()
    print(f"\nDone in {(time.time() - started) / 60:.0f} min -> "
          f"{config.RESULTS_CSV}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--validate" in args:
        ok, _ = validate()
        sys.exit(0 if ok else 1)
    if "--seeds" in args:
        chosen = [int(a) for a in args[args.index("--seeds") + 1:]]
        main(chosen)
    else:
        main()
