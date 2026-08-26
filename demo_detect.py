"""
demo_detect.py
==============
STEP 4: does the squad notice?

Run:  python demo_detect.py

Every fault from Step 3, run against three seeds, plus a no-fault control
on each. For each run it reports whether the squad identified the right
robot with the right fault, how long that took, and how many healthy
robots got accused of something along the way.

WHAT COUNTS AS A DETECTION
--------------------------
The correct robot accused of the correct fault by at least one peer.
Latency is measured from the step the fault was injected to the step of
the first such accusation.

WHAT COUNTS AS A FALSE POSITIVE
-------------------------------
Any accusation against a robot that has nothing wrong with it. On the
control runs that is every accusation, by definition. Note that a faulty
robot can itself raise false accusations -- a robot whose radio has died
hears nothing from anybody and concludes that everybody else has failed.
That is counted here, because it is a real cost of the mechanism, and it
is the argument for Step 5 acting on corroboration rather than on any
single robot's word.

NOTHING IS RECOVERED
--------------------
No robot is quarantined, no map is rolled back, no work is reallocated.
The squad notices and carries on regardless, which is why the mission
outcomes here match Step 3's. Acting on what is noticed is Step 5.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import demo_squad

SEEDS = [42, 7, 2024]
VICTIM = config.FAULT_DEMO_ROBOT
WHEN = config.FAULT_DEMO_STEP


def score_run(out, fault):
    """
    Scoring rules for one completed run, kept in one place so the sweep in
    sweep_comms.py and this file cannot drift apart on what counts as a
    detection.
    """
    facility, squad, points, deviations, history, trace, stats = out

    detections = stats["detections"]

    # A DETECTION MUST COME AFTER THE FAULT. Scoring any accusation of the
    # right robot with the right name as a success produced negative
    # latencies -- an accusation made 80 s before the robot was broken was
    # being counted as having detected it. That is a false positive wearing
    # the right label, and it flatters the detector twice over: once by
    # counting a hit, and again by not counting the miss.
    hit_step = None
    if fault is not None:
        found = detections.get(VICTIM, {}).get(fault)
        if found is not None and found[0] >= WHEN:
            hit_step = found[0]

    # Anything said about a robot that is not broken, about a broken robot
    # in a way it is not broken, or about the right fault before it existed.
    false_positives = []
    for accused_id, faults_seen in detections.items():
        for name, (step, accusers) in faults_seen.items():
            correct = (fault is not None and accused_id == VICTIM
                       and name == fault and step >= WHEN)
            if correct:
                continue
            false_positives.append((accused_id, name, step, accusers))

    return {
        "fault": fault or "none (control)",
        "detected": hit_step is not None,
        "latency_s": ((hit_step - WHEN) * config.DT_S
                      if hit_step is not None else None),
        "accusers": (detections.get(VICTIM, {}).get(fault, (None, []))[1]
                     if fault is not None else []),
        "false_positives": false_positives,
        "visited": sum(1 for p in points if p.visited),
        "total": len(points),
        "steps": stats["steps"],
        "excused": stats["silence_excused"],
        "energy": demo_squad.squad_metrics(facility, squad, points)["energy"],
        "contact": demo_squad.contact_fraction(trace, len(squad)),
    }


def assess(seed, fault):
    """One run. Returns what was detected, when, and what was wrongly
    accused."""
    faults = [] if fault is None else [(VICTIM, WHEN, fault)]
    # Detection on, recovery explicitly OFF. This file measures how long
    # the squad takes to notice and how often it accuses the innocent; if
    # it quarantined anybody the accusations would start changing the maps
    # the next accusation is drawn from, and the latencies would no longer
    # mean what the column header says.
    out = demo_squad.run(seed, verbose=False, faults=faults, recover=False)
    row = score_run(out, fault)
    row["seed"] = seed
    return row


def main():
    rows = []
    for seed in SEEDS:
        rows.append(assess(seed, None))
        for fault in config.FAULT_TYPES:
            rows.append(assess(seed, fault))

    print("\n" + "=" * 88)
    print("  STEP 4 -- DETECTION. Five faults, three seeds, plus controls.")
    print(f"  Robot {VICTIM} broken at t = {WHEN * config.DT_S:.0f} s. "
          "Nothing is recovered.")
    print("=" * 88)
    print(f"  {'seed':<6s} {'fault':<20s} {'detected':>9s} {'latency':>9s} "
          f"{'accusers':>10s} {'false pos':>10s} {'points':>8s}")
    print("-" * 88)
    for r in rows:
        det = "-" if r["fault"].startswith("none") else (
            "YES" if r["detected"] else "no")
        lat = f"{r['latency_s']:.0f} s" if r["latency_s"] is not None else "-"
        acc = ",".join(str(a) for a in r["accusers"]) or "-"
        print(f"  {r['seed']:<6d} {r['fault']:<20s} {det:>9s} {lat:>9s} "
              f"{acc:>10s} {len(r['false_positives']):>10d} "
              f"{r['visited']:>4d}/{r['total']:<3d}")

    # ---- summary by fault -------------------------------------------
    print("-" * 88)
    print("  BY FAULT")
    print(f"  {'fault':<20s} {'detected':>10s} {'mean latency':>14s} "
          f"{'false positives':>16s}")
    for fault in config.FAULT_TYPES:
        mine = [r for r in rows if r["fault"] == fault]
        hits = [r for r in mine if r["detected"]]
        lats = [r["latency_s"] for r in hits]
        fp = sum(len(r["false_positives"]) for r in mine)
        mean_lat = f"{np.mean(lats):.0f} s" if lats else "-"
        print(f"  {fault:<20s} {len(hits)}/{len(mine):<8d} {mean_lat:>14s} "
              f"{fp:>16d}")

    controls = [r for r in rows if r["fault"].startswith("none")]
    control_fp = sum(len(r["false_positives"]) for r in controls)
    print("-" * 88)
    print(f"  CONTROL RUNS (nothing broken): {control_fp} false positive(s) "
          f"across {len(controls)} runs")
    for r in controls:
        for accused, name, step, who in r["false_positives"]:
            print(f"     seed {r['seed']}: robot {accused} accused of {name} "
                  f"by {who} at {step * config.DT_S:.0f} s")
    print(f"  Silences excused by the range gate: "
          f"{sum(r['excused'] for r in controls):,} across the control runs")
    print("=" * 88 + "\n")

    _figure(rows)
    return rows


def _figure(rows):
    """Latency per fault per seed, and where the false positives fall."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    faults = list(config.FAULT_TYPES)
    width = 0.25
    for i, seed in enumerate(SEEDS):
        vals = []
        for f in faults:
            r = next(r for r in rows if r["seed"] == seed and r["fault"] == f)
            vals.append(r["latency_s"] if r["detected"] else 0)
        ax.bar(np.arange(len(faults)) + i * width, vals, width,
               label=f"seed {seed}")
    ax.set_xticks(np.arange(len(faults)) + width)
    ax.set_xticklabels([f.replace("_", "\n") for f in faults], fontsize=8)
    ax.set_ylabel("detection latency (s)")
    ax.set_title("Time from fault to first correct accusation\n"
                 "(zero = never detected)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    labels = ["control"] + faults
    counts = []
    for lab in labels:
        key = "none (control)" if lab == "control" else lab
        counts.append(sum(len(r["false_positives"]) for r in rows
                          if r["fault"] == key))
    ax.bar(range(len(labels)), counts, color="#C1442E")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([lab.replace("_", "\n") for lab in labels], fontsize=8)
    ax.set_ylabel("false accusations (3 seeds)")
    ax.set_title("Healthy robots wrongly accused")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig("demo_detect_result.png", dpi=125)
    print("Saved demo_detect_result.png")


if __name__ == "__main__":
    main()
