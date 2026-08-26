"""
demo_squad.py
=============
STEP 2: three robots inspect the facility together.

Run:  python demo_squad.py
      python demo_squad.py --compare      (also runs the single robot)
      python demo_squad.py --no-prior     (blank starting maps)

WHAT IS NEW COMPARED WITH demo_inspect.py
-----------------------------------------
Three robots instead of one, and every consequence of that:

  * each robot has its own OccupancyGrid. There is no shared map object
    anywhere in this project -- grep for one.
  * robots only learn what another robot saw by being within
    COMMS_RANGE_M of it when it transmits, and even then the message can
    be lost.
  * before committing to an inspection point a robot broadcasts what that
    point would cost it, and stands down if somebody else can do it
    cheaper.

WHAT TO LOOK FOR IN THE OUTPUT
------------------------------
Three robots should finish the same forty points in roughly a third of
the time. They will travel MORE total distance than one robot does --
three robots each have their own approach and return legs, and that
overhead is real. Duration is what improves, and duration is what matters
when the mission is bounded by battery rather than by patience.

A LIMITATION WORTH STATING BEFORE SOMEBODY ASKS
-----------------------------------------------
Robots collide with the facility but not with each other: step_motion
checks the ground truth grid, which contains no robots. With three robots
on an 80 x 55 m site the chance of them occupying the same half-metre is
small, and modelling it would mean a mutual-avoidance controller that is
not what this project is about. Say it out loud in Chapter 3 rather than
letting a panel find it.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

import config
from environment import Facility
from inspection import (generate_inspection_points, reachability_planner,
                        points_by_zone)
from deviations import inject_deviations, update_detection, detection_summary
from comms import Radio
from mapping import OccupancyGrid
from squad import SquadMember, TrajectoryTrace
from faults import FaultInjector
from detection import squad_accusations

ROBOT_COLOURS = ["#1D4E89", "#C1442E", "#1D9E75"]


def mission_over(points, squad):
    """
    The round ends when every point is either done, or every robot still
    alive has given up on it.

    Note it takes ALL of them giving up. One robot failing to reach a
    point says nothing about whether another one can -- they are in
    different places with different maps -- so a point only leaves the
    board when the whole squad has failed at it.
    """
    alive = [m for m in squad if m.robot.alive]
    if not alive:
        return True
    for p in points:
        if p.visited:
            continue
        if any(p.index not in m.given_up for m in alive):
            return False
    return True


def run(seed=config.DEFAULT_SEED, verbose=True, use_prior=True,
        use_auction=True, faults=None, detect=True, recover=None,
        reallocation=None, with_deviations=True, squad_size=None):
    facility = Facility()
    ranker = reachability_planner(facility)

    # --- the mission: identical to the single-robot one on this seed ---
    points = generate_inspection_points(facility, seed, ranker)
    # count=0 is condition C0: drawings that are perfectly accurate. The
    # deviation stream is drawn from either way, so switching it off does
    # not shift any other random quantity and the seeds stay paired.
    deviations = inject_deviations(facility, points, seed, ranker,
                                   verbose=verbose,
                                   count=None if with_deviations else 0)

    # --- deploy --------------------------------------------------------
    n_robots = config.SQUAD_SIZE if squad_size is None else squad_size
    poses = config.SQUAD_START_POSES[:n_robots]
    for i, (x, y, _th) in enumerate(poses):
        if not facility._has_clearance(x, y, config.ROBOT_RADIUS_M + 0.3):
            raise ValueError(f"start pose {i} at ({x}, {y}) is not clear of "
                             "obstacles -- fix SQUAD_START_POSES")

    squad = [SquadMember(i, pose, facility, seed, use_prior, use_auction,
                         reallocation)
             for i, pose in enumerate(poses)]
    radio = Radio(np.random.default_rng([seed, config.RNG_STREAM_COMMS]))
    trace = TrajectoryTrace()
    injector = FaultInjector(faults)

    history = {"t": [], "visited": [], "energy": [], "detected": []}
    merges = 0
    ended = "step limit reached"
    step = 0
    last_progress = 0
    if recover is None:
        recover = config.RECOVERY_ENABLED
    recovery_log = []

    for step in range(config.INSPECTION_MAX_STEPS):

        # ---------- 0. BREAK SOMETHING ----------
        # Before anything else this step, so the damage is in force for the
        # whole of it. Nothing is told that this happened.
        for robot_id, _at, name in injector.apply_due(squad, step):
            if verbose:
                print(f"  [fault] robot {robot_id}: {name} at step {step} "
                      f"(t = {step * config.DT_S:.0f} s)")

        # ---------- 1. SENSE ----------
        for m in squad:
            m.sense_and_map(facility, step)

        # ---------- 2. TALK ----------
        # Maps first, so a robot bids using the freshest belief it has.
        for m in squad:
            m.send_heartbeat(step, radio, squad, points)
            m.offer_map(step, radio, squad)
        for m in squad:
            merges += m.process_inbox(step)

        # ---------- 2b. NOTICE ----------
        # Each robot judges its peers from what actually reached it. No
        # robot is told it is broken, and nothing acts on an accusation --
        # recovery is Step 5.
        if detect:
            peer_ids = [other.id for other in squad]
            for m in squad:
                if not m.robot.alive:
                    continue
                m.run_detectors(peer_ids, points, step)

            # ---------- 2c. ACT ----------
            # Conclusions are broadcast, and anything two robots
            # independently agree on gets acted on. Never on one robot's
            # word -- see config.RECOVERY_MIN_ACCUSERS.
            if step % config.DETECTOR_EVERY_N_STEPS == 0:
                for m in squad:
                    for action in m.act_on_suspicions(points, step, radio,
                                                      squad, recover):
                        recovery_log.append(action)
                        if verbose:
                            print(f"  [recovery] robot {action['by']} "
                                  f"{'+'.join(action['did'])} robot "
                                  f"{action['suspect']} for "
                                  f"{action['fault']} at step {step} "
                                  f"(accusers {action['accusers']})")
                for m in squad:
                    m.process_inbox(step)

        # ---------- 3. AUCTION ----------
        # Inboxes are drained between bidders, so the second robot to bid
        # has already heard what the first one claimed. Fixed order keeps
        # the run reproducible.
        for m in squad:
            m.refresh_claim(step, radio, squad)
            m.choose_target(points, step, radio, squad)
            for other in squad:
                if other is not m:
                    other.process_inbox(step)

        # ---------- 4. DRIVE ----------
        for m in squad:
            m.drive(facility, step, radio, squad)
        for m in squad:
            m.process_inbox(step)

        # ---------- 5. REPORT DEVIATIONS ----------
        # Checked against every robot's own map. Detection latches, so the
        # squad detects a deviation if ANY of them overturned it -- which
        # is what "the squad detected 6 of 7" means.
        if use_prior and step % config.DEVIATION_CHECK_EVERY_STEPS == 0:
            for m in squad:
                update_detection(deviations, m.grid, step)

        # ---------- 6. RECORD ----------
        if step % config.TRACE_EVERY_N_STEPS == 0:
            trace.record(step, squad)
            for m in squad:
                m.record_trail()
            history["t"].append(step * config.DT_S)
            history["visited"].append(sum(1 for p in points if p.visited))
            history["energy"].append(sum(m.robot.total_energy_j for m in squad))
            history["detected"].append(sum(1 for d in deviations if d.detected))

        # ---------- 7. DONE? ----------
        if mission_over(points, squad):
            ended = "all points visited or given up on"
            if verbose:
                print(f"Round complete at step {step} "
                      f"(t = {step * config.DT_S:.0f} s)")
            break

        # Stalled: nobody has been able to take on any work for a while.
        # Whatever is left is unreachable as far as this squad is
        # concerned, so stop rather than mill about until the step ceiling.
        if any(m.target is not None for m in squad if m.robot.alive):
            last_progress = step
        elif step - last_progress > config.SQUAD_STALL_STEPS:
            ended = "no robot could reach anything still outstanding"
            if verbose:
                print(f"Round abandoned at step {step} "
                      f"(t = {step * config.DT_S:.0f} s) -- "
                      f"{sum(1 for p in points if not p.visited)} point(s) "
                      "unreachable")
            break
        if not any(m.robot.alive for m in squad):
            ended = "every robot out of energy"
            break

    if use_prior:
        for m in squad:
            update_detection(deviations, m.grid, step)

    for p in points:
        if not p.visited:
            p.abandoned = True

    trace.record(step, squad)
    stats = {"steps": step, "ended": ended, "use_prior": use_prior,
             "use_auction": use_auction, "merges": merges,
             "radio": radio.summary(), "faults": list(injector.fired),
             "detections": squad_accusations(squad),
             "silence_excused": sum(m.detector.silence_excused
                                    for m in squad),
             "recover": recover,
             "recovery": recovery_log,
             "released": sorted(set().union(*[m.released for m in squad])
                                if squad else set())}
    return facility, squad, points, deviations, history, trace, stats


# ---------------------------------------------------------------------
def squad_metrics(facility, squad, points):
    """The numbers Chapter 4 wants, computed once and shared by the report."""
    visited = [p for p in points if p.visited]
    # What the squad THINKS it inspected against what it actually did. The
    # gap is the wrong-position fault's real damage: a displaced robot
    # drives to where it believes a gauge is, believes it arrived, and
    # files an inspection it never performed. Scored on true positions,
    # which is analysis and never available to any robot.
    truly = [p for p in points if p.truly_visited]
    invalidated = [p for p in points if p.invalidated]
    unreachable = [p for p in points if p.unreachable]
    missed_reachable = [p for p in points
                        if not p.visited and not p.unreachable]

    total_distance = sum(m.robot.distance_travelled_m for m in squad)
    total_energy = sum(m.robot.total_energy_j for m in squad)

    # Redundant coverage: ground seen by more than one robot. Uses each
    # robot's OWN observations, not its merged map -- otherwise every
    # exchange would look like duplicated driving, which it is not.
    counts = np.zeros((facility.n_rows, facility.n_cols), dtype=np.uint8)
    for m in squad:
        counts += m.observed_mask().astype(np.uint8)
    seen = int((counts >= 1).sum())
    twice = int((counts >= 2).sum())

    cell_area = facility.res ** 2
    return {
        "visited": len(visited), "total": len(points),
        "believed": len(visited), "truly": len(truly),
        "invalidated": len(invalidated),
        "unreachable": len(unreachable), "missed": len(missed_reachable),
        "success": len(missed_reachable) == 0,
        "distance": total_distance, "energy": total_energy,
        "per_point": total_energy / max(len(visited), 1),
        "area_seen_m2": seen * cell_area,
        "area_twice_m2": twice * cell_area,
        "redundancy": (twice / seen) if seen else 0.0,
    }


def squad_map_metrics(facility, squad):
    """
    Coverage, surface F1 and observed-cell error for the map the squad
    collectively produced.

    ANALYSIS ONLY, AND IT MATTERS THAT THIS IS SAID OUT LOUD. No robot
    holds this map and none ever could: it is the prior plus every robot's
    own observations added together, assembled here by the analyst because
    a report needs one number for "the map the squad produced". Building it
    inside the simulation would be the global map object this project
    spends its architecture avoiding.

    Each robot's OWN observations are summed, not its merged grid, because
    merged grids already contain each other and adding them would count the
    same evidence three times.

    The error is scored over cells the squad actually observed, per the
    convention in CLAUDE.md -- scoring all 440,000 would mostly measure how
    accurate the drawings are.
    """
    union = OccupancyGrid(facility, owner_id=-1)
    if squad and squad[0].grid.prior_L is not None:
        union.seed_prior(facility.documented_grid)

    observed = np.zeros((facility.n_rows, facility.n_cols), dtype=bool)
    for m in squad:
        own = m.grid.own_observations()
        union.L += own
        observed |= (own != 0.0)
    np.clip(union.L, -config.LOG_ODDS_CLAMP, config.LOG_ODDS_CLAMP,
            out=union.L)

    coverage = union.coverage_fraction(facility) * 100.0
    _prec, _rec, f1 = union.surface_scores(facility)

    cls = union.classified()
    decided = observed & (cls >= 0)
    if decided.any():
        truth = (facility.grid == 1).astype(np.int8)
        err = float((cls[decided] != truth[decided]).sum() / decided.sum())
    else:
        err = 0.0
    return coverage, f1, err * 100.0


def contact_fraction(trace, squad_size, range_m=None):
    """
    Fraction of the mission a robot could hear at least one team-mate,
    averaged over the squad.

    THE NUMBER THAT DECIDES WHETHER RECOVERY CAN WORK AT ALL. Quarantine
    requires two robots to corroborate an accusation, and two robots that
    are never in contact cannot corroborate anything. At 25 m this sat
    between 13 % and 64 %, which is why the comms range had to be derived
    rather than assumed.

    Measured on TRUE positions, from the trajectory trace. Radio range is
    physics, so this is the simulator's view and not any robot's -- no
    robot could compute it, and none is shown it.
    """
    if range_m is None:
        range_m = config.COMMS_RANGE_M

    arr = trace.to_arrays()
    ids, steps = arr["id"], arr["step"]
    per_robot = {}
    for rid in range(squad_size):
        sel = ids == rid
        per_robot[rid] = (steps[sel], arr["x"][sel], arr["y"][sel])

    n = min(len(v[0]) for v in per_robot.values()) if per_robot else 0
    if n == 0:
        return 0.0

    in_contact = np.zeros((squad_size, n), dtype=bool)
    for a in range(squad_size):
        for b in range(squad_size):
            if a == b:
                continue
            d = np.hypot(per_robot[a][1][:n] - per_robot[b][1][:n],
                         per_robot[a][2][:n] - per_robot[b][2][:n])
            in_contact[a] |= (d <= range_m)
    return float(in_contact.mean())


def report(facility, squad, points, deviations, history, trace, stats,
           baseline=None):
    m = squad_metrics(facility, squad, points)
    det = detection_summary(deviations)
    duration = stats["steps"] * config.DT_S

    print("\n" + "=" * 70)
    print(f"  INSPECTION ROUND -- SQUAD OF {len(squad)}")
    print("=" * 70)
    print("  Every robot holds its own map. Nothing is shared except by")
    print("  radio, and only inside " f"{config.COMMS_RANGE_M:.0f} m.")
    print("-" * 70)
    print(f"  Inspection points visited  : {m['visited']:3d} / {m['total']} "
          f"({100*m['visited']/m['total']:.1f} %)")
    print(f"  Cut off by a deviation     : {m['unreachable']:3d}")
    print(f"  Missed but reachable       : {m['missed']:3d}")
    print(f"  MISSION SUCCESS            : {'YES' if m['success'] else 'NO':>3s}")
    print(f"  Ended because              : {stats['ended']}")
    print(f"  Mission duration           : {duration:6.0f} s")
    print("-" * 70)

    print(f"  {'robot':<7s} {'status':<10s} {'points':>7s} {'dist_m':>9s} "
          f"{'energy_J':>10s} {'batt_%':>7s} {'comms_J':>8s}")
    for sm in squad:
        r = sm.robot
        n = sum(1 for p in points if p.visited_by == sm.id)
        print(f"  {sm.id:<7d} {r.status():<10s} {n:7d} "
              f"{r.distance_travelled_m:9.1f} {r.total_energy_j:10.1f} "
              f"{r.battery_fraction*100:7.1f} {r.energy['comms']:8.2f}")
    print(f"  {'TOTAL':<7s} {'':<10s} {m['visited']:7d} "
          f"{m['distance']:9.1f} {m['energy']:10.1f}")
    print("-" * 70)

    grouped = points_by_zone(points)
    per_zone = [f"{code} {sum(1 for p in grouped[code] if p.visited)}"
                f"/{len(grouped[code])}"
                for code in sorted(grouped, key=lambda c: int(c[1:]))]
    print("  Per zone : " + "   ".join(per_zone))

    if stats.get("faults"):
        print("-" * 70)
        print("  FAULTS INJECTED")
        for robot_id, at_step, name in stats["faults"]:
            print(f"    robot {robot_id}: {name:<20s} at "
                  f"{at_step * config.DT_S:6.0f} s")
        print("    Nothing detected them. Detection is Step 4; this run only")
        print("    shows the damage.")

    print("-" * 70)
    print("  COORDINATION")
    radio = stats["radio"]
    print(f"    Messages broadcast       : {radio['sent']:6d}")
    print(f"    Deliveries               : {radio['delivered']:6d}")
    print(f"    Dropped, out of range    : {radio['lost_to_range']:6d}")
    print(f"    Dropped, interference    : {radio['lost_to_noise']:6d}")
    print(f"    Map merges accepted      : {stats['merges']:6d}")
    print(f"    Ground seen by >1 robot  : {m['area_twice_m2']:6.0f} m2 of "
          f"{m['area_seen_m2']:.0f} m2  ({m['redundancy']*100:.1f} % duplicated)")

    if stats["use_prior"]:
        print("-" * 70)
        print("  DEVIATIONS FROM THE DRAWINGS")
        for i, d in enumerate(deviations):
            when = (f"{d.detected_step * config.DT_S:6.0f}s"
                    if d.detected_step is not None else "     --")
            print(f"  {i:<3d} {d.kind:<8s} {d.zone_code:<5s} {d.name:<30s} "
                  f"{'YES' if d.detected else 'no':>6s} {when:>7s}")
        print(f"  >> DEVIATION DETECTION RATE : {det['detected']} / "
              f"{det['total']}  ({det['rate']*100:.0f} %)")

    print("-" * 70)
    print(f"  >> ENERGY PER POINT VISITED : {m['per_point']:6.0f} J")
    print(f"  Trajectory samples logged  : {len(trace):6d}")

    if baseline is not None:
        print("-" * 70)
        print("  AGAINST ONE ROBOT DOING THE SAME MISSION")
        print(f"  {'':<26s} {'1 robot':>10s} {len(squad):>7d} robots {'change':>10s}")
        rows = [
            ("points visited", baseline["visited"], m["visited"], "{:.0f}"),
            ("mission duration (s)", baseline["duration"], duration, "{:.0f}"),
            ("distance, all robots (m)", baseline["distance"], m["distance"], "{:.1f}"),
            ("total energy (J)", baseline["energy"], m["energy"], "{:.0f}"),
            ("energy per point (J)", baseline["per_point"], m["per_point"], "{:.0f}"),
        ]
        for name, one, many, fmt in rows:
            pct = (many - one) / one * 100 if one else float("nan")
            print(f"  {name:<26s} {fmt.format(one):>10s} {fmt.format(many):>14s} "
                  f"{pct:+9.1f} %")
        print(f"  {'deviations detected':<26s} "
              f"{baseline['detected']:>10d} {det['detected']:>14d}")
    print("=" * 70 + "\n")

    _figure(facility, squad, points, deviations, history, m)


def _figure(facility, squad, points, deviations, history, m):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ext = [0, facility.width_m, 0, facility.height_m]

    ax = axes[0, 0]
    ax.imshow(facility.render_rgb(), origin="lower", extent=ext,
              interpolation="nearest")
    for sm in squad:
        ax.plot(sm.trail_x, sm.trail_y, lw=1.0, alpha=0.85,
                color=ROBOT_COLOURS[sm.id % len(ROBOT_COLOURS)],
                label=f"robot {sm.id}")
    for p in points:
        if p.visited:
            ax.scatter([p.x], [p.y], s=30, zorder=5, marker="o",
                       c=ROBOT_COLOURS[(p.visited_by or 0) % len(ROBOT_COLOURS)],
                       edgecolors="white", linewidths=0.7)
        else:
            ax.scatter([p.x], [p.y], s=52, zorder=6, marker="X", c="#D62728",
                       edgecolors="white", linewidths=0.7)
    for d in deviations:
        x0, y0, x1, y1 = d.rect
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               ec="#000000", lw=1.4, ls="--"))
    ax.set_title("Three routes, and who inspected what")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[0, 1]
    cls = squad[0].grid.classified().astype(float)
    cls[cls == -1] = 0.5
    ax.imshow(1 - cls, cmap="gray", origin="lower", extent=ext, vmin=0, vmax=1)
    ax.set_title("Robot 0's PRIVATE map, after merging what it was told")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    # Who saw what, from each robot's own observations. Overlap is
    # duplicated effort; the auction exists to keep it small.
    ax = axes[1, 0]
    counts = np.zeros((facility.n_rows, facility.n_cols), dtype=np.uint8)
    for sm in squad:
        counts += sm.observed_mask().astype(np.uint8)
    shade = np.zeros((facility.n_rows, facility.n_cols, 3), dtype=np.float32)
    shade[:] = (0.97, 0.96, 0.94)
    shade[counts == 1] = (0.72, 0.83, 0.72)
    shade[counts == 2] = (0.95, 0.72, 0.35)
    shade[counts >= 3] = (0.80, 0.20, 0.20)
    shade[facility.grid == 1] = (0.35, 0.35, 0.38)
    ax.imshow(shade, origin="lower", extent=ext, interpolation="nearest")
    ax.legend(handles=[
        Patch(facecolor=(0.72, 0.83, 0.72), label="seen by 1 robot"),
        Patch(facecolor=(0.95, 0.72, 0.35), label="seen by 2"),
        Patch(facecolor=(0.80, 0.20, 0.20), label="seen by 3")],
        loc="upper right", fontsize=8)
    ax.set_title(f"Duplicated ground: {m['redundancy']*100:.1f} % of what "
                 "was seen")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    ax = axes[1, 1]
    ax.plot(history["t"], history["visited"], lw=2, color="#1D9E75",
            label="points visited (squad)")
    ax.plot(history["t"], history["detected"], lw=2, color="#C1442E",
            label="deviations detected")
    ax.axhline(len(points), ls=":", lw=1, color="0.4")
    ax.set_xlabel("time (s)"); ax.set_ylabel("count")
    ax.set_title("Mission progress")
    ax.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("demo_squad_result.png", dpi=125)
    print("Saved demo_squad_result.png")


def single_robot_baseline(seed, use_prior):
    """Run the same mission with one robot, for the comparison block."""
    import demo_inspect
    out = demo_inspect.run(seed, verbose=False, use_prior=use_prior)
    _facility, _grid, robot, pts, devs, _hist, _trail, st = out
    visited = sum(1 for p in pts if p.visited)
    return {"visited": visited,
            "duration": st["steps"] * config.DT_S,
            "distance": robot.distance_travelled_m,
            "energy": robot.total_energy_j,
            "per_point": robot.total_energy_j / max(visited, 1),
            "detected": sum(1 for d in devs if d.detected)}


if __name__ == "__main__":
    import sys

    use_prior = "--no-prior" not in sys.argv
    use_auction = "--no-auction" not in sys.argv
    out = run(use_prior=use_prior, use_auction=use_auction)
    base = None
    if "--compare" in sys.argv:
        print("\nRunning the same mission with a single robot for comparison...")
        base = single_robot_baseline(config.DEFAULT_SEED, use_prior)
    report(*out, baseline=base)

    path = out[5].save("squad_trace.npz")
    print(f"Saved {path} ({len(out[5])} samples)")
