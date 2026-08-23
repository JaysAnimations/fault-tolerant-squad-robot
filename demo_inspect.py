"""
demo_inspect.py
===============
The mission Design Change 01 describes: one robot, issued the facility's
documented layout, drives an inspection round and reports what has changed
since the drawings were last revised.

Run:  python demo_inspect.py

HOW THIS DIFFERS FROM THE OTHER TWO DEMOS
-----------------------------------------
  demo_single.py   a human wrote a route. Retained as the baseline that
                   catches regressions in sensing, mapping and energy.
  demo_explore.py  no route and no map. This is now the EMERGENCY
                   RESPONSE scenario: post-incident, when the drawings
                   cannot be trusted and the robot must map from nothing.
  demo_inspect.py  a map but no route. The robot is given the drawings and
                   works out its own order of visits -- which is what a
                   routine inspection round actually looks like.

WHAT THE ROBOT KNOWS AND WHAT IT DOES NOT
-----------------------------------------
It is issued: the documented layout, seeded into its occupancy grid at
+/- 2.0 log-odds, and the list of inspection points.

It is NOT told: which deviations were injected, where they are, or which
points a deviation has cut off. It has to discover all of that. The
`unreachable` flag on an inspection point is written by the analysis code
and is never read by anything the robot does -- the robot has to work out
for itself that it cannot get somewhere, and give up, which is the
behaviour decision 3 asks for.

The robot navigates and maps from its BELIEVED pose throughout. The true
pose is used for exactly two things: the physics of driving, and the
LiDAR observing the real world. Both are the simulator's job, not the
robot's.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

import config
from environment import Facility
from sensors import Lidar2D
from mapping import OccupancyGrid
from robot import Robot
from planner import WavefrontPlanner
from control import choose_velocity
from inspection import (generate_inspection_points, reachability_planner,
                        points_by_zone)
from deviations import inject_deviations, update_detection, detection_summary


def choose_next_point(ranker, grid, points, robot_xy):
    """
    Pick the nearest unvisited inspection point, measured by TRAVEL
    distance over the robot's own map rather than in a straight line.

    One BFS flood from the robot gives the distance to all forty points at
    once. That is the same trick frontier.py uses to rank frontiers, and
    it is the reason this project plans with a wavefront rather than A*:
    ranking forty candidates costs one flood, not forty path searches.

    Nearest-first is greedy and does not produce the shortest possible
    tour -- that is the travelling salesman problem and is not what this
    demo is for. Route optimisation belongs with the multi-robot auction
    in Step 2, where the points get shared out between three robots.

    Returns (point, distance_in_coarse_cells), or (None, None) when every
    remaining point is unreachable on the robot's current belief.

    `ranker` inflates obstacles less than the navigation planner does --
    see config.REACHABILITY_INFLATE_CELLS. Ranking asks "how far is it",
    which must not exclude a room whose doorway is narrower than the
    navigation planner's safety margin.
    """
    pending = [p for p in points if not p.visited and not p.abandoned]
    if not pending:
        return None, None

    blocked = ranker._coarse_blocked(grid)
    sr, sc = ranker._to_coarse(*robot_xy)
    if not (0 <= sr < ranker.c_rows and 0 <= sc < ranker.c_cols):
        return None, None
    if blocked[sr, sc]:
        snapped = ranker._nearest_free(~blocked, sr, sc)
        if snapped is None:
            return None, None
        sr, sc = snapped
    dist = ranker._flood(blocked, [(sr, sc)])

    best, best_d = None, None
    for p in pending:
        r, c = ranker._to_coarse(p.x, p.y)
        if not (0 <= r < ranker.c_rows and 0 <= c < ranker.c_cols):
            continue
        d = int(dist[r, c])
        if d < 0:
            continue          # the robot's own map says there is no way in
        if best is None or d < best_d:
            best, best_d = p, d
    return best, best_d


def run(seed=config.DEFAULT_SEED, verbose=True):
    # The robot's own randomness (sensor noise, wheel slip) is drawn from a
    # stream of its own, so that switching deviations on or off does not
    # change the noise the robot experiences. See config section 8.
    rng = np.random.default_rng([seed, config.RNG_STREAM_ROBOT])

    facility = Facility()
    ranker = reachability_planner(facility)

    # --- build the mission -------------------------------------------
    points = generate_inspection_points(facility, seed, ranker)
    deviations = inject_deviations(facility, points, seed, ranker,
                                   verbose=verbose)

    # --- issue the robot its equipment and its drawings ---------------
    lidar = Lidar2D()
    grid = OccupancyGrid(facility, owner_id=0)
    grid.seed_prior(facility.documented_grid)
    planner = WavefrontPlanner(facility)

    sx, sy = config.START_POSE_XY
    robot = Robot(robot_id=0, x=sx, y=sy, theta=config.START_THETA_RAD, rng=rng)

    target = None
    path, path_index = [], 0
    replan_countdown, plan_cooldown = 0, 0
    best_dist, steps_since_progress, steps_on_point = float("inf"), 0, 0

    ranges = angles = valid = None
    history = {"t": [], "visited": [], "energy": [], "detected": []}
    trail_x, trail_y = [robot.x], [robot.y]
    ended = "step limit reached"
    step = 0

    for step in range(config.INSPECTION_MAX_STEPS):

        # ---------- 1. SENSE ----------
        if step % config.INSPECTION_SCAN_EVERY_N_STEPS == 0 and robot.sensing:
            ranges, angles, valid = lidar.scan(
                facility, robot.x, robot.y, robot.theta, rng)
            robot.pay_sensing(config.DT_S * config.INSPECTION_SCAN_EVERY_N_STEPS)

            # ---------- 2. MAP ----------
            # Folded in at the BELIEVED pose, on top of the prior. Where a
            # deviation contradicts the drawings, this is what overturns
            # them -- about 4 observations for a new obstacle, about 8 for
            # a removed one.
            grid.integrate_scan(robot.bx, robot.by, robot.btheta,
                                ranges, angles, valid, source_id=robot.id)

        me = (robot.bx, robot.by)

        # ---------- 3. CHOOSE A POINT ----------
        if target is None:
            target, _ = choose_next_point(ranker, grid, points, me)
            if target is None:
                # Nothing left that the robot believes it can get to.
                # Everything still pending is declared unreachable -- which
                # is the robot's own conclusion from its own map, not
                # something it was told.
                for p in points:
                    if not p.visited:
                        p.abandoned = True
                ended = "all points visited or given up on"
                if verbose:
                    print(f"Round complete at step {step} "
                          f"(t = {step * config.DT_S:.0f} s)")
                break
            path, path_index = [], 0
            replan_countdown, plan_cooldown = 0, 0
            best_dist, steps_since_progress, steps_on_point = float("inf"), 0, 0

        goal = target.xy

        # ---------- 4. PLAN ----------
        replan_countdown -= 1
        plan_cooldown -= 1
        need_plan = (not path) or (path_index >= len(path)) or (replan_countdown <= 0)
        if need_plan and plan_cooldown <= 0:
            new_path = planner.plan(grid, me, goal)
            if new_path is None:
                # No route with the full 0.8 m safety margin. Before giving
                # up, look for one with the minimum margin.
                #
                # WHY THIS IS NEEDED AND WHAT IT COST TO LEARN: the control
                # building has a single 2.0 m doorway. Inflating obstacles
                # by 0.8 m on each side leaves 0.4 m, which the coarse grid
                # rounds away, so the wide-margin planner believes the room
                # is sealed and returns None. The robot then steered
                # straight at the goal, pushed against the outside wall for
                # 500 steps, and gave up -- losing an entire zone and
                # generating most of the run's collisions.
                #
                # A 2.0 m doorway is passable by a 0.5 m robot. The margin
                # is a preference, not a law, so when the comfortable route
                # does not exist we take the tight one.
                new_path = ranker.plan(grid, me, goal)
            replan_countdown = config.INSPECTION_REPLAN_EVERY_N_STEPS
            if new_path:
                path, path_index = new_path, 0
            else:
                plan_cooldown = config.INSPECTION_PLAN_FAIL_COOLDOWN

        advanced = False
        if path and path_index < len(path):
            node = path[path_index]
            if np.hypot(node[0] - robot.bx, node[1] - robot.by) < \
                    config.INSPECTION_PATH_NODE_TOLERANCE_M:
                path_index += 1
                advanced = True
                node = path[min(path_index, len(path) - 1)]
        else:
            node = goal

        # ---------- 5. PROGRESS ----------
        # Progress means advancing along the planned route, not shrinking
        # the straight-line gap. Those differ whenever the robot correctly
        # detours around a building -- or around a deviation.
        dist = float(np.hypot(goal[0] - robot.bx, goal[1] - robot.by))
        steps_on_point += 1
        if advanced or dist < best_dist - 0.05:
            best_dist = min(best_dist, dist)
            steps_since_progress = 0
        else:
            steps_since_progress += 1

        if dist < config.INSPECTION_REACHED_M:
            target.visited = True
            target.visit_step = step
            # Analysis only: how far the robot really was from the point it
            # believed it had reached. Never used for any decision.
            target.visit_error_m = float(np.hypot(target.x - robot.x,
                                                  target.y - robot.y))
            target = None
        elif (steps_since_progress > config.INSPECTION_NO_PROGRESS_STEPS
              or steps_on_point > config.INSPECTION_MAX_STEPS_PER_POINT):
            target.abandoned = True
            if verbose:
                print(f"  [gave up] point {target.index} "
                      f"({target.x:.1f}, {target.y:.1f}) in {target.zone_code} "
                      f"— closest approach {best_dist:.1f} m")
            target = None

        # ---------- 6. MOVE ----------
        v, w, _ = choose_velocity(robot, ranges, angles, node)
        robot.step_motion(v, w, facility)

        if not robot.alive:
            ended = "energy exhausted"
            if verbose:
                print(f"Battery exhausted at step {step}")
            break

        # ---------- 7. REPORT DEVIATIONS ----------
        if step % config.DEVIATION_CHECK_EVERY_STEPS == 0:
            update_detection(deviations, grid, step)

        # ---------- 8. RECORD ----------
        if step % 25 == 0:
            history["t"].append(step * config.DT_S)
            history["visited"].append(sum(1 for p in points if p.visited))
            history["energy"].append(robot.total_energy_j)
            history["detected"].append(sum(1 for d in deviations if d.detected))

            trail_x.append(robot.x)
            trail_y.append(robot.y)

    update_detection(deviations, grid, step)

    stats = {"steps": step, "ended": ended}
    return (facility, grid, robot, points, deviations, history,
            (trail_x, trail_y), stats)


def report(facility, grid, robot, points, deviations, history, trail, stats):
    visited = [p for p in points if p.visited]
    unreachable = [p for p in points if p.unreachable]
    missed = [p for p in points if not p.visited]
    # A mission succeeds if every point that COULD be reached was reached.
    # A point a deviation cut off is reported separately, because failing
    # to visit it is not a failure of the squad.
    reachable_missed = [p for p in missed if not p.unreachable]
    success = len(reachable_missed) == 0

    det = detection_summary(deviations)
    errors = [p.visit_error_m for p in visited if p.visit_error_m is not None]

    print("\n" + "=" * 66)
    print("  INSPECTION ROUND WITH A PRIOR MAP -- SINGLE ROBOT")
    print("=" * 66)
    print("  The robot was issued the documented layout and the list of")
    print("  points. It was NOT told where the deviations are.")
    print("-" * 66)
    print(f"  Inspection points visited  : {len(visited):3d} / {len(points)} "
          f"({100*len(visited)/len(points):.1f} %)")
    print(f"  Cut off by a deviation     : {len(unreachable):3d}"
          + (f"   -> points {[p.index for p in unreachable]}" if unreachable else ""))
    print(f"  Missed but reachable       : {len(reachable_missed):3d}"
          + (f"   -> points {[p.index for p in reachable_missed]}"
             if reachable_missed else ""))
    print(f"  MISSION SUCCESS            : {'YES' if success else 'NO':>3s}"
          "   (all reachable points visited)")
    print(f"  Ended because              : {stats['ended']}")
    print("-" * 66)

    grouped = points_by_zone(points)
    per_zone = [f"{code} {sum(1 for p in grouped[code] if p.visited)}"
                f"/{len(grouped[code])}"
                for code in sorted(grouped, key=lambda c: int(c[1:]))]
    print("  Per zone : " + "   ".join(per_zone))

    print("-" * 66)
    print("  DEVIATIONS FROM THE DRAWINGS")
    print(f"  {'#':<3s} {'type':<8s} {'zone':<5s} {'where':<30s} "
          f"{'found':>6s} {'at':>7s} {'overturned':>11s}")
    for i, d in enumerate(deviations):
        when = (f"{d.detected_step * config.DT_S:6.0f}s"
                if d.detected_step is not None else "     --")
        # "overturned" is the peak fraction of the deviation's evidence
        # cells the robot's map contradicted. It says whether a miss was a
        # near miss or a total one, which is the difference between a
        # threshold that needs tuning and a robot that never went there.
        print(f"  {i:<3d} {d.kind:<8s} {d.zone_code:<5s} {d.name:<30s} "
              f"{'YES' if d.detected else 'no':>6s} {when:>7s} "
              f"{d.best_fraction*100:10.0f} %")
    print(f"  >> DEVIATION DETECTION RATE : {det['detected']} / {det['total']}"
          f"  ({det['rate']*100:.0f} %)")
    for kind in sorted(det["by_kind"]):
        k = det["by_kind"][kind]
        mean_t = (np.mean(k["steps"]) * config.DT_S if k["steps"] else float("nan"))
        print(f"       {kind:<8s} {k['detected']}/{k['total']} detected, "
              f"mean time to detect {mean_t:6.0f} s")

    print("-" * 66)
    print(f"  Distance travelled         : {robot.distance_travelled_m:6.1f} m")
    print(f"  Collisions                 : {robot.collisions:6d}")
    print(f"  Final odometry error       : {robot.pose_error_m():6.2f} m")
    print(f"  Mean error on arrival      : "
          f"{np.mean(errors) if errors else float('nan'):6.2f} m"
          "   (true distance to a point the robot believed it had reached)")
    print(f"  Mission duration           : {stats['steps']*config.DT_S:6.0f} s")
    print("-" * 66)
    for k, val in robot.energy.items():
        pct = 100 * val / robot.total_energy_j if robot.total_energy_j else 0
        print(f"    {k:<10s} {val:9.1f} J   ({pct:4.1f} %)")
    print(f"    {'TOTAL':<10s} {robot.total_energy_j:9.1f} J")
    print(f"  Battery remaining          : {robot.battery_fraction*100:5.1f} %")
    print(f"  >> ENERGY PER POINT VISITED : "
          f"{robot.total_energy_j / max(len(visited), 1):6.0f} J")
    print("=" * 66 + "\n")

    # ---------------- figure ----------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ext = [0, facility.width_m, 0, facility.height_m]

    ax = axes[0, 0]
    ax.imshow(facility.render_rgb(), origin="lower", extent=ext,
              interpolation="nearest")
    ax.plot(trail[0], trail[1], lw=1.0, color="#C1442E", alpha=0.85)
    vx = [p.x for p in points if p.visited]
    vy = [p.y for p in points if p.visited]
    mx = [p.x for p in points if not p.visited]
    my = [p.y for p in points if not p.visited]
    ax.scatter(vx, vy, s=34, c="#1D9E75", marker="o", edgecolors="white",
               linewidths=0.8, zorder=5, label=f"visited ({len(vx)})")
    if mx:
        ax.scatter(mx, my, s=52, c="#D62728", marker="X", edgecolors="white",
                   linewidths=0.8, zorder=6, label=f"not visited ({len(mx)})")
    for d in deviations:
        x0, y0, x1, y1 = d.rect
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               ec="#000000", lw=1.6, ls="--"))
    ax.set_title("Route driven, and the points reached")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[0, 1]
    cls = grid.classified().astype(float)
    cls[cls == -1] = 0.5
    ax.imshow(1 - cls, cmap="gray", origin="lower", extent=ext, vmin=0, vmax=1)
    ax.set_title("The robot's map at the end of the round")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    # Where the robot's own map now contradicts the drawings it was issued.
    # This panel is the deviation report, drawn straight from the robot's
    # belief -- no ground truth is consulted to produce it.
    ax = axes[1, 0]
    contra = np.zeros((facility.n_rows, facility.n_cols, 3), dtype=np.float32)
    contra[:] = (0.97, 0.96, 0.94)
    contra[facility.documented_grid == 1] = (0.80, 0.80, 0.80)
    # Drawn at zero tolerance so the panel shows the cells the robot
    # actually overturned. The detection METRIC allows +/- 2 cells, for the
    # pose-error reason given in OccupancyGrid.contradicts_prior.
    now_solid, now_open = grid.contradicts_prior(tolerance_cells=0)
    contra[now_solid] = (0.86, 0.15, 0.15)
    contra[now_open] = (0.15, 0.35, 0.80)
    ax.imshow(contra, origin="lower", extent=ext, interpolation="nearest")
    for d in deviations:
        x0, y0, x1, y1 = d.rect
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               ec="#1D9E75" if d.detected else "#000000",
                               lw=1.8))
    ax.legend(handles=[
        Patch(facecolor=(0.86, 0.15, 0.15), label="drawn open, observed solid"),
        Patch(facecolor=(0.15, 0.35, 0.80), label="drawn solid, observed open"),
        Patch(facecolor="none", edgecolor="#1D9E75", label="deviation, detected"),
        Patch(facecolor="none", edgecolor="#000000", label="deviation, missed")],
        loc="upper right", fontsize=7)
    ax.set_title("Where the robot's map contradicts the drawings")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    ax = axes[1, 1]
    ax.plot(history["t"], history["visited"], lw=2, color="#1D9E75",
            label="inspection points visited")
    ax.plot(history["t"], history["detected"], lw=2, color="#C1442E",
            label="deviations detected")
    ax.axhline(len(points), ls=":", lw=1, color="0.4")
    ax.set_xlabel("time (s)"); ax.set_ylabel("count")
    ax.set_title("Mission progress")
    ax.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("demo_inspect_result.png", dpi=125)
    print("Saved demo_inspect_result.png")


if __name__ == "__main__":
    report(*run())
