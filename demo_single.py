"""
demo_single.py
==============
MILESTONE 1: one robot drives through the facility and builds a map.

Run it with:      python demo_single.py

This is deliberately the simplest possible complete loop:
    sense -> map -> decide -> move -> account for energy
Everything else in the project (multiple robots, faults, detection,
recovery) is added on top of this loop. If you understand this file, you
understand the architecture.

NOTE ON THE CONTROLLER:
  The robot follows a fixed waypoint tour with reactive obstacle
  avoidance. This is a scaffold, NOT the final system -- the real system
  uses frontier-based exploration so the robot decides for itself where to
  go. The waypoint version exists so you can verify sensing, mapping and
  energy accounting are correct before adding autonomy on top.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # save to file instead of opening a window
import matplotlib.pyplot as plt

import config
from environment import Facility
from sensors import Lidar2D
from mapping import OccupancyGrid
from robot import Robot, wrap_angle
from planner import WavefrontPlanner


# ---------------------------------------------------------------------
# A tour that visits every major aisle of the plant, including the
# enclosed control room (which the robot must find the door of).
# ---------------------------------------------------------------------
# --------------------------------------------------------------------
# IMPORTANT LESSON, WRITE THIS ONE DOWN:
# A purely reactive controller cannot escape a local minimum. If a straight
# line between two consecutive waypoints crosses a solid barrier, the robot
# will push against it forever. This tour is therefore built around the
# plant's actual topology:
#
#   Full-width corridors (safe to traverse end to end):  y = 2, 9.5, 13, 30.5, 38
#   The middle band (y 15-26) is SPLIT IN TWO by the vertical pipe rack at
#   x = 29.5, so the left and right halves are toured separately and joined
#   via the y = 13 and y = 27 corridors.
#
# The real system replaces this with A* path planning over the robot's own
# occupancy map, driven by frontier selection. That is Module 2.
# --------------------------------------------------------------------
WAYPOINTS = [
    # --- perimeter road (major road, 6 m) ---
    (3.5, 3.5),   (76.5, 3.5),  (76.5, 53.5), (3.5, 53.5),  (3.5, 3.5),
    # --- south of the main pipe rack ---
    (3.5, 23.0),  (76.0, 23.0),
    # --- north of the main pipe rack ---
    (76.0, 28.5), (3.5, 28.5),
    # --- process unit aisles ---
    (16.0, 34.5), (40.0, 34.5),
    (40.0, 40.5), (16.0, 40.5),
    (22.0, 47.0), (42.0, 47.0),
    # --- compressor house, entered by the south door ---
    (55.5, 31.0), (55.5, 34.5), (47.5, 34.5), (47.5, 43.5),
    (67.5, 43.5), (67.5, 34.5), (55.5, 34.5), (55.5, 31.0),
    # --- tank farm, entered by the single bund gap ---
    (41.0, 14.0), (46.5, 14.0), (46.5, 18.5), (71.2, 18.5),
    (71.2, 9.0),  (46.5, 9.0),  (46.5, 14.0), (41.0, 14.0),
    # --- utility area ---
    (5.0, 13.5),  (27.0, 13.5), (27.0, 16.5),
    (33.7, 16.5), (33.7, 11.0), (33.7, 17.0),
    # --- store and control building ---
    (13.2, 43.0), (13.2, 49.0), (13.2, 43.0),
    (15.5, 36.0), (10.0, 36.0),
]

WAYPOINT_TOLERANCE_M = 0.9
SCAN_EVERY_N_STEPS = 3          # LiDAR runs at ~3 Hz, control at 10 Hz
MAX_STEPS = 40000
MAX_STEPS_PER_WAYPOINT = 4000   # hard ceiling per waypoint -- see note below
NO_PROGRESS_STEPS = 600         # give up once the robot stops closing on the goal

# WHY THE CEILING IS GENEROUS, AND WHY THIS ONCE BROKE THE WHOLE DEMO
# -------------------------------------------------------------------
# This was 1200 when the facility was 60 x 40 m. The site is now 80 x 55 m,
# and the longest leg of the tour is 73 m -- which needs 1217 steps at
# maximum speed assuming the robot never turns and never slows for an
# obstacle. The budget was therefore smaller than the shortest possible
# traversal, so waypoint 1 was unreachable BY ARITHMETIC, and every
# subsequent waypoint inherited the deficit until the tour collapsed.
#
# The lesson: a timeout expressed in steps is coupled to the size of the
# map. NO_PROGRESS_STEPS is the real stuck detector -- it fires when the
# robot stops closing on its goal, which is what "stuck" actually means.
# The ceiling below is only a backstop, so it should be loose.
REPLAN_EVERY_N_STEPS = 120      # rerun the planner as the map improves
PATH_NODE_TOLERANCE_M = 0.45    # how close counts as reaching a path node
                                # loose tolerance lets the robot cut corners
                                # off the planned route and clip obstacles
PLAN_FAIL_COOLDOWN = 60         # steps to wait before retrying a failed plan




def choose_velocity(robot, ranges, angles, valid, goal_xy):
    """
    Reactive go-to-goal controller with obstacle repulsion.

    CRITICAL: this uses the robot's BELIEVED pose (bx, by, btheta), never
    its true pose. That is what makes the navigation honest.
    """
    gx, gy = goal_xy
    dx, dy = gx - robot.bx, gy - robot.by
    dist_to_goal = np.hypot(dx, dy)

    desired_heading = np.arctan2(dy, dx)
    heading_error = wrap_angle(desired_heading - robot.btheta)

    # --- baseline: turn toward the goal, drive faster when well aligned ---
    w = 1.6 * heading_error
    alignment = max(0.0, np.cos(heading_error))
    v = config.MAX_LINEAR_SPEED_MPS * alignment

    # --- reactive avoidance from the most recent scan ---
    if ranges is not None:
        rel = wrap_angle(angles - robot.btheta)

        # BRAKING CONE: deliberately NARROW (+/- 25 deg).
        # An earlier version used +/- 55 deg and the robot braked for the
        # side walls of every corridor, chattering and burning ~80% of its
        # energy on turning. Keep this cone tight.
        cone = np.abs(rel) < np.deg2rad(25)
        d_front = ranges[cone].min() if cone.any() else config.LIDAR_MAX_RANGE_M

        # SIDE CLEARANCE: proportional repulsion, not bang-bang steering.
        left = (rel > np.deg2rad(20)) & (rel < np.deg2rad(85))
        right = (rel < -np.deg2rad(20)) & (rel > -np.deg2rad(85))
        d_left = ranges[left].min() if left.any() else config.LIDAR_MAX_RANGE_M
        d_right = ranges[right].min() if right.any() else config.LIDAR_MAX_RANGE_M

        influence = 1.0        # metres: repulsion only inside this radius
        if d_left < influence:
            w -= 1.6 * (influence - d_left)
        if d_right < influence:
            w += 1.6 * (influence - d_right)

        if d_front < 1.2:
            v *= float(np.clip((d_front - 0.40) / 0.80, 0.0, 1.0))

        if d_front < 0.50:
            # genuinely blocked: stop and rotate toward the freer side
            v = 0.0
            w = 1.2 * (1.0 if d_left >= d_right else -1.0)

    return v, w, dist_to_goal


def run(seed=config.DEFAULT_SEED, verbose=True):
    rng = np.random.default_rng(seed)

    facility = Facility()
    facility.validate_waypoints(WAYPOINTS)   # fail loudly, not silently
    lidar = Lidar2D()
    grid = OccupancyGrid(facility, owner_id=0)
    planner = WavefrontPlanner(facility)

    # The tour goals are destinations, not a drivable route. The planner
    # works out how to get to each one using the robot's own map, and the
    # reactive controller then follows the planned nodes.
    path = []
    path_index = 0
    replan_countdown = 0
    plan_cooldown = 0

    start = WAYPOINTS[0]
    robot = Robot(robot_id=0, x=start[0], y=start[1], theta=0.0, rng=rng)

    wp_index = 1
    steps_on_waypoint = 0
    best_dist = float("inf")
    steps_since_progress = 0
    skipped = 0
    ranges = angles = valid = None

    history = {"t": [], "coverage": [], "energy": [], "pose_err": []}
    trail_x, trail_y = [robot.x], [robot.y]

    for step in range(MAX_STEPS):

        # ---------- 1. SENSE ----------
        if step % SCAN_EVERY_N_STEPS == 0 and robot.sensing:
            ranges, angles, valid = lidar.scan(
                facility, robot.x, robot.y, robot.theta, rng)
            robot.pay_sensing(config.DT_S * SCAN_EVERY_N_STEPS)

            # ---------- 2. MAP ----------
            # Integrated at the BELIEVED pose -- errors in odometry
            # therefore smear the map, exactly as in reality.
            grid.integrate_scan(robot.bx, robot.by, robot.btheta,
                                ranges, angles, valid, source_id=robot.id)

        # ---------- 3. DECIDE ----------
        goal = WAYPOINTS[wp_index]

        # Replan periodically: the robot's map improves as it explores, so
        # a route that looked open may turn out to be blocked, and vice versa.
        replan_countdown -= 1
        plan_cooldown -= 1
        need_plan = (not path) or (path_index >= len(path)) or (replan_countdown <= 0)
        if need_plan and plan_cooldown <= 0:
            new_path = planner.plan(grid, (robot.bx, robot.by), goal)
            replan_countdown = REPLAN_EVERY_N_STEPS
            if new_path:
                path = new_path
                path_index = 0
            else:
                # Planning failed. Do NOT ask again on the next step -- that
                # turns a 40 ms planner into a 400 ms-per-second tax and was
                # what made a 40 s run take over 15 minutes. Back off, steer
                # at the goal directly, and let the map improve first.
                plan_cooldown = PLAN_FAIL_COOLDOWN

        # Follow the current path node; fall back to the goal itself if
        # planning failed entirely.
        advanced_along_path = False
        if path and path_index < len(path):
            target = path[path_index]
            if np.hypot(target[0] - robot.bx, target[1] - robot.by) < PATH_NODE_TOLERANCE_M:
                path_index += 1
                advanced_along_path = True
                target = path[min(path_index, len(path) - 1)]
        else:
            target = goal

        v, w, _ = choose_velocity(robot, ranges, angles, valid, target)
        dist = float(np.hypot(goal[0] - robot.bx, goal[1] - robot.by))

        # --- progress-based stuck detection ---
        # A reactive controller cannot escape a local minimum. Rather than
        # let it grind against a wall for a fixed step budget, give up as
        # soon as it stops closing on the goal. This saves a great deal of
        # wasted energy and makes the skipped-waypoint count meaningful.
        steps_on_waypoint += 1
        # Progress means "advancing along the planned route", NOT "the
        # straight-line gap to the goal is shrinking". Those differ whenever
        # the robot correctly detours around a building -- the detour moves
        # it further from the goal in a straight line while bringing it
        # closer in travel distance. Judging by straight-line distance made
        # the detector abandon reachable waypoints mid-detour.
        if advanced_along_path or dist < best_dist - 0.05:
            best_dist = min(best_dist, dist)
            steps_since_progress = 0
        else:
            steps_since_progress += 1

        reached = dist < WAYPOINT_TOLERANCE_M
        stuck = (steps_since_progress > NO_PROGRESS_STEPS
                 or steps_on_waypoint > MAX_STEPS_PER_WAYPOINT)

        if reached or stuck:
            if stuck and not reached:
                skipped += 1
                if verbose:
                    print(f"  [unreachable] waypoint {wp_index} "
                          f"{WAYPOINTS[wp_index]} — no progress "
                          f"(closest approach {best_dist:.1f} m)")
            wp_index += 1
            steps_on_waypoint = 0
            best_dist = float("inf")
            steps_since_progress = 0
            path, path_index, replan_countdown = [], 0, 0
            if wp_index >= len(WAYPOINTS):
                if verbose:
                    print(f"Tour complete at step {step} "
                          f"(t = {step * config.DT_S:.0f} s), "
                          f"{skipped} waypoint(s) skipped")
                break

        # ---------- 4. MOVE + PAY ----------
        robot.step_motion(v, w, facility)

        if not robot.alive:
            if verbose:
                print(f"Battery exhausted at step {step}")
            break

        # ---------- 5. RECORD ----------
        if step % 25 == 0:
            history["t"].append(step * config.DT_S)
            history["coverage"].append(grid.coverage_fraction(facility) * 100)
            history["energy"].append(robot.total_energy_j)
            history["pose_err"].append(robot.pose_error_m())
            trail_x.append(robot.x)
            trail_y.append(robot.y)

    return facility, grid, robot, history, (trail_x, trail_y)


def report(facility, grid, robot, history, trail):
    cov = grid.coverage_fraction(facility) * 100
    iou = grid.occupied_iou(facility)
    prec, rec, f1 = grid.surface_scores(facility)
    err = grid.error_rate(facility) * 100
    area = facility.free_area_m2() * cov / 100

    print("\n" + "=" * 58)
    print("  SINGLE-ROBOT MAPPING RESULT")
    print("=" * 58)
    print(f"  Coverage of free space     : {cov:6.2f} %")
    print(f"  Surface precision          : {prec:6.3f}")
    print(f"  Surface recall             : {rec:6.3f}")
    print(f"  Surface F1  (map accuracy) : {f1:6.3f}")
    print(f"  Strict IoU (no tolerance)  : {iou:6.3f}")
    print(f"  Misclassification rate     : {err:6.2f} %")
    print(f"  Distance travelled         : {robot.distance_travelled_m:6.1f} m")
    print(f"  Collisions                 : {robot.collisions:6d}")
    print(f"  Final odometry error       : {robot.pose_error_m():6.2f} m")
    print("-" * 58)
    print("  ENERGY BREAKDOWN (joules)")
    for k, val in robot.energy.items():
        pct = 100 * val / robot.total_energy_j if robot.total_energy_j else 0
        print(f"    {k:<10s} {val:9.1f} J   ({pct:4.1f} %)")
    print(f"    {'TOTAL':<10s} {robot.total_energy_j:9.1f} J")
    print(f"  Battery remaining          : {robot.battery_fraction*100:5.1f} %")
    print("-" * 58)
    print(f"  >> ENERGY PER m2 MAPPED    : "
          f"{robot.total_energy_j / max(area, 1e-6):6.2f} J/m2")
    print("=" * 58 + "\n")

    # ---------------- figure ----------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    ext = [0, facility.width_m, 0, facility.height_m]

    ax = axes[0, 0]
    ax.imshow(facility.render_rgb(), origin="lower", extent=ext,
              interpolation="nearest")
    ax.plot(trail[0], trail[1], lw=1.2, color="tab:red", alpha=0.85)
    ax.set_title("Ground truth + robot path")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    ax = axes[0, 1]
    cls = grid.classified().astype(float)
    cls[cls == -1] = 0.5          # unknown -> mid grey
    ax.imshow(1 - cls, cmap="gray", origin="lower", extent=ext, vmin=0, vmax=1)
    ax.set_title("Map built by the robot (white=free, black=obstacle, grey=unknown)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    ax = axes[1, 0]
    ax.plot(history["t"], history["coverage"], lw=2, color="tab:blue")
    ax.set_xlabel("time (s)"); ax.set_ylabel("coverage of free space (%)")
    ax.set_title("Coverage over time"); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    cats = list(robot.energy.keys())
    vals = [robot.energy[c] for c in cats]
    ax.bar(cats, vals, color=["tab:blue", "tab:orange", "tab:green",
                              "tab:red", "tab:purple"])
    ax.set_ylabel("energy (J)")
    ax.set_title(f"Energy breakdown - total {robot.total_energy_j:.0f} J")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig("demo_single_result.png", dpi=125)
    print("Saved demo_single_result.png")


if __name__ == "__main__":
    out = run()
    report(*out)
