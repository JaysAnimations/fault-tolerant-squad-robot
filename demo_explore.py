"""
demo_explore.py
===============
STEP 1 COMPLETE: the robot explores the facility with no route given to it.

Compare this file with demo_single.py. That one contains a hand-written list
of 42 waypoints -- a human decided where the robot should go. This one
contains no route at all. The robot looks at its own map, finds the boundary
between mapped and unmapped space, and drives there. Repeat until no such
boundary remains.

That difference is the whole of Objective ii, and it is what makes the
system autonomous rather than scripted.

Run:  python demo_explore.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from environment import Facility
from sensors import Lidar2D
from mapping import OccupancyGrid
from robot import Robot
from planner import WavefrontPlanner
from frontier import FrontierExplorer
from control import choose_velocity

# --- start pose: just inside the gate on the perimeter road ---
START_XY = (3.5, 3.5)

SCAN_EVERY_N_STEPS = 3
MAX_STEPS = 40000
REPLAN_EVERY_N_STEPS = 120
PATH_NODE_TOLERANCE_M = 0.45
PLAN_FAIL_COOLDOWN = 60

# If the robot stops closing on its chosen frontier for this long, that
# frontier is banned for a while and another is chosen. Without it the robot
# re-picks the same unreachable frontier forever.
NO_PROGRESS_STEPS = 500
BAN_DURATION_STEPS = 4000

# Warm-up: spin on the spot so the first scans give the explorer some
# known-free space to find frontiers on. On a completely blank map there are
# no frontiers, because a frontier needs known space on one side.
WARMUP_STEPS = 40


def run(seed=config.DEFAULT_SEED, verbose=True):
    rng = np.random.default_rng(seed)

    facility = Facility()
    lidar = Lidar2D()
    grid = OccupancyGrid(facility, owner_id=0)
    planner = WavefrontPlanner(facility)
    explorer = FrontierExplorer(planner)

    robot = Robot(robot_id=0, x=START_XY[0], y=START_XY[1], theta=0.0, rng=rng)

    path, path_index, replan_countdown, plan_cooldown = [], 0, 0, 0
    best_dist, steps_since_progress = float("inf"), 0
    goals_taken, goals_abandoned = 0, 0

    ranges = angles = valid = None
    history = {"t": [], "coverage": [], "energy": [], "frontiers": []}
    trail_x, trail_y = [robot.x], [robot.y]
    goal_marks = []

    for step in range(MAX_STEPS):

        # ---------- 1. SENSE ----------
        if step % SCAN_EVERY_N_STEPS == 0 and robot.sensing:
            ranges, angles, valid = lidar.scan(
                facility, robot.x, robot.y, robot.theta, rng)
            robot.pay_sensing(config.DT_S * SCAN_EVERY_N_STEPS)
            grid.integrate_scan(robot.bx, robot.by, robot.btheta,
                                ranges, angles, valid, source_id=robot.id)

        # ---------- 2. WARM-UP ----------
        # Turn in place until the first scans have created enough known-free
        # space for a frontier to exist.
        if step < WARMUP_STEPS:
            robot.step_motion(0.0, config.MAX_ANGULAR_SPEED_RPS, facility)
            continue

        # ---------- 3. CHOOSE A FRONTIER ----------
        me = (robot.bx, robot.by)
        # Validity checking costs ~4 ms, so do it every few steps rather
        # than every step. The robot cannot outrun a frontier in 0.5 s.
        if step % config.FRONTIER_CHECK_EVERY == 0:
            need_goal = not explorer.goal_still_valid(grid, me, step=step)
        else:
            need_goal = explorer.goal_xy is None

        if steps_since_progress > NO_PROGRESS_STEPS:
            explorer.ban(explorer.goal_rc, step + BAN_DURATION_STEPS)
            goals_abandoned += 1
            need_goal = True

        if need_goal:
            goal = explorer.select(grid, me, step=step)
            if goal is None:
                if verbose:
                    print(f"Exploration complete at step {step} "
                          f"(t = {step * config.DT_S:.0f} s)")
                break
            goals_taken += 1
            goal_marks.append(goal)
            path, path_index = [], 0
            replan_countdown, plan_cooldown = 0, 0
            best_dist, steps_since_progress = float("inf"), 0

        goal = explorer.goal_xy

        # ---------- 4. PLAN ----------
        replan_countdown -= 1
        plan_cooldown -= 1
        need_plan = (not path) or (path_index >= len(path)) or (replan_countdown <= 0)
        if need_plan and plan_cooldown <= 0:
            new_path = planner.plan(grid, me, goal)
            replan_countdown = REPLAN_EVERY_N_STEPS
            if new_path:
                path, path_index = new_path, 0
            else:
                plan_cooldown = PLAN_FAIL_COOLDOWN

        advanced = False
        if path and path_index < len(path):
            target = path[path_index]
            if np.hypot(target[0] - robot.bx, target[1] - robot.by) < PATH_NODE_TOLERANCE_M:
                path_index += 1
                advanced = True
                target = path[min(path_index, len(path) - 1)]
        else:
            target = goal

        # ---------- 5. PROGRESS CHECK ----------
        dist = float(np.hypot(goal[0] - robot.bx, goal[1] - robot.by))
        if advanced or dist < best_dist - 0.05:
            best_dist = min(best_dist, dist)
            steps_since_progress = 0
        else:
            steps_since_progress += 1

        # ---------- 6. MOVE ----------
        v, w, _ = choose_velocity(robot, ranges, angles, target)
        robot.step_motion(v, w, facility)

        if not robot.alive:
            if verbose:
                print(f"Battery exhausted at step {step}")
            break

        # ---------- 7. RECORD ----------
        if step % 25 == 0:
            history["t"].append(step * config.DT_S)
            history["coverage"].append(grid.coverage_fraction(facility) * 100)
            history["energy"].append(robot.total_energy_j)
            trail_x.append(robot.x)
            trail_y.append(robot.y)

    stats = {"goals_taken": goals_taken, "goals_abandoned": goals_abandoned,
             "steps": step}
    return facility, grid, robot, history, (trail_x, trail_y), goal_marks, stats


def report(facility, grid, robot, history, trail, goal_marks, stats):
    cov = grid.coverage_fraction(facility) * 100
    prec, rec, f1 = grid.surface_scores(facility)
    area = facility.free_area_m2() * cov / 100

    print("\n" + "=" * 60)
    print("  AUTONOMOUS FRONTIER EXPLORATION -- SINGLE ROBOT")
    print("=" * 60)
    print("  No route was given to the robot. Every goal below was chosen")
    print("  by the robot from its own map.")
    print("-" * 60)
    print(f"  Frontiers visited          : {stats['goals_taken']:6d}")
    print(f"  Frontiers abandoned        : {stats['goals_abandoned']:6d}")
    print(f"  Coverage of free space     : {cov:6.2f} %")
    print(f"  Surface F1  (map accuracy) : {f1:6.3f}")
    print(f"  Distance travelled         : {robot.distance_travelled_m:6.1f} m")
    print(f"  Collisions                 : {robot.collisions:6d}")
    print(f"  Final odometry error       : {robot.pose_error_m():6.2f} m")
    print(f"  Mission duration           : {stats['steps']*config.DT_S:6.0f} s")
    print("-" * 60)
    for k, val in robot.energy.items():
        pct = 100 * val / robot.total_energy_j if robot.total_energy_j else 0
        print(f"    {k:<10s} {val:9.1f} J   ({pct:4.1f} %)")
    print(f"    {'TOTAL':<10s} {robot.total_energy_j:9.1f} J")
    print(f"  Battery remaining          : {robot.battery_fraction*100:5.1f} %")
    print("-" * 60)
    print(f"  >> ENERGY PER m2 MAPPED    : "
          f"{robot.total_energy_j / max(area, 1e-6):6.2f} J/m2")
    print("=" * 60 + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ext = [0, facility.width_m, 0, facility.height_m]

    ax = axes[0, 0]
    ax.imshow(facility.render_rgb(), origin="lower", extent=ext,
              interpolation="nearest")
    ax.plot(trail[0], trail[1], lw=1.1, color="#C1442E", alpha=0.9)
    if goal_marks:
        gm = np.array(goal_marks)
        ax.scatter(gm[:, 0], gm[:, 1], s=16, c="#1D4E89", marker="x",
                   linewidths=1.2, label="frontiers chosen")
        ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Path taken, and the frontiers the robot chose")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    ax = axes[0, 1]
    cls = grid.classified().astype(float)
    cls[cls == -1] = 0.5
    ax.imshow(1 - cls, cmap="gray", origin="lower", extent=ext, vmin=0, vmax=1)
    ax.set_title("Map the robot built (white=free, black=obstacle, grey=unknown)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    ax = axes[1, 0]
    ax.plot(history["t"], history["coverage"], lw=2, color="#1D4E89")
    ax.set_xlabel("time (s)"); ax.set_ylabel("coverage of free space (%)")
    ax.set_title("Coverage over time"); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    cats = list(robot.energy.keys())
    ax.bar(cats, [robot.energy[c] for c in cats],
           color=["#1D4E89", "#BA7517", "#1D9E75", "#C1442E", "#7A5EA8"])
    ax.set_ylabel("energy (J)")
    ax.set_title(f"Energy breakdown - total {robot.total_energy_j:.0f} J")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig("demo_explore_result.png", dpi=125)
    print("Saved demo_explore_result.png")


if __name__ == "__main__":
    report(*run())
