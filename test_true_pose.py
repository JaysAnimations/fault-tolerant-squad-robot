"""
test_true_pose.py
=================
Assert that navigation and mapping use the BELIEVED pose and never the
true one.

    py test_true_pose.py

WHY THIS EXISTS. CLAUDE.md calls "believed pose only" a correctness
requirement rather than a style preference: if control or mapping ever
reads robot.x, the robot is navigating by ground truth and the SLAM claim
collapses. Until now nothing checked it. The failure is silent -- every
demo would still run, the maps would simply be better than odometry drift
permits, and nobody would notice until a panel asked.

WHAT IS AND IS NOT A VIOLATION. The LiDAR scan is taken FROM the true
pose, and that is correct: a real sensor observes the world from where it
physically is. What must never happen is the result being integrated at
that pose, or the controller and planner steering by it. So this file
checks the three places where the believed pose has to be used, and leaves
the sensor alone.

    1. control.choose_velocity     -- blind to the true pose
    2. grid.integrate_scan         -- handed the believed pose
    3. planner.plan                -- started from the believed pose

TWO OF THE THREE ARE CHECKED ON A LIVE MISSION rather than in isolation,
because integrate_scan and plan both take plain numbers. Calling them
directly with the believed pose would prove nothing -- the risk is not in
those functions, it is in what squad.py hands them. So the mission is run
with both call sites traced, and what is checked is the argument that
actually arrived.
"""
import sys

import numpy as np

import config
import control
import demo_squad
import squad as squad_module
from environment import Facility
from robot import Robot
from sensors import Lidar2D

# Far enough that no rounding could hide it, and not a round number, so a
# stray copy of it in output is recognisable.
POISON_M = 13.7

# Stop the traced mission once it has answered the question rather than
# running the full ~25 s of it, so nobody skips this test for being slow.
#
# THE STOPPING RULE COUNTS DRIFTED SAMPLES PER CALL SITE, NOT SAMPLES. A
# plain sample count stops far too early for the planner: sensing happens
# every few steps and replanning every few hundred, so 200 samples is 322
# scans and only 10 plans -- none of them late enough for odometry to have
# separated the two poses. Stopping on the count that the check actually
# uses is what makes the run long enough to prove something.
MIN_DRIFT_M = 0.05          # comfortably above float noise; healthy drift
                            # reaches 0.18 m mean over a full mission
DRIFTED_PER_SITE = 20       # before the mission is cut short


class _Enough(Exception):
    """Not an error -- the tracer has all the samples it needs."""


# ---------------------------------------------------------------------
# 1. The controller must be blind to the true pose
# ---------------------------------------------------------------------
def check_controller():
    """
    Move the true pose a long way and require the command not to change;
    then move the believed pose and require that it DOES.

    The second half is the half that matters. Without it this would pass
    just as happily on a controller that ignored both poses, which is not
    the property being claimed.
    """
    facility = Facility()
    rng = np.random.default_rng(0)
    x, y, theta = config.SQUAD_START_POSES[0]
    robot = Robot(0, x, y, theta, rng)
    lidar = Lidar2D()

    ranges, angles, _valid = lidar.scan(facility, robot.x, robot.y,
                                        robot.theta, rng)
    goal = (robot.bx + 8.0, robot.by + 3.0)

    baseline = control.choose_velocity(robot, ranges, angles, goal)

    # --- displace the TRUE pose: the command must not move --------------
    robot.x += POISON_M
    robot.y -= POISON_M
    robot.theta = -robot.theta
    poisoned = control.choose_velocity(robot, ranges, angles, goal)
    robot.x, robot.y, robot.theta = x, y, theta

    if poisoned != baseline:
        return [f"choose_velocity changed when only the TRUE pose moved: "
                f"{baseline} -> {poisoned}. It is steering by ground truth."]

    # --- displace the BELIEVED pose: the command MUST move --------------
    robot.bx += POISON_M
    moved = control.choose_velocity(robot, ranges, angles, goal)
    robot.bx = x

    if moved == baseline:
        return ["choose_velocity did not change when the BELIEVED pose "
                "moved, so the check above proves nothing -- the "
                "controller is not reading either pose."]
    return []


# ---------------------------------------------------------------------
# 2 and 3. What squad.py hands the map and the planner, on a live mission
# ---------------------------------------------------------------------
def trace_mission(seed=7):
    """
    Run a real mission with sense_and_map and drive traced, recording the
    pose handed to the grid and the start handed to the planner, together
    with the calling robot's true and believed poses at that instant.

    The wrappers are installed on the SquadMember methods rather than on
    OccupancyGrid or the planner, because only there is `self.robot` in
    hand -- without it a recorded pose could not be attributed to the
    robot whose belief it is supposed to match.
    """
    samples = []
    drifted = {"integrate_scan": 0, "plan_start": 0}

    real_sense = squad_module.SquadMember.sense_and_map
    real_drive = squad_module.SquadMember.drive

    def record(kind, robot, handed):
        drift = robot.pose_error_m()
        samples.append({
            "kind": kind,
            "robot": robot.id,
            "handed": tuple(float(v) for v in handed),
            "true": (robot.x, robot.y, robot.theta),
            "believed": (robot.bx, robot.by, robot.btheta),
            "drift": drift,
        })
        if drift > MIN_DRIFT_M:
            drifted[kind] += 1
        if all(n >= DRIFTED_PER_SITE for n in drifted.values()):
            raise _Enough

    def traced_sense(self, facility, step):
        robot, grid = self.robot, self.grid
        inner = grid.integrate_scan

        def spy(x, y, theta, *args, **kwargs):
            record("integrate_scan", robot, (x, y, theta))
            return inner(x, y, theta, *args, **kwargs)

        grid.integrate_scan = spy          # shadows the bound method
        try:
            return real_sense(self, facility, step)
        finally:
            del grid.integrate_scan        # and reveals it again

    def traced_drive(self, facility, step, radio, squad):
        robot, planner = self.robot, self.planner
        inner = planner.plan

        def spy(occupancy_map, start_xy, goal_xy):
            record("plan_start", robot, start_xy)
            return inner(occupancy_map, start_xy, goal_xy)

        planner.plan = spy
        try:
            return real_drive(self, facility, step, radio, squad)
        finally:
            del planner.plan

    squad_module.SquadMember.sense_and_map = traced_sense
    squad_module.SquadMember.drive = traced_drive
    try:
        demo_squad.run(seed, verbose=False)
    except _Enough:
        pass
    finally:
        squad_module.SquadMember.sense_and_map = real_sense
        squad_module.SquadMember.drive = real_drive

    return samples


def check_samples(samples):
    problems = []
    kinds = {s["kind"] for s in samples}

    for kind, what in (("integrate_scan", "the map"),
                       ("plan_start", "the planner")):
        seen = [s for s in samples if s["kind"] == kind]
        if not seen:
            problems.append(f"no {kind} calls were traced -- the test "
                            f"cannot say anything about {what}")
            continue

        # NON-VACUOUS FIRST. While the two poses still agree, "believed"
        # and "true" are the same numbers and every comparison below passes
        # for free. Odometry has to have drifted before this proves
        # anything, so require that it has.
        drifted = [s for s in seen if s["drift"] > MIN_DRIFT_M]
        if not drifted:
            problems.append(
                f"{kind}: odometry never drifted more than {MIN_DRIFT_M} m, "
                "so true and believed are indistinguishable and this check "
                "is vacuous")
            continue

        for s in drifted:
            n = len(s["handed"])
            if s["handed"] != s["believed"][:n]:
                problems.append(
                    f"{kind}: robot {s['robot']} was handed {s['handed']}, "
                    f"which is not its believed pose {s['believed'][:n]}")
                break
            if s["handed"] == s["true"][:n]:
                problems.append(
                    f"{kind}: robot {s['robot']} was handed its TRUE pose "
                    f"{s['handed']} -- {what} is using ground truth")
                break

    return problems, kinds


# ---------------------------------------------------------------------
def main():
    print("TRUE-POSE GUARD -- navigation and mapping must use the believed "
          "pose\n")

    problems = check_controller()
    verdict = ("FAIL" if problems else
               "blind to the true pose, and does react to the believed one")
    print(f"  1. control.choose_velocity   {verdict}")

    samples = trace_mission()
    more, _kinds = check_samples(samples)
    problems += more

    for kind, what in (("integrate_scan", "2. grid.integrate_scan     "),
                       ("plan_start", "3. planner.plan            ")):
        seen = [s for s in samples if s["kind"] == kind]
        drifted = [s for s in seen if s["drift"] > MIN_DRIFT_M]
        worst = max((s["drift"] for s in seen), default=0.0)
        print(f"  {what} {len(seen):>4d} calls traced, {len(drifted):>4d} "
              f"with odometry drifted (worst {worst:.3f} m)")

    print()
    if problems:
        print("FAIL -- the believed-pose requirement is not being met:")
        for p in problems:
            print("  -", p)
        return 1
    print("PASS -- the controller, the map and the planner all use the "
          "believed pose,")
    print("       measured on calls where the two poses had genuinely "
          "diverged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
