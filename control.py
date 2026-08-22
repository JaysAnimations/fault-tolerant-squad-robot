"""
control.py
==========
The low-level "how do I drive toward that point without hitting anything"
controller. Shared by every demo so there is exactly one copy.

This is deliberately dumb. It knows nothing about maps, frontiers, or
missions -- you hand it a target point and the latest LiDAR scan, and it
returns a speed and a turn rate. All the intelligence lives above it.
"""

import numpy as np
import config


def wrap_angle(a):
    """Fold any angle into (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def choose_velocity(robot, ranges, angles, goal_xy):
    """
    Reactive go-to-goal with obstacle repulsion.

    Returns (v, w, distance_to_goal).

    CRITICAL: uses the robot's BELIEVED pose (bx, by, btheta), never its true
    pose. If this ever reads robot.x directly, the SLAM claim collapses.
    """
    gx, gy = goal_xy
    dx, dy = gx - robot.bx, gy - robot.by
    dist_to_goal = float(np.hypot(dx, dy))

    heading_error = wrap_angle(np.arctan2(dy, dx) - robot.btheta)

    w = 1.6 * heading_error
    alignment = max(0.0, np.cos(heading_error))
    v = config.MAX_LINEAR_SPEED_MPS * alignment

    if ranges is not None:
        rel = wrap_angle(angles - robot.btheta)

        # Braking cone is NARROW on purpose (+/- 25 deg). A wider cone makes
        # the robot brake for corridor side-walls and chatter, burning most
        # of its energy on turning. This was a real bug -- do not widen it.
        cone = np.abs(rel) < np.deg2rad(25)
        d_front = ranges[cone].min() if cone.any() else config.LIDAR_MAX_RANGE_M

        left = (rel > np.deg2rad(20)) & (rel < np.deg2rad(85))
        right = (rel < -np.deg2rad(20)) & (rel > -np.deg2rad(85))
        d_left = ranges[left].min() if left.any() else config.LIDAR_MAX_RANGE_M
        d_right = ranges[right].min() if right.any() else config.LIDAR_MAX_RANGE_M

        influence = 1.0
        if d_left < influence:
            w -= 1.6 * (influence - d_left)
        if d_right < influence:
            w += 1.6 * (influence - d_right)

        if d_front < 1.2:
            v *= float(np.clip((d_front - 0.40) / 0.80, 0.0, 1.0))

        if d_front < 0.50:
            v = 0.0
            w = 1.2 * (1.0 if d_left >= d_right else -1.0)

    return v, w, dist_to_goal
