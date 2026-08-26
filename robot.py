"""
robot.py
========
A single differential-drive robot: where it is, how it moves, how much
energy that costs, and how much battery it has left.

TWO POSES, AND WHY IT MATTERS:
  self.x, self.y, self.theta        -> TRUE pose (physics). The robot cannot see this.
  self.bx, self.by, self.btheta     -> BELIEVED pose (odometry). This is what
                                       the robot actually uses.
  These drift apart over time because wheels slip. The gap between them is
  what SLAM exists to close, and the residual between them is one of your
  fault-detection signals. Keep them strictly separate in your code -- if a
  panel sees the robot using its true pose, the whole SLAM claim collapses.

ENERGY:
  Every joule spent is attributed to a category (drive / turn / sense /
  compute / comms). Do not just track a single total -- the breakdown is
  what lets you argue about WHERE fault tolerance saves energy.
"""

import numpy as np
import config


def wrap_angle(a):
    """Fold any angle into (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class Robot:

    def __init__(self, robot_id, x, y, theta, rng):
        self.id = robot_id
        self.rng = rng

        # --- true pose (physics ground truth) ---
        self.x, self.y, self.theta = float(x), float(y), float(theta)

        # --- believed pose (odometry estimate) ---
        self.bx, self.by, self.btheta = float(x), float(y), float(theta)

        # --- health state ---
        self.alive = True          # False = permanently dead
        self.mobile = True         # False = motors failed, can still sense/relay
        self.sensing = True        # False = LiDAR dead, can still drive/relay
        self.connected = True      # False = out of comms

        # --- energy ledger, all in joules ---
        self.energy = {
            "drive": 0.0,
            "turn": 0.0,
            "sense": 0.0,
            "compute": 0.0,
            "comms": 0.0,
        }
        self.battery_j = config.BATTERY_CAPACITY_J
        self.battery_drain_multiplier = 1.0   # >1.0 simulates a battery fault

        # --- odometers, for the progress-monitor fault detector ---
        self.distance_travelled_m = 0.0
        self.rotation_travelled_rad = 0.0
        self.commanded_distance_m = 0.0   # what we ASKED for
        self.commanded_rotation_rad = 0.0  # ...including asking to turn
        self.collisions = 0

    # -----------------------------------------------------------------
    @property
    def total_energy_j(self):
        return sum(self.energy.values())

    @property
    def battery_fraction(self):
        return max(0.0, self.battery_j / config.BATTERY_CAPACITY_J)

    def _spend(self, category, joules):
        joules = float(joules)
        self.energy[category] += joules
        self.battery_j -= joules * self.battery_drain_multiplier
        if self.battery_j <= 0.0:
            self.battery_j = 0.0
            self.alive = False
            self.mobile = False
            self.sensing = False
            self.connected = False

    # -----------------------------------------------------------------
    def step_motion(self, v_cmd, w_cmd, facility, dt=config.DT_S):
        """
        Advance one timestep under commanded linear speed v (m/s) and
        angular speed w (rad/s).

        Returns True if the move succeeded, False if it was blocked by a
        collision (in which case the robot stays put but still pays the
        energy -- exactly like a real robot pushing against a wall).
        """
        if not self.alive:
            return False

        v = float(np.clip(v_cmd, -config.MAX_LINEAR_SPEED_MPS,
                          config.MAX_LINEAR_SPEED_MPS))
        w = float(np.clip(w_cmd, -config.MAX_ANGULAR_SPEED_RPS,
                          config.MAX_ANGULAR_SPEED_RPS))

        ds = v * dt              # distance this step
        dth = w * dt             # rotation this step

        # RECORD WHAT WAS ASKED FOR BEFORE CHECKING WHETHER IT HAPPENED.
        # These are "what we ASKED for", and they are the only signal that
        # separates a robot whose motors have failed from one that is
        # legitimately parked: both stay still, but only one is still
        # asking to move. Counting them after the mobility check made the
        # two indistinguishable in telemetry, which would have left the
        # immobilised fault undetectable by anything except ground truth.
        #
        # ROTATION MATTERS AS MUCH AS DISTANCE, and leaving it out very
        # nearly hid the fault completely. An immobilised robot whose next
        # path node is behind it commands a pure turn -- v is zero, because
        # the controller will not drive until it is facing the right way.
        # It cannot turn, so its heading never changes, so it commands the
        # same pure turn forever. Distance commanded: zero. It looks
        # exactly like a robot that has parked, while it is in fact
        # straining against dead motors.
        self.commanded_distance_m += abs(ds)
        self.commanded_rotation_rad += abs(dth)

        # A robot with failed motors pays idle power but goes nowhere.
        if not self.mobile:
            self._spend("compute", config.P_COMPUTE_W * dt)
            return False

        # --- proposed TRUE motion ---
        new_theta = wrap_angle(self.theta + dth)
        mid_theta = self.theta + dth / 2.0    # midpoint integration
        new_x = self.x + ds * np.cos(mid_theta)
        new_y = self.y + ds * np.sin(mid_theta)

        # --- collision check against ground truth ---
        blocked = not facility._has_clearance(new_x, new_y, config.ROBOT_RADIUS_M)

        # --- energy is spent whether or not we actually moved ---
        self._spend("drive", config.E_DRIVE_J_PER_M * abs(ds))
        self._spend("turn", config.E_TURN_J_PER_RAD * abs(dth))
        self._spend("compute", config.P_COMPUTE_W * dt)

        if blocked:
            self.collisions += 1
            # Rotation still happens on a blocked robot (it can spin in place)
            self.theta = new_theta
            self._advance_odometry(0.0, dth)
            self.rotation_travelled_rad += abs(dth)
            return False

        self.x, self.y, self.theta = new_x, new_y, new_theta
        self.distance_travelled_m += abs(ds)
        self.rotation_travelled_rad += abs(dth)
        self._advance_odometry(ds, dth)
        return True

    def _advance_odometry(self, ds, dth):
        """
        Update the robot's BELIEVED pose using noisy wheel measurements.
        The noise is multiplicative because encoder error scales with
        distance travelled -- which is why odometry drift accumulates.
        """
        ds_meas = ds * (1.0 + self.rng.normal(0.0, config.ODOM_LINEAR_DRIFT_STD))
        dth_meas = dth * (1.0 + self.rng.normal(0.0, config.ODOM_ANGULAR_DRIFT_STD))

        mid = self.btheta + dth_meas / 2.0
        self.bx += ds_meas * np.cos(mid)
        self.by += ds_meas * np.sin(mid)
        self.btheta = wrap_angle(self.btheta + dth_meas)

    # -----------------------------------------------------------------
    def pay_sensing(self, dt=config.DT_S):
        self._spend("sense", config.P_SENSE_W * dt)

    def pay_comms(self, kilobytes):
        self._spend("comms", config.E_COMMS_J_PER_KB * kilobytes)

    # -----------------------------------------------------------------
    def pose_error_m(self):
        """
        Distance between where the robot THINKS it is and where it IS.
        Used only for analysis/plots -- never for control.
        """
        return float(np.hypot(self.x - self.bx, self.y - self.by))

    def status(self):
        if not self.alive:
            return "DEAD"
        flags = []
        if not self.mobile:
            flags.append("immobile")
        if not self.sensing:
            flags.append("blind")
        if not self.connected:
            flags.append("offline")
        return "OK" if not flags else "+".join(flags)

    def __repr__(self):
        return (f"<Robot {self.id} @({self.x:.1f},{self.y:.1f}) "
                f"{self.status()} batt={self.battery_fraction*100:.0f}%>")
