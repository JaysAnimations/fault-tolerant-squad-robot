"""
sensors.py
==========
Simulated 2D LiDAR.

HOW IT WORKS (you must be able to explain this at the defence):
  A real LiDAR fires a laser pulse and times the reflection. We reproduce
  that geometrically by "ray marching": for each of N beam directions we
  step outward from the robot in small increments and check whether we have
  entered an obstacle cell. The distance at which we first hit something is
  the reported range. If we never hit anything within max range, we report
  a "no return".

WHY IT IS VECTORISED:
  A naive version uses two nested Python loops (rays x steps) and is ~100x
  slower. Because you need 120 experimental runs, speed matters. We build a
  single (n_rays x n_steps) numpy array of sample points and resolve every
  ray simultaneously.

FAULT HOOKS:
  The scan() method accepts optional fault parameters. This is where the
  fault-injection team plugs in -- you do NOT need to modify this file to
  add degraded or Byzantine sensor behaviour.
"""

import numpy as np
import config


class Lidar2D:
    def __init__(self,
                 n_rays=config.LIDAR_N_RAYS,
                 fov_deg=config.LIDAR_FOV_DEG,
                 max_range_m=config.LIDAR_MAX_RANGE_M,
                 step_m=config.LIDAR_RAY_STEP_M):

        self.n_rays = n_rays
        self.max_range = max_range_m
        self.step = step_m

        # Beam directions relative to the robot's heading, evenly spaced.
        fov = np.deg2rad(fov_deg)
        if abs(fov_deg - 360.0) < 1e-6:
            # full circle: don't duplicate the 0 and 2*pi beam
            self.ray_angles = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)
        else:
            self.ray_angles = np.linspace(-fov / 2, fov / 2, n_rays)

        # Precompute the sample distances along every ray: [step, 2*step, ...]
        self.n_steps = int(np.ceil(max_range_m / step_m))
        self.sample_dists = (np.arange(1, self.n_steps + 1) * step_m)  # (S,)

    # -----------------------------------------------------------------
    def scan(self, facility, x, y, theta, rng,
             noise_std=config.LIDAR_NOISE_STD_M,
             range_scale=1.0,
             dropout_prob=0.0,
             bias_m=0.0):
        """
        Produce one full LiDAR scan from pose (x, y, theta).

        Returns
        -------
        ranges : (n_rays,) float array
            Measured distance per beam. Beams with no return are set to
            max_range (use `valid` to distinguish).
        angles : (n_rays,) float array
            Absolute world-frame bearing of each beam.
        valid : (n_rays,) bool array
            True where the beam actually struck something.

        Fault parameters (all default to "healthy")
        -------------------------------------------
        noise_std     : inflate to simulate a DEGRADED sensor
        range_scale   : <1.0 simulates a sensor losing effective range
        dropout_prob  : probability an individual beam returns nothing
        bias_m        : systematic offset -- this is your BYZANTINE fault.
                        A biased sensor produces readings that look
                        perfectly plausible in isolation but are wrong,
                        which is precisely why it can only be caught by
                        cross-checking against other robots.
        """
        # --- absolute beam directions ---
        angles = theta + self.ray_angles                      # (N,)
        cos_a = np.cos(angles)[:, None]                       # (N,1)
        sin_a = np.sin(angles)[:, None]

        eff_max = self.max_range * range_scale
        n_steps = int(np.ceil(eff_max / self.step))
        dists = self.sample_dists[:n_steps][None, :]          # (1,S)

        # --- sample points along every ray, all at once ---
        px = x + cos_a * dists                                # (N,S)
        py = y + sin_a * dists

        rows, cols = facility.world_to_grid(px, py)
        inside = facility.in_bounds(rows, cols)

        # A sample is "blocked" if it is an obstacle OR outside the world.
        blocked = ~inside
        rr = np.clip(rows, 0, facility.n_rows - 1)
        cc = np.clip(cols, 0, facility.n_cols - 1)
        blocked |= (facility.grid[rr, cc] > 0) & inside

        # First blocked sample index per ray. argmax on a bool array returns
        # the first True; if there is no True it returns 0, so we guard with
        # any().
        hit_any = blocked.any(axis=1)                         # (N,)
        first_hit = np.argmax(blocked, axis=1)                # (N,)

        ranges = np.where(hit_any,
                          self.sample_dists[:n_steps][first_hit],
                          eff_max)
        valid = hit_any.copy()

        # --- sensor imperfections ---
        if noise_std > 0:
            ranges = ranges + rng.normal(0.0, noise_std, size=self.n_rays)
        if bias_m != 0.0:
            ranges = ranges + bias_m
        if dropout_prob > 0:
            dropped = rng.random(self.n_rays) < dropout_prob
            valid &= ~dropped

        # --- clamp to physically meaningful values ---
        ranges = np.clip(ranges, config.LIDAR_MIN_RANGE_M, eff_max)
        valid &= ranges < (eff_max - 1e-6)

        return ranges, angles, valid
