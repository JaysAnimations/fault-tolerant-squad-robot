"""
config.py
=========
EVERY tunable number in the whole project lives in this file.

Rule for the team: if you find yourself editing a number inside any other
file, stop -- move it here instead. During your defence you want to be able
to say "all experimental parameters are centralised and version-controlled",
and you want to be able to change an experiment without touching logic.

Units are stated in every variable name. Do not mix units.
"""

# =====================================================================
# 1. WORLD GEOMETRY
# =====================================================================
# The simulated oil & gas facility. Chosen to be big enough that a single
# robot struggles to cover it in a reasonable time (this is what justifies
# using a squad at all -- see experiment C4).

WORLD_WIDTH_M = 80.0      # facility width  (x axis, metres)
WORLD_HEIGHT_M = 55.0     # facility height (y axis, metres)
GRID_RES_M = 0.10         # size of one occupancy-grid cell, metres
                          # 0.10 m => 600 x 400 = 240,000 cells. Fine for numpy.


# =====================================================================
# 2. LIDAR SENSOR MODEL
# =====================================================================
# Modelled on a low-cost 2D spinning LiDAR (e.g. RPLIDAR A1 class).
# Keep N_RAYS modest -- it is the single biggest driver of runtime.

LIDAR_N_RAYS = 72            # rays per scan
LIDAR_FOV_DEG = 360.0        # full rotation
LIDAR_MAX_RANGE_M = 8.0      # beyond this the sensor reports "no return"
LIDAR_MIN_RANGE_M = 0.15     # blind zone right next to the robot
LIDAR_RAY_STEP_M = 0.05      # ray-marching resolution. Smaller = more accurate,
                             # slower. Must be <= GRID_RES_M to avoid skipping walls.
LIDAR_NOISE_STD_M = 0.02     # Gaussian range noise, std dev (healthy sensor)


# =====================================================================
# 3. ROBOT PLATFORM
# =====================================================================
ROBOT_RADIUS_M = 0.25        # used for collision checking
MAX_LINEAR_SPEED_MPS = 0.6   # forward speed cap
MAX_ANGULAR_SPEED_RPS = 1.2  # turn rate cap, radians/second
DT_S = 0.10                  # simulation timestep, seconds (10 Hz control loop)

# Odometry drift. Real wheel encoders slip. These multiply the commanded
# motion to produce the robot's *believed* motion, which is what makes SLAM
# non-trivial and what loop closure has to correct.
# These model a FUSED pose estimate (wheel encoders + IMU), not raw wheel
# odometry. That is what the literature describes as standard practice for
# industrial mobile robots, and it is the honest description of what we
# simulate: we model dead reckoning but not scan matching or loop closure,
# so the fusion has to carry the accuracy that localisation would otherwise
# provide.
#
# WHY THIS MATTERS MORE THAN IT LOOKS: the Byzantine fault detector works by
# comparing robots' maps against each other. Drift in HEALTHY robots is the
# noise floor of that comparison. At the previous values a healthy robot
# accumulated ~1.5 m of error over a mission, which is the same order as the
# pose offset we inject as a fault -- so a faulty robot was statistically
# indistinguishable from a tired one. Reducing drift is not cosmetic; it is
# what makes fault detection possible at all.
ODOM_LINEAR_DRIFT_STD = 0.005   # fractional error per step on distance
ODOM_ANGULAR_DRIFT_STD = 0.003  # fractional error per step on rotation


# =====================================================================
# 4. ENERGY MODEL   <-- THIS IS THE HEART OF YOUR CONTRIBUTION
# =====================================================================
# E_total = E_drive*distance + E_turn*|rotation| + P_sense*t + P_compute*t
#           + E_comms*bytes_sent
#
# !!! IMPORTANT !!!
# These are PLACEHOLDER values. Before your final experiment runs, the
# hardware team must replace them with coefficients measured from the
# INA219 current sensors on the physical robots. That measurement is what
# turns this from "a simulation we invented" into "an empirically
# calibrated model". Do not skip it -- it is your strongest defence point.

E_DRIVE_J_PER_M = 8.0        # joules to travel one metre in a straight line
E_TURN_J_PER_RAD = 3.5      # joules to rotate one radian on the spot
                             # (higher than driving: both wheels fight each other)
P_SENSE_W = 0.80             # LiDAR power draw, watts (continuous while scanning)
P_COMPUTE_W = 1.20           # onboard processing, watts (continuous while alive)
E_COMMS_J_PER_KB = 0.05      # joules per kilobyte transmitted

BATTERY_CAPACITY_J = 33000.0 # ~2x 18650 cells at 3.7V 2500mAh. A robot that
                             # exhausts this is dead for the rest of the mission.


# =====================================================================
# 5. OCCUPANCY GRID MAPPING (log-odds)
# =====================================================================
# Standard log-odds occupancy mapping (Thrun, Burgard & Fox -- reference [3]
# in HENRY's chapter 2, so you can cite it directly).

LOG_ODDS_OCCUPIED = 0.85     # added to a cell when a beam ENDS there
LOG_ODDS_FREE = -0.40        # added to every cell the beam PASSES THROUGH
LOG_ODDS_CLAMP = 8.0         # saturate at +/- this, so the map stays correctable
LOG_ODDS_OCC_THRESHOLD = 1.0   # above this we call a cell "occupied"
LOG_ODDS_FREE_THRESHOLD = -1.0 # below this we call a cell "free"
                               # between the two = "unknown"


# =====================================================================
# 6. REPRODUCIBILITY
# =====================================================================
# Every experimental run is seeded. Run 7 of condition C2 must produce
# byte-identical results every time you run it. Panels ask about this.

DEFAULT_SEED = 42
