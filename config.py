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


# =====================================================================
# 7. FRONTIER EXPLORATION  (Step 1b)
# =====================================================================
# A frontier is known-free space touching unknown space -- the only place
# new information can come from. The robot drives to frontiers until none
# remain, at which point every reachable area has been seen.

FRONTIER_MIN_SIZE_CELLS = 3      # ignore smaller regions: usually edge noise
FRONTIER_UNKNOWN_FRACTION = 0.5  # coarse cell counts as unknown above this
FRONTIER_REACHED_M = 1.5         # close enough to consider a frontier visited

# Selection rule:  score = size - FRONTIER_DISTANCE_WEIGHT * distance
#   weight -> 0    : always chase the largest frontier, wherever it is
#   weight -> large: always take the nearest, however small
# Since this project reports energy per m2, the rule must trade information
# against travel rather than optimise either alone.
# TUNED EMPIRICALLY. Full-mission sweep on seed 42 (weight / min-travel /
# min-commit -> coverage, collisions, energy per m2):
#     0.15 / 10 /  60  ->  93.1 %, 915 collisions, 5.88 J/m2
#     0.15 / 12 / 250  ->  94.9 %,  54 collisions, 7.63 J/m2   <-- chosen
#     0.30 / 16 / 250  ->  91.4 %,   0 collisions, 8.07 J/m2
#     0.80 / 12 / 200  ->  69.8 %, 336 collisions, 3.95 J/m2
#     1.50 / 12 / 200  ->  55.7 %,   0 collisions, 9.58 J/m2
#
# The pattern is a genuine trade-off worth reporting in Chapter 4: a strong
# locality preference finishes the local area cheaply but declares
# exploration complete while unreachable frontiers remain, so coverage
# collapses. A weak preference covers the site but wanders. We optimise for
# coverage, since an incomplete industrial map is worthless regardless of
# how little energy it cost to produce.
FRONTIER_DISTANCE_WEIGHT = 0.15

FRONTIER_MIN_COMMIT_STEPS = 250  # hold a chosen frontier at least this long
                                 # 60 caused 915 collisions; 250 gives 54
FRONTIER_CHECK_EVERY = 5         # steps between goal-validity checks
FRONTIER_MIN_TRAVEL_CELLS = 12   # 4.8 m: closer frontiers are consumed by looking


# =====================================================================
# 8. RANDOM NUMBER STREAMS  (Design Change 01)
# =====================================================================
# A run has ONE seed, but several things draw from it: where the
# inspection points go, which deviations are injected, and the sensor and
# odometry noise the robot experiences.
#
# If they all shared one generator, switching condition C0 (no deviations)
# to C1 (deviations) would consume a different number of draws and shift
# every later random number -- so the two conditions would no longer be
# the same mission, and the paired comparison the experiment depends on
# would be broken.
#
# Each purpose therefore gets its own independent stream, derived from the
# run seed as default_rng([seed, STREAM_ID]). Seed 7 gives the same
# inspection points in every condition whether deviations are injected or
# not, which is exactly what decision 14 in the work log requires.
RNG_STREAM_ROBOT = 0        # sensor noise, odometry drift
RNG_STREAM_INSPECTION = 1   # inspection point positions
RNG_STREAM_DEVIATIONS = 2   # which deviations, and where
RNG_STREAM_COMMS = 3        # radio packet loss
# In the squad, each robot draws from [seed, RNG_STREAM_ROBOT, robot_id],
# so robot 1's sensor noise does not shift when robot 0 happens to drive
# somewhere else. Without that, changing one robot's behaviour would
# change every robot's noise and the paired comparison would be lost.


# =====================================================================
# 9. MISSION: ZONES AND INSPECTION POINTS  (Design Change 01)
# =====================================================================
# The mission is no longer "cover as much area as possible" but "visit
# every inspection point". Points are stratified over 11 functional zones
# so that no seed can produce a mission where every point sits in one
# corner of the site.
#
# Each zone is an axis-aligned rectangle (x0, y0, x1, y1) in metres. The
# rectangles are deliberately DISJOINT, so a point belongs to exactly one
# zone, and each is drawn around a functional area of the layout in
# environment.py. They are allowed to contain obstacles -- point placement
# rejects any position that is not free, so a zone rectangle is a region to
# sample from, not a claim that the whole rectangle is drivable.
ZONES = [
    # code, name,                          x0,    y0,    x1,    y1
    ("Z1",  "Perimeter road, south",       0.4,   0.4,  79.6,   6.2),
    ("Z2",  "Perimeter road, north",       0.4,  52.6,  79.6,  54.6),
    ("Z3",  "Utility area",                6.4,   6.4,  28.0,  22.5),
    ("Z4",  "Substation interior",        29.4,   8.9,  37.6,  14.6),
    ("Z5",  "Tank farm (inside bund)",    43.5,   7.0,  72.5,  20.5),
    ("Z6",  "Main pipe rack corridor",     6.4,  22.6,  76.0,  28.5),
    ("Z7",  "Process unit",               15.0,  34.6,  42.0,  47.5),
    ("Z8",  "Pump row",                   15.0,  28.8,  42.0,  34.5),
    ("Z9",  "Compressor house interior",  45.4,  33.4,  69.6,  45.6),
    ("Z10", "Control building interior",   7.4,  32.4,  13.1,  39.6),
    ("Z11", "Store / workshop interior",   7.4,  45.4,  18.6,  52.1),
]

INSPECTION_POINTS_TARGET = 40     # total points per run, every run
INSPECTION_MIN_PER_ZONE = 2       # stratification: no zone may be ignored
INSPECTION_MAX_PER_ZONE = 5       # and none may hoard the mission
# 11 zones x 2..5 spans 22..55 points, so a target of 40 is always
# satisfiable. Holding the TOTAL fixed while randomising positions is what
# keeps mission difficulty comparable between seeds (decision 1).

INSPECTION_CLEARANCE_M = 0.55     # the robot (r = 0.25 m) must be able to
                                  # stand on the point, with margin

# Obstacle inflation used when asking "can a robot physically get there?",
# as opposed to "what route should it drive?".
#
# THESE ARE TWO DIFFERENT QUESTIONS AND THEY NEED DIFFERENT NUMBERS. The
# navigation planner inflates by 2 coarse cells (0.8 m) because a path that
# scrapes a wall produces collisions. But 0.8 m of inflation closes the
# control building's 2.0 m doorway completely, so a reachability test using
# it declares the whole room unreachable -- and the room is demonstrably
# reachable, since the scripted tour in demo_single.py drives into it.
#
# One coarse cell is 0.4 m, comfortably more than the 0.25 m robot radius,
# so this stays a conservative test of whether the robot fits.
REACHABILITY_INFLATE_CELLS = 1

# Obstacle inflation used for NAVIGATION, in coarse cells of 0.4 m.
#
# The planner tries the normal margin first and falls back to the tight one
# only when the normal margin finds no route at all. Two numbers rather
# than one because they trade against each other and the trade is measured:
# at 0.4 m clearance the scripted tour collided 3026 times, at 0.8 m it
# collides 20 times, and at 1.2 m the site fragments into 5 disconnected
# regions. So 0.8 m is right for driving -- but it is 0.8 m from BOTH
# sides, which closes the control building's 2.0 m doorway completely and
# makes the planner declare a demonstrably reachable room unreachable.
#
# Comfortable margin normally, minimum margin only where the geometry
# leaves no choice. The tight value matches REACHABILITY_INFLATE_CELLS on
# purpose: it is the same physical claim, that 0.4 m clears a 0.25 m robot.
PLANNER_INFLATE_CELLS = 2
PLANNER_TIGHT_INFLATE_CELLS = 1
INSPECTION_MIN_SEPARATION_M = 2.5  # two points 0.3 m apart are one point
INSPECTION_PLACEMENT_ATTEMPTS = 500  # rejection sampling budget per point
INSPECTION_REACHED_M = 0.9        # believed distance that counts as a visit
                                  # (matches WAYPOINT_TOLERANCE_M in the
                                  # scripted tour, so the two are comparable)

# Fixed deployment point: the charging station just inside the gate. Held
# constant across every run and every condition so start position is not a
# nuisance variable in the comparison (decision 6 / decision 13).
START_POSE_XY = (3.5, 3.5)
START_THETA_RAD = 0.0

# --- running the inspection mission (demo_inspect.py) -----------------
# Same shape as the scripted tour's loop, so the two are comparable.
INSPECTION_MAX_STEPS = 40000
INSPECTION_SCAN_EVERY_N_STEPS = 3       # LiDAR at ~3 Hz, control at 10 Hz
INSPECTION_REPLAN_EVERY_N_STEPS = 120   # the map improves as it drives, and
                                        # a route that looked open may not be
INSPECTION_PATH_NODE_TOLERANCE_M = 0.45
INSPECTION_PLAN_FAIL_COOLDOWN = 60
INSPECTION_NO_PROGRESS_STEPS = 500      # stop closing on a point for this
                                        # long and the robot gives up on it
INSPECTION_MAX_STEPS_PER_POINT = 4000   # backstop only; the no-progress rule
                                        # is the real detector


# =====================================================================
# 10. PRIOR MAP  (Design Change 01, section 5)
# =====================================================================
# Robots are issued the facility's documented layout at the start of the
# mission, seeded into their occupancy grid at +/- this many log-odds.
#
# THIS SINGLE NUMBER IS THE WHOLE DESIGN:
#   too high -> the robot argues with reality and never sees a deviation
#   too low  -> the prior buys nothing and we are back to exploring
#
# At 2.0, with the classification threshold at 1.0 and a beam-hit worth
# +0.85, roughly 4 contradicting observations overturn a "free" cell and
# roughly 8 overturn an "occupied" one (LOG_ODDS_FREE is deliberately
# weaker than LOG_ODDS_OCCUPIED). The drawings are therefore believed, but
# overturnable -- which is the entire point.
PRIOR_LOG_ODDS = 2.0


# =====================================================================
# 11. DEVIATIONS FROM THE DRAWINGS  (Design Change 01, section 4)
# =====================================================================
# Ground truth contains things the documented layout does not show.
# Injecting them per run, from the seed, is what makes deviation count a
# per-run variable rather than a fixture -- which is what makes it usable
# statistically.
#
# Set DEVIATIONS_MIN/MAX to 0 for condition C0 (no deviations).

DEVIATIONS_MIN_PER_RUN = 5
DEVIATIONS_MAX_PER_RUN = 8

# Relative likelihood of each type. Normalised in code, so these are
# weights and need not sum to 1.
#   added   -- scaffolding, a temporary barrier, parked equipment:
#              the drawings say open, reality says blocked
#   removed -- equipment taken out for maintenance:
#              the drawings say solid, reality says open
#   blocked -- an aisle closed off. The operationally interesting one,
#              because it changes the COST of the mission and not just
#              the map.
DEVIATION_TYPE_WEIGHTS = {"added": 0.4, "removed": 0.3, "blocked": 0.3}

DEVIATION_ADDED_MIN_SIZE_M = 1.2   # a scaffold tower
DEVIATION_ADDED_MAX_SIZE_M = 3.0   # a parked trailer
DEVIATION_PLACEMENT_ATTEMPTS = 300

# At most one deviation per run may cut an inspection point off from the
# rest of the site (decision 3). More than that and a bad seed would
# decide the experiment. Any candidate that would exceed this is reverted
# and another is drawn.
MAX_UNREACHABLE_POINTS = 1

# Corridors that can plausibly be closed off, as barrier rectangles
# (name, x0, y0, x1, y1). Hand-picked rather than random because "an aisle
# closed off" is only meaningful if there IS an aisle there; a random
# rectangle in open ground is an added obstacle, which is the other type.
#
# Most of these have an alternative route, so they force a detour. A few
# (the bund access gap especially) would seal a whole area off; those get
# reverted by the MAX_UNREACHABLE_POINTS rule above, which is the rule
# doing its job rather than a candidate list that needs fixing.
BLOCKED_ROUTE_CANDIDATES = [
    ("compressor house south door",    54.0, 33.0, 57.0, 33.5),
    ("compressor house west door",     45.0, 38.5, 45.5, 41.0),
    ("compressor house centre aisle",  56.5, 36.0, 59.5, 39.0),
    ("pipe rack bay at x = 24 m",      21.4, 25.0, 26.6, 26.0),
    ("pipe rack bay at x = 54 m",      51.4, 25.0, 56.6, 26.0),
    ("process aisle, exchangers 1-2",  21.0, 36.0, 23.0, 38.2),
    ("process aisle, exchangers 2-3",  27.0, 36.0, 29.0, 38.2),
    ("utility aisle, skids 1-2",       13.6,  9.0, 15.4, 12.0),
    ("utility aisle, skids 2-3",       20.1,  9.0, 21.9, 12.0),
    ("north road behind the store",    13.0, 52.6, 14.0, 54.5),
    ("east yard corridor",             73.5, 12.0, 79.5, 13.0),
    ("bund access gap",                43.0, 12.5, 43.5, 15.5),
]

# A deviation counts as DETECTED when this fraction of its evidence cells
# (see deviations.py) have been overturned in the robot's own map -- that
# is, the robot now believes the opposite of what the drawings told it.
#
# It is a fraction and not a cell count because deviations differ in size
# by two orders of magnitude: a scaffold tower is a few hundred cells, a
# removed exchanger nearly nine thousand.
DEVIATION_DETECT_FRACTION = 0.25

# Spatial tolerance when comparing the robot's map against the drawings,
# in cells. DELIBERATELY THE SAME NUMBER as the tolerance in
# OccupancyGrid.surface_scores(), and for exactly the same reason: the
# robot integrates its scans at its BELIEVED pose, so a correctly observed
# obstacle lands a cell or two away from its true footprint. Scoring
# detection at zero tolerance measures odometry error rather than whether
# the robot found the deviation.
#
# Measured on seed 42: at zero tolerance the run detects 1 deviation of 5,
# and four of the misses have their evidence sitting 1-3 cells off. At 2
# cells it detects 4 of 5. The fifth is a genuine miss -- the route never
# passed within 10.5 m of it, and the LiDAR only reaches 8 m.
DEVIATION_TOLERANCE_CELLS = 2
DEVIATION_CHECK_EVERY_STEPS = 50   # steps between detection checks. Also
                                   # the resolution of the detection
                                   # latency reported per deviation type.


# =====================================================================
# 12. THE SQUAD  (Step 2)
# =====================================================================
# Three robots, each with a PRIVATE map. There is no shared map object
# anywhere in the code -- that is what "decentralised" means here, and a
# panel will check it. Robots learn what other robots saw only by being
# close enough to hear them say so.

SQUAD_SIZE = 3

# Fixed deployment: a rank of robots on the south perimeter road beside
# the gate, verified clear of obstacles at start-up. Robot 0 stands on the
# single-robot start position so the two missions are anchored to the same
# place and the comparison is fair. 4 m apart so they do not begin on top
# of one another.
SQUAD_START_POSES = [
    (3.5, 3.5, 0.0),
    (7.5, 3.5, 0.0),
    (11.5, 3.5, 0.0),
]

# --- radio -----------------------------------------------------------
COMMS_RANGE_M = 25.0          # beyond this, robots cannot hear each other.
                              # Steel vessels and cable trays are what makes
                              # this short in a real plant.
COMMS_PACKET_LOSS_PROB = 0.05  # an in-range message still fails 5 % of the
                               # time. Without this, comms would be a
                               # perfect channel and the comms-loss fault
                               # would have nothing to degrade from.

# Modelled payload sizes, in kilobytes. A map message does NOT carry the
# whole 240,000-cell grid: a real system sends a compressed update of the
# cells that changed, and charging 940 KB per meeting would make comms
# dominate an energy budget it has no business dominating. This is the
# modelled size of that update, and it is a parameter precisely so the
# assumption is visible rather than buried.
COMMS_MAP_PACKET_KB = 64.0
COMMS_CLAIM_PACKET_KB = 0.05   # a claim is a point number and a cost

COMMS_EXCHANGE_EVERY_N_STEPS = 200   # how often a robot offers its map

# --- auction over inspection points ----------------------------------
# A robot broadcasts what a point would cost it to reach. It does not
# pursue a point another robot has claimed more cheaply.
#
# CLAIMS ARE LEASES, NOT DEEDS. A claim is refreshed while the robot is
# still pursuing it and expires if the refresh stops. That is the whole
# reason for the timeout: a robot that dies holding six claims must not
# block those six points for the rest of the mission.
CLAIM_REFRESH_EVERY_N_STEPS = 100
CLAIM_TIMEOUT_STEPS = 400      # 40 s without a refresh and the claim lapses

# End the round when the whole squad has been unable to take on any work
# for this long. Points still open at that moment are unreachable as far
# as the squad is concerned.
#
# This is the mission's real termination rule and it must exist. A point
# cut off by a deviation is skipped by every robot in turn -- each looks at
# it, finds no route on its own map, and moves on -- so no robot ever
# "fails" at it in a way that would close it. Without this check, seed 2024
# ran to the 40,000-step ceiling and took ten minutes to admit that one
# valve was behind a wall.
SQUAD_STALL_STEPS = 600

# --- getting unstuck -------------------------------------------------
# The reactive controller can wedge itself in a corner. Its braking cone
# is +/- 25 degrees, but the collision box is a square, so a corner just
# outside the cone reads as clear ahead while blocking every attempt to
# move. The robot then commands full speed into it forever: on seed 2024
# one robot logged 2,219 collisions from a standstill and inspected one
# point in the whole mission.
#
# Nothing changes on its own in that state -- the map is already correct,
# the plan is already valid, and the robot is simply jammed -- so it needs
# an action that is not "drive at the goal". It reverses briefly and turns,
# which is what a real robot does and what the reactive layer cannot
# express, since it only ever drives forwards.
SQUAD_ESCAPE_AFTER_BLOCKED_STEPS = 40   # 4 s of commanded motion, no motion
SQUAD_ESCAPE_STEPS = 20                 # then back out for 2 s
SQUAD_ESCAPE_SPEED_FRACTION = 0.5       # of MAX_LINEAR_SPEED_MPS, reversed
SQUAD_ESCAPE_TURN_RPS = 0.6             # turn while reversing, so a second
                                        # attempt approaches differently

# --- trajectory trace -------------------------------------------------
# Logged so a mission can be replayed as video later without re-running
# the simulation. Kept deliberately thin: one sample per robot per second.
TRACE_EVERY_N_STEPS = 25

