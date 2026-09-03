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
# THE FOUR MAIN COEFFICIENTS ARE NO LONGER MAGIC NUMBERS. They are derived
# below from named component specifications, so Chapter 3 can show the
# arithmetic rather than asking a panel to take four numbers on trust.
# CHANGED IN SESSION 14, and flagged per the convention -- every energy
# figure in the project moves. Old -> new: E_DRIVE 8.0 -> 4.0 (-50 %),
# E_TURN 3.5 -> 2.0 (-43 %), P_SENSE 0.80 -> 2.00 (+150 %), P_COMPUTE
# 1.20 -> 0.60 (-50 %). The previous values were placeholders invented in
# Session 0 and never defended.
#
# The composition shift matters more than the totals: sensing rises from
# ~11.5 % of a mission's energy to ~39 %, so energy becomes far more
# TIME-dependent and less DISTANCE-dependent. A robot that is switched on
# and doing nothing is now expensive, which is the whole argument against a
# squad that flails for 2000 s.

# --- Component specifications, from manufacturer datasheets -------------
# TT gearbox motor, 1:48, published bench measurements (Adafruit 3777 /
# Cytron): rated 3-6 V, no-load current 150 mA +/- 10 %, LOADED CURRENT
# <= 200 mA at 6 V, 200 RPM +/- 10 % at 6 V, stall 1.5 A.
MOTOR_SUPPLY_V          = 6.0     # top of the rated 3-6 V range
MOTOR_CURRENT_LOADED_A  = 0.20    # <= 200 mA at 6 V, vendor bench test
MOTOR_COUNT             = 2       # differential drive: two wheels, two motors

# COEFFICIENTS ARE COMPUTED AT THE SIMULATOR'S OPERATING POINT, NOT THE
# MOTOR'S MAXIMUM, because energy per metre is power divided by the speed
# actually travelled. Bound to the motion caps rather than restating 0.6
# and 1.2, so the two cannot drift apart -- the same reasoning that made
# SENSOR_MIN_PROGRESS share BYZANTINE_MIN_PROGRESS's value in Session 13.
MOTOR_NOMINAL_SPEED_MPS = MAX_LINEAR_SPEED_MPS     # 0.6 m/s
MOTOR_TURN_RATE_RAD_S   = MAX_ANGULAR_SPEED_RPS    # 1.2 rad/s

# ESP32 datasheet, Table 5-4 (Wi-Fi current consumption), 3.3 V, 25 C:
#   receive 802.11b/g/n draws 95-100 mA; transmit peaks at 240 mA (802.11b).
#   The robot broadcasts a heartbeat every HEARTBEAT_PERIOD_S and listens
#   the rest of the time, so it sits in RECEIVE mode almost always and the
#   240 mA transmit peak is a duty-cycle footnote, not the steady state.
#   0.12 A at 5 V covers the chip's 100 mA plus the DevKit board's AMS1117
#   linear regulator and USB-serial bridge, which are on the 5 V rail.
MCU_SUPPLY_V            = 5.0     # ESP32 DevKit board input, not the 3.3 V chip rail
MCU_ACTIVE_CURRENT_A    = 0.12    # 100 mA chip receive + board overhead

# RPLIDAR A1 datasheet, Figure 2-8 (typical current), work mode at 5 V:
#   scanner system 300 mA typical, motor system 100 mA typical.
#   Startup surge is 500 mA -- that is a battery-sizing figure, not a
#   steady-state one, and it is deliberately not used here.
LIDAR_SUPPLY_V          = 5.0
LIDAR_SCANNER_CURRENT_A = 0.300   # scanner system, work mode
LIDAR_MOTOR_CURRENT_A   = 0.100   # motor system, work mode

# --- Derived coefficients -- computed, never hand-edited ----------------
# KEEP THIS STRUCTURE. A single-robot prototype (HARDWARE_ONE_ROBOT.md)
# will measure E_DRIVE_J_PER_M, E_TURN_J_PER_RAD and P_COMPUTE_W with an
# INA219. When those land, the SPECIFICATIONS above are replaced and these
# four formulas are not touched -- and the datasheet figures become the
# cross-check rather than the source. P_SENSE_W stays a datasheet figure
# either way, because that build uses line sensors rather than a scanning
# LiDAR: say so plainly rather than implying it was measured.
E_DRIVE_J_PER_M  = (MOTOR_COUNT * MOTOR_SUPPLY_V * MOTOR_CURRENT_LOADED_A
                    / MOTOR_NOMINAL_SPEED_MPS)          # 2.4 W / 0.6 m/s = 4.0
E_TURN_J_PER_RAD = (MOTOR_COUNT * MOTOR_SUPPLY_V * MOTOR_CURRENT_LOADED_A
                    / MOTOR_TURN_RATE_RAD_S)            # 2.4 W / 1.2 rad/s = 2.0
P_COMPUTE_W      = MCU_SUPPLY_V * MCU_ACTIVE_CURRENT_A  # 5.0 * 0.12 = 0.60
P_SENSE_W        = LIDAR_SUPPLY_V * (LIDAR_SCANNER_CURRENT_A
                                     + LIDAR_MOTOR_CURRENT_A)  # 5.0 * 0.40 = 2.00

# If the arithmetic above ever stops producing these numbers, something has
# been hand-edited that should have been derived. Cheap to check, and it
# fails at import rather than silently in row 140 of a 2.6-hour suite.
assert abs(E_DRIVE_J_PER_M - 4.00) < 1e-9, E_DRIVE_J_PER_M
assert abs(E_TURN_J_PER_RAD - 2.00) < 1e-9, E_TURN_J_PER_RAD
assert abs(P_COMPUTE_W - 0.60) < 1e-9, P_COMPUTE_W
assert abs(P_SENSE_W - 2.00) < 1e-9, P_SENSE_W

# --- the fifth coefficient, derived in Session 15 from radio airtime -----
# WAS 0.05 AND WAS THE LAST INVENTED NUMBER IN THE ENERGY MODEL. CHANGED,
# and flagged per the convention. Session 14 left it as a placeholder
# because no datasheet gives joules per kilobyte directly. Nothing does --
# but the two figures that determine it are both in the SAME row of the
# SAME ESP32 table already quoted for P_COMPUTE_W, so no new part and no
# new datasheet is needed. Energy per kilobyte is transmit power times the
# time it takes to put a kilobyte on the air.
#
# ESP32 datasheet v5.3, Table 5-4, 3.3 V at 25 C:
#     Transmit 802.11b, DSSS 1 Mbps, POUT = +19.5 dBm ...... 240 mA
#     Receive  802.11b/g/n ................................. 95~100 mA
# The 1 Mbps DSSS rate is named in the transmit row itself, so the bit rate
# and the current come from the same measurement and cannot be mismatched.
# It is also the rate ESP-NOW uses, which is the protocol HARDWARE_SPEC.md
# specifies for the demonstrator.
ESP32_TX_CURRENT_A      = 0.240   # Table 5-4, transmit 802.11b DSSS 1 Mbps
ESP32_RX_CURRENT_A      = 0.100   # Table 5-4, receive, top of the 95~100 mA band
RADIO_BITRATE_BPS       = 1.0e6   # the DSSS 1 Mbps named in that same row
BYTES_PER_KB            = 1024    # payload sizes above are binary kilobytes

# CHARGE THE INCREMENT OVER RECEIVE, NOT THE WHOLE TRANSMIT CURRENT. The
# radio's 100 mA receive draw is ALREADY inside MCU_ACTIVE_CURRENT_A above
# ("100 mA chip receive + board overhead") and is billed continuously to
# the compute category for every second the robot is switched on. Billing
# the full 240 mA here as well would charge the listening baseline twice
# over for the duration of every transmission. What the radio actually
# costs extra when it talks is the difference.
RADIO_TX_EXCESS_CURRENT_A = ESP32_TX_CURRENT_A - ESP32_RX_CURRENT_A   # 0.140 A
RADIO_AIRTIME_S_PER_KB    = BYTES_PER_KB * 8 / RADIO_BITRATE_BPS      # 8.192 ms

# BILLED AT THE 5 V RAIL, for the same reason P_COMPUTE_W is. The extra
# current is drawn at the chip's 3.3 V rail, but it arrives through the
# DevKit's AMS1117, which is a LINEAR regulator -- it passes the current
# straight through and burns the voltage difference as heat, so the battery
# supplies that 140 mA at 5 V, not at 3.3 V. Using MCU_SUPPLY_V rather than
# restating 5.0 keeps this tied to the coefficient it must stay consistent
# with.
E_COMMS_J_PER_KB = (MCU_SUPPLY_V * RADIO_TX_EXCESS_CURRENT_A
                    * RADIO_AIRTIME_S_PER_KB)     # 5.0 * 0.140 * 0.008192

assert abs(E_COMMS_J_PER_KB - 0.0057344) < 1e-9, E_COMMS_J_PER_KB

# TWO CAVEATS, STATED RATHER THAN CORRECTED, because correcting either one
# would mean inventing a number the datasheet does not give. Both push the
# same way -- the figure above is a floor, not a best estimate:
#
#  1. Section 5.4 says "all transmitters' measurements are based on a 50 %
#     duty cycle", so 240 mA is an average over a half-on transmitter
#     rather than a continuous-transmit current. Taken at face value here.
#  2. The airtime counts PAYLOAD ONLY. A real 802.11b frame also carries a
#     PHY preamble and MAC headers, so true airtime per kilobyte is higher.
#
# Even the most pessimistic reading of the same table -- charge the full
# 240 mA and double-count receive -- gives 5.0 * 0.240 * 0.008192 =
# 0.0098 J/kB, still five times BELOW the old 0.05 placeholder. The
# placeholder cannot be rescued by either caveat; it was simply too big.

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
RNG_STREAM_FAULTS = 4       # which fault a seed gets in the suite
RNG_STREAM_PARTITION = 5    # how the round is divided between robots
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

# The distance at which the inspection is actually PERFORMED, as opposed to
# the distance at which the robot decides it has arrived. Used only to score
# whether a point was genuinely inspected -- never by a robot, which cannot
# know its own true position.
#
# WHY IT IS NOT THE SAME NUMBER AS THE ARRIVAL TOLERANCE. A healthy robot
# lands a mean 0.89 m from the point it believes it has reached, because
# odometry drift is real and the arrival test is made on the believed pose.
# Verify at 0.9 m and roughly half of all *healthy* arrivals fail their own
# test, which would report a fault-free squad as inspecting nothing.
#
# 2 m is the physically motivated figure: it is about the range at which a
# camera can read a gauge or a valve position, which is what an inspection
# point represents. A robot 2 m from a gauge has inspected it. A robot 6 m
# away -- one pipe-rack bay, the wrong-position fault -- has not.
INSPECTION_VERIFY_RADIUS_M = 2.0

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
# CHANGED 25.0 -> 35.0, DERIVED FROM THE SWEEP. This is an experimental
# parameter and every result moves with it, so here is the measurement
# (sweep_comms.py, 4 ranges x 3 seeds x 6 conditions):
#
#   range   in contact   detected   control FP   duration   energy
#    15 m      43.3 %      11/15         4         543 s    10650 J
#    25 m      68.8 %      13/15         1         502 s     9782 J
#    35 m      89.8 %      13/15         2         434 s     8662 J
#    50 m      98.5 %      14/15         3         407 s     8388 J
#
# WHY 35 m. The deciding column is "in contact", because Step 5 quarantines
# a robot only when two robots independently agree, and two robots that are
# rarely in contact can never compare notes. At 25 m that failed in
# practice, not just in principle: on seed 42 only one healthy robot ever
# accumulated enough shared ground to judge the faulty one at all, so the
# corroboration never arrived and recovery could not fire. At 35 m both
# healthy robots reach a verdict.
#
# It is not free and it is not a fudge. Duration and energy both IMPROVE
# (434 s and 8662 J against 502 s and 9782 J), because robots that can hear
# each other duplicate less work -- the longer radio pays for itself in
# mission cost. What it costs is false positives, which climb from 1 to 2
# on the healthy controls: more contact means more opportunity to accuse.
#
# WHY NOT 50 m. It buys 98.5 % contact, which is very nearly a fully
# connected squad -- and a fully connected squad is not the system this
# project claims to have built. The site diagonal is 97 m, so at 50 m
# almost every pair is always in range and "range-limited, decentralised"
# stops meaning anything. 35 m keeps roughly a tenth of the mission out of
# contact, which is enough for partial connectivity to remain a real
# condition the recovery mechanism has to survive.
COMMS_RANGE_M = 35.0

# Ranges compared by sweep_comms.py. The baseline above has to be argued
# for, not picked: Step 4 measured healthy robots spending 36-87 % of a
# mission unable to hear each other at 25 m, and quarantine in Step 5
# needs TWO robots to corroborate an accusation. Two robots that are out
# of contact two thirds of the time cannot corroborate anything, so the
# range determines whether the graded contribution is demonstrable at all.
COMMS_RANGE_SWEEP_M = (15.0, 25.0, 35.0, 50.0)
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

# ...AND THE LEASE ITSELF IS A FAULT-TOLERANCE MECHANISM, so it has to sit
# under the fault-tolerance flag rather than running in every condition.
#
# THIS FLAG GATES ALL THREE ROUTES BY WHICH ONE ROBOT'S UNFINISHED WORK
# CAN BECOME ANOTHER ROBOT'S RESPONSIBILITY. A naive system has none of
# them, and they were found one at a time:
#
#   1. a claim lapsing on a timeout              (found Session 9)
#   2. a cheaper bid taking a claim off its holder (found Session 9)
#   3. a robot giving up and telling the squad   (found Session 10)
#
# The third was left in C2 for a session because it looked like base
# mission behaviour. It is not: a robot that abandons a point and
# announces it has performed self-reported failure detection followed by
# task reallocation, which is exactly the capability under test. It is
# also what produced the immobilised inversion, where the naive condition
# beat the fault-tolerant one -- an immobilised robot gave up on every
# point it held and the healthy robots collected them, with fault
# tolerance switched off.
#
# The name says reallocation rather than claim expiry because expiry is
# only one of the three.
#
# This was measured the hard way. With expiry always on, a robot that died
# holding a claim had its work quietly picked up by whoever bid next --
# nobody detected anything, nobody decided anything, the auction simply
# re-let the lease. That is recovery, and it was running in BOTH arms of
# the C2/C3 comparison, which is why Session 8 measured fault tolerance
# moving the mission from 578/595 points to 579/595 and concluded it
# "barely changes the outcome". The mechanism was on both sides of the
# equals sign.
#
# With expiry off a claim is permanent: the robot that claimed point 17
# holds it until it visits it, and if it dies the point is never visited.
# That is the genuinely naive baseline, and it matches the hardware
# demonstrator, where fault tolerance off means the surviving robot
# finishes its own lane and stops.
REALLOCATION_ENABLED = True

# --- the round is divided up once, before anybody moves ---------------
# Every condition starts from the SAME assignment, computed from the seed
# alone. What differs between conditions is only what happens to that
# assignment when a robot fails.
#
# WHY A STATIC PARTITION REPLACED PERMANENT CLAIMS. Permanent claims were
# meant to make the naive condition naive, and they did something else:
# they starved the faulty robot of work. A displaced robot claimed one
# point, failed to reach it, kept the claim and idled for the rest of the
# round -- so it never filed a single false inspection and C2 scored a
# clean 39 believed / 39 truly. The baseline was being protected from the
# fault by an accident of the bidding rule.
#
# With a fixed assignment the displaced robot attempts all thirteen of its
# own points and files thirteen inspections it never performed, which is
# the damage the naive condition is supposed to take.
#
# It also removes a confound: under permanent claims the conditions
# diverged from step 0, because the bidding rule differed from step 0. Now
# they are identical until the fault fires.
#
# And it is the conventional system rather than a strawman -- Chapter 2's
# ANYmal deployment runs pre-defined inspection rounds, not a live
# auction. Objective iii is task REALLOCATION, so the baseline should be
# an allocation that is never revisited.
PARTITION_ITERATIONS = 12     # Lloyd iterations; converges long before this

# When a robot gives up on a point it says so, and the others take its word
# for it -- unless they are MUCH closer than it was. This is the fraction:
# a robot only re-attempts an abandoned point if its own cost is below this
# multiple of the cost the robot that failed had.
#
# 0.5 means "at least twice as close". Set it to 0 and a failure is treated
# as final for everybody, which is unsafe -- a robot can fail for reasons
# that are about where it happened to be standing rather than about the
# point. Set it to 1.0 and everyone re-attempts everything, which is the
# behaviour that made three robots slower than one on seed 2024.
GIVE_UP_OVERRIDE_FRACTION = 0.5

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


# =====================================================================
# 13. FAULT INJECTION  (Step 3)
# =====================================================================
# Five ways a robot can fail. They are implemented GENERICALLY -- the code
# knows "sensor degradation", not "condensate on the sensor window". The
# industrial cause of each goes in the write-up, and the translation table
# in CLAUDE.md is the bridge between the two. Keeping the causes out of the
# variable names is deliberate: the mechanism is what we simulate, and the
# mechanism is what we can defend.
#
# NOTHING HERE DETECTS ANYTHING. Step 3 is only about breaking a robot on
# purpose and being able to see the mission get worse. Detection is Step 4
# and recovery is Step 5; conflating them makes it impossible to report how
# long detection took, because the answer would be "instantly, we cheated".

# Which robot breaks, when, and how. Each entry is
#     (robot_id, step, fault_name)
# so a fault is addressable by robot AND by time, which is what the paired
# experimental design needs: seed 7 in C2 and C3 must break the same robot
# at the same instant, and only the response may differ.
#
# Empty by default. Conditions C0 and C1 inject no faults at all.
FAULT_INJECTIONS = []

FAULT_TYPES = (
    "sensor_degradation",   # noisy, short-sighted, dropping returns
    "wrong_position",       # believes it is somewhere it is not
    "comms_loss",           # radio dead, still drives and maps
    "immobilised",          # motors dead, still senses and relays
    "battery_drain",        # burns charge far faster than its work justifies
)

# --- sensor degradation ----------------------------------------------
# Feeds straight into the parameters sensors.py already accepts. Chosen so
# the damage is unmistakable without making the sensor useless: at 0.30 m
# noise a wall lands three cells either side of itself, which smears the
# map without erasing it.
FAULT_SENSOR_NOISE_STD_M = 0.30     # healthy is 0.02
FAULT_SENSOR_RANGE_SCALE = 0.45     # 8.0 m of reach becomes 3.6 m
FAULT_SENSOR_DROPOUT_PROB = 0.35    # a third of beams return nothing

# --- wrong position (the Byzantine one) -------------------------------
# Applied to the robot's BELIEVED pose, never to its sensor readings. That
# distinction is the whole point of this fault and it is worth being able
# to state at the defence: the robot's readings stay perfectly good and its
# map stays internally consistent, so nothing about it looks broken from
# the inside. It is simply drawing a correct map in the wrong place, and
# the only way to catch that is to compare it against robots that are not.
#
# 6.0 m is exactly the pipe-rack column spacing. The fault therefore models
# the real failure it is named after: the robot matches its scan to rack
# bay N+1 instead of bay N, because the two are geometrically identical.
# That is why the facility has 11 evenly spaced columns in the first place.
FAULT_POSE_OFFSET_M = (6.0, 0.0)
FAULT_POSE_OFFSET_RAD = 0.0         # pure translation. A heading error can
                                    # be added here, but one clean bay of
                                    # displacement is the clearer story.

# --- rapid battery drain ----------------------------------------------
# Multiplies every joule the robot spends. It does not change what the
# robot DOES, only how fast the charge disappears -- which is what makes it
# the one fault that can be caught before it takes effect, by comparing
# drain against work done. That is Step 4's problem, not this file's.
#
# WHY THE NUMBER IS THIS LARGE. An inspection round is cheap next to the
# battery: a squad robot finishes on about 92 % charge, having spent some
# 2,500 J of 33,000 J. A 6x drain therefore does nothing at all -- it ends
# the round on 67 % instead of 92 % and the mission is unchanged, which is
# a fault that cannot be observed and so cannot be recovered from either.
# To kill a robot part-way through a round the multiplier has to exceed
# roughly 33,000 / 2,500, and it has to do it on the fraction of the round
# that remains after the fault fires. 25x kills it with about a third of
# its assignment still outstanding, which is the case Step 5 must handle.
#
# If the energy coefficients change when the hardware measurements arrive,
# this number has to be revisited -- it is a ratio against them, not an
# independent quantity.
FAULT_BATTERY_DRAIN_MULTIPLIER = 25.0

# When demo_faults.py injects a fault, it does so here: far enough in that
# the squad is spread out and working, early enough that the damage has
# time to show.
FAULT_DEMO_STEP = 1200
FAULT_DEMO_ROBOT = 1

# --- WHEN the fault fires ---------------------------------------------
# Every fault used to fire at step 1200. That measures one point on a
# curve and reports it as the answer: a fault at 10 % of the round and a
# fault at 90 % are different experiments, and the difference is exactly
# what fault tolerance is for. An early fault leaves most of a robot's
# lane unvisited and should be devastating without reallocation and
# survivable with it; a late one costs almost nothing either way.
#
# So the step is drawn per seed, from the fault stream, and is therefore
# IDENTICAL in every condition that seed appears in. Paired seeding
# requires the fault to be a property of the seed, not of the arm.
#
# The fraction is of an EXPECTED round rather than the actual one,
# because the actual duration is not known until the round has been run
# -- and it depends on the condition, which would break the pairing. 5000
# steps is the measured middle of the range: healthy rounds finish in
# roughly 3,500 to 9,000 steps. So faults land between 500 and 2,500
# steps, which is 10-50 % of a typical round.
EXPECTED_MISSION_STEPS = 5000
FAULT_TIMING_MIN_FRACTION = 0.10
FAULT_TIMING_MAX_FRACTION = 0.50


# =====================================================================
# 16. THE EXPERIMENT SUITE  (Step 6)
# =====================================================================
# Six conditions x 30 seeds = 180 runs, one row each in results.csv.
#
# PAIRED SEEDING IS THE WHOLE DESIGN. Seed 7 produces the same inspection
# points, the same deviations, the same sensor noise and the same fault at
# the same step in every condition it appears in. Only the condition
# differs, so a difference between two rows with the same seed is caused
# by the condition and by nothing else. Never reseed per condition.
#
# 7, 42 and 2024 are included deliberately: they are the seeds every
# development session measured against, 2024 especially -- it is the long,
# awkward one with a point cut off by a deviation, and a suite that
# excluded it would flatter the system.
EXPERIMENT_SEEDS = tuple(range(1, 29)) + (42, 2024)

# Which fault a seed gets. Drawn from its own stream so that adding or
# removing a condition cannot shift it, and fixed per seed so that C2, C5,
# C3 and C4 all break the same way at the same moment.
#
# The robot and the step are held constant rather than drawn. Varying them
# too would spread 30 seeds across three nuisance dimensions at once and
# leave roughly two runs per cell -- enough to add variance, nowhere near
# enough to measure anything with.
EXPERIMENT_FAULT_STEP = FAULT_DEMO_STEP
EXPERIMENT_FAULT_ROBOT = FAULT_DEMO_ROBOT

RESULTS_CSV = "results.csv"
TRACE_DIR = "traces"


# =====================================================================
# 14. FAULT DETECTION  (Step 4)
# =====================================================================
# One detector per fault. Every one of them runs on a PEER's evidence, not
# on the robot itself: a robot cannot be trusted to notice that it is the
# broken one, and the wrong-position fault proves it -- that robot's own
# readings are perfect and its map is internally consistent.
#
# Detection only. Nothing here quarantines, rolls back or reallocates;
# that is Step 5. Keeping them apart is what makes detection latency a
# number we can report rather than an assumption.

# --- heartbeats -------------------------------------------------------
# How often a robot describes itself to whoever is listening. Every 5 s.
# Faster costs comms energy for little gain; slower coarsens every latency
# we report, since a fault cannot be noticed before the next heartbeat.
HEARTBEAT_EVERY_N_STEPS = 50
COMMS_HEARTBEAT_PACKET_KB = 0.2    # a pose, two energies, four counters

# --- how often the detectors run --------------------------------------
# There is no point checking faster than the evidence arrives. Heartbeats
# come every 50 steps, so running the cheap detectors every 25 costs
# nothing and never delays a finding by more than 2.5 s.
#
# The Byzantine check is different and has to be throttled hard. It
# reconstructs two full 440,000-cell beliefs and compares them, per peer,
# and running that every step made an 18-run measurement take over forty
# minutes -- the kind of 3x-slower change CLAUDE.md says to treat as a
# problem rather than a detail. Every 100 steps costs 1/100th of that and
# coarsens the reported latency by at most 10 s, which is nothing beside
# the ~170 s this fault takes to become visible at all.
DETECTOR_EVERY_N_STEPS = 25
BYZANTINE_CHECK_EVERY_N_STEPS = 100


# =====================================================================
# 15. RECOVERY  (Step 5)
# =====================================================================
# What the squad DOES about a robot it has decided is faulty. Set
# RECOVERY_ENABLED = False for condition C2, the naive baseline that
# detects nothing and acts on nothing.

RECOVERY_ENABLED = True

# TWO ACCUSERS BEFORE ANYBODY IS QUARANTINED. NEVER ONE.
#
# This is the most important number in this section and it is not a
# tuning parameter. A robot whose radio has died hears silence from
# everybody and concludes that everybody has failed -- and from its own
# evidence that inference is *correct*. Step 4 measured it happening. Act
# on a single accusation and the one genuinely broken robot quarantines
# the two healthy ones.
#
# Step 4 also measured the Byzantine margin as thin enough that a single
# map comparison is not safe evidence on its own. Requiring corroboration
# from an independent robot is what makes an accusation actionable, and it
# is the whole reason the squad has three robots rather than two.
# ...but only for the DESTRUCTIVE action. Session 8 found that gating
# everything on two accusers had a consequence nobody intended: comms loss
# could never be acted on at all. Detection runs at 2/3, but two accusers
# requires both healthy robots to reach the verdict independently, and
# they rarely can -- so the rule that protects an isolated robot from
# being quarantined also guaranteed its work was never reallocated.
#
# The two actions do not carry the same risk, so they should not demand
# the same evidence:
#
#   quarantine + rollback   destructive. Discards a robot's observations
#                           and cannot be undone. Two accusers.
#   claim release           cheap. Worst case is that two robots drive to
#                           the same inspection point and one wastes a
#                           journey. One accuser.
#
# Acting wrongly on a claim release costs a detour. Acting wrongly on a
# quarantine throws away a healthy robot's map.
RECOVERY_QUARANTINE_ACCUSERS = 2
RECOVERY_REALLOCATE_ACCUSERS = 1

# How much a degraded robot's map is still worth. NOT zero: a noisy sensor
# still sees walls, it just sees them badly, and throwing the data away
# entirely loses real coverage. Down-weighting is the proportionate
# response and log-odds makes it a multiplication.
RECOVERY_SENSOR_TRUST = 0.25

# --- what to do about an isolated robot: NOTHING ----------------------
# CHANGED IN SESSION 14 from True, and flagged per the convention. It is
# the only behavioural change of that session, and it is justified by
# measurement rather than by argument.
#
# The 174-run suite measured points_truly_visited as IDENTICAL across C2,
# C5 and C3 on comms_loss -- 39.60 in all three -- while C3 spent 265 J
# per point against C2's 213. Reallocating an isolated robot's lane
# therefore cost 24 % more energy for exactly zero coverage benefit.
#
# The reason is that the premise was wrong. Reallocation exists because a
# robot that cannot finish its work needs somebody else to do it. A robot
# with a dead radio finishes its own lane perfectly well: it is not
# broken, it is alone. What it cannot do is TELL anyone, and taking its
# work away does not fix that -- it just makes two healthy robots re-cover
# ground that was already being covered.
#
# Session 13 recorded the same thing from the other side: the arm that
# responded correctly measured worse on completeness than the arm that did
# nothing. That was read as an instrument problem. With the energy column
# beside it, it is better read as the response being wrong.
#
# So comms loss is now OBSERVED AND LOGGED ONLY. Detection still runs and
# is still scored; no claim is released, nothing is quarantined and no
# trust is changed. The isolated robot's map merges on reconnect exactly
# as before, which is the mechanism that was always doing the real work.
#
# Kept as a flag rather than deleted, because Session 14 needs the old
# behaviour to separate this change from the coefficient change: C3 is the
# only condition that runs recovery at all, so re-running C3 with this set
# True is what isolates the effect.
RECOVERY_COMMS_LOSS_REALLOCATE = False

COMMS_SUSPICION_PACKET_KB = 0.05   # a suspect id and a fault name

# --- comms loss, and the trap it sets ---------------------------------
# A healthy robot behind a storage tank goes quiet exactly like a failed
# one. Silence is only evidence when the robot SHOULD have been heard.
#
# So each robot predicts where its peers are, from the last pose they
# broadcast plus how far they could possibly have travelled since. If the
# CLOSEST the peer could now be is still beyond COMMS_RANGE_M, silence is
# expected and no timeout runs.
#
# That prediction decays: MAX_LINEAR_SPEED_MPS times elapsed time grows
# without limit, so after long enough the peer could be anywhere and the
# estimate justifies nothing. Past PEER_POSITION_DECAY_S we stop trusting
# it and fall back to the hard timeout, which is deliberately long.
COMMS_SILENCE_TIMEOUT_S = 25.0       # silence from a peer that should be
                                     # in range -- five missed heartbeats
PEER_POSITION_DECAY_S = 60.0         # after this, a position estimate is
                                     # too stale to say anything, and the
                                     # hard timeout takes over

# The conservative backstop: silence from anyone, wherever we think they
# are. MEASURED, not chosen. On healthy runs a pair of robots stays
# genuinely out of radio contact for as long as 678 s (seed 2024) -- they
# spend 36-87 % of a mission unable to hear each other, because a 25 m
# radio on an 80 x 55 m site is a short radio. Any hard timeout below that
# accuses healthy robots, and false-positive rate is a reported number.
#
# At 900 s this rarely fires inside a mission, and that is the honest
# position: a robot that has been beyond the horizon for the whole round
# genuinely cannot be told apart from one that has failed. The detector
# that does the real work is the gated short timeout, which fires when a
# peer is predicted to be back within earshot and still says nothing.
COMMS_SILENCE_HARD_TIMEOUT_S = 900.0

# How far inside the radio range a peer must be PREDICTED to be before its
# silence counts as evidence. Prediction is dead reckoning on somebody
# else's behalf and it is wrong by metres, so judging at exactly the range
# boundary turns that error straight into false accusations -- every one
# of the first batch read "predicted 25.0 m away", which is the boundary
# to the centimetre.
COMMS_PREDICTION_MARGIN_M = 5.0

# ...and the margin GROWS with the length of the silence, because the
# prediction gets worse the longer we have been guessing. A peer is
# predicted along the straight line to the point it announced, but it
# drives a real route that detours round buildings, so it is usually
# somewhere short of where we put it. On seed 42 that placed a healthy
# robot at its destination 20 m away -- comfortably in range by a fixed
# margin -- while it was really still behind the store and out of earshot.
#
# Half the top speed, as a per-second allowance for the peer being
# somewhere other than on our predicted track. Combined with the decay
# cut-off this leaves a usable window: silence counts as evidence between
# 25 s and 60 s, and only from a peer predicted well inside range.
PEER_PREDICTION_DRIFT_MPS = 0.3

# Assumed peer speed when we have only one heartbeat and cannot measure
# its ground speed. A fraction of the maximum, because a robot that has
# declared a destination is usually driving toward it. Assuming it stayed
# put instead is the pessimistic choice here: it predicts the peer is
# still nearby, which is what manufactures a false accusation.
PEER_PREDICTION_SPEED_FRACTION = 0.8

# --- sensor degradation -----------------------------------------------
# MEASURED, and the measurement changed the design. Over a full mission
# the victim robot's own reports were:
#
#                        healthy                degraded
#   valid-return ratio   0.35-0.96, med 0.62    0.04-0.96, med 0.27
#   range variance       1.5-10.2,  med 6.8     0.1-10.2,  med 0.8
#
# Two things to notice. The ranges overlap almost entirely, so NEITHER
# channel discriminates on its own -- a healthy robot in open ground gets
# few returns too, because there is nothing out there to hit. And the
# variance moves DOWNWARD, not up: losing range squashes every reading
# toward the same short cap. A threshold looking for high variance would
# have been the wrong way round.
#
# So the test is the conjunction: few returns AND uniform ranges, which
# open ground does not produce, plus persistence across several reports.
# Tightened after the first pass produced false positives: healthy robots
# reached valid=0.34 with variance=2.33 for three reports running, which
# the looser pair (0.45 / 2.5) convicted. The degraded robot sits at
# valid 0.25-0.26 with variance 0.51-0.79, so there is room below the
# healthy floor -- but not much, and the variance channel is doing most of
# the discriminating.
SENSOR_VALID_RATIO_MIN = 0.30
SENSOR_RANGE_VARIANCE_MIN = 1.5

# CHANGED IN SESSION 13: 3 -> 5. Flagged rather than slipped in, because it
# is an experimental parameter and it moves results.
#
# Why: M4. This detector never received the persistence and progress gating
# the Byzantine detector was given in Session 11, and it showed. On seed 7,
# robot 1 was accused of a degraded sensor at step 400 in all five runs for
# a fault injected at step 1076 -- scored as a correct detection only
# because the suspect happened to be the eventual victim. Five consecutive
# bad reports is the same shape of rule as BYZANTINE_MIN_CHECKS = 5.
#
# What it costs: detection latency on a genuine fault, roughly two extra
# heartbeats, ~25 s. That is the right thing to spend here -- an accusation
# made before the fault exists is worse than a late one, because it flatters
# the detector twice: it counts a hit AND hides a miss.
SENSOR_MIN_HEARTBEATS = 5            # five bad reports running, so one
                                     # awkward corner is not an accusation

# The other half of M4, and a new parameter rather than a changed one.
# Deliberately the same value as BYZANTINE_MIN_PROGRESS: the argument is
# identical, and two numbers that mean the same thing should not be able to
# drift apart. Early in a round the squad is still on the perimeter road,
# which produces exactly the signature this detector looks for -- few
# returns, all of them uniformly distant. Open ground mimicking a degraded
# sensor is a known limitation of this test (Session 7), and refusing to
# judge until the round is a quarter done is what stops it becoming an
# accusation.
SENSOR_MIN_PROGRESS = 0.25

# --- wrong position ---------------------------------------------------
# THE ONE THAT NEEDS THREE ROBOTS. A pairwise disagreement tells you that
# one of the two is wrong, not which. With a third robot the odd one out
# is the one that disagrees with BOTH of the others while they agree with
# each other.
#
# Compared over each robot's OWN observations, never the merged map --
# merging puts the faulty robot's data inside everyone else's before any
# comparison can run, and Session 6 measured the merged maps *converging*
# under this fault. See CLAUDE.md.
# MEASURED AS A RATE, NOT A COUNT, and that change is what made the
# detector work. Conflict in raw cells does not separate the conditions at
# all -- healthy pairs produced 62 to 540 conflicting cells and a displaced
# robot produced 115 to 812, straight through each other. The overlap
# between two robots' territories varies by a factor of two between pairs,
# so a count measures how much ground they happen to share as much as it
# measures how much they disagree about it.
#
# As a fraction of the cells both have actually observed (seed 42):
#     healthy pairs        0.21 - 1.17 %
#     displaced robot      1.78 - 1.97 %   (the pairs involving the victim)
#     the other pair       0.20 - 0.22 %   (still agreeing with each other)
#
# Healthy conflict is never zero: two robots integrate the same wall at
# their own drifted poses, so they disagree about its edges. That ~1 % is
# the noise floor the fault has to clear, and it is why Session 2 reduced
# odometry drift -- at the old drift this detector would have no signal at
# all.
# Tracked over whole missions, the two distributions very nearly touch:
#
#     healthy, seed 42     up to 1.15 %
#     healthy, seed 2024   up to 1.51 %      <-- the ceiling
#     displaced robot      1.69 % upward, reaching 2.01 %
#
# THE MARGIN IS THIN AND THAT IS THE RESULT, not a tuning problem. Seed
# 2024 is a long, collision-heavy run, and healthy odometry drift over 800 s
# produces very nearly as much cross-agent conflict as displacing a robot
# by a whole pipe-rack bay. Session 2 predicted exactly this when it cut
# the drift constants: "drift in healthy robots is the noise floor for the
# cross-agent Byzantine detector." This is that floor, measured.
#
# Consequences worth stating in Chapter 4: a smaller displacement than 6 m
# would not be separable at all, and Step 5 should require corroboration
# from more than one robot before quarantining anybody.
# NORMALISED PER RUN, WHICH IS WHAT MADE THIS DETECTOR WORK.
# Absolute thresholds cannot separate the two conditions, because each
# seed has its own drift level: seed 2024's healthy squad sits at 1.4-1.5 %
# conflict, which is higher than seed 42's squad manages WITH a robot
# displaced by a whole pipe-rack bay. A fixed number is asking the wrong
# question.
#
# The right question is comparative, and every robot can already answer it
# alone. A robot's ledger holds all three contributions -- its own and both
# peers' -- so it can compute all three pairwise conflict rates without
# sending a single extra message. Then:
#
#     is the suspect's mean disagreement with the other two much larger
#     than the disagreement between those two?
#
# That divides this run's drift out of the comparison. Measured over three
# seeds, end of mission:
#
#     healthy, above the floor     ratio 0.93 - 1.08
#     displaced robot              ratio 1.66 - 5.74
#     a faulty robot judging others      0.19 - 1.23
#
# The last line matters: a displaced robot disagrees with BOTH peers, so
# its own baseline is high and its ratios stay low. It accuses nobody,
# which is the correct behaviour and falls out of the arithmetic rather
# than having to be special-cased.
BYZANTINE_RATIO = 1.4                # between the healthy ceiling (1.08)
                                     # and the faulty floor (1.66)
BYZANTINE_MIN_MEAN_RATE = 0.008      # ...but only once the disagreement is
                                     # big enough to mean anything. A
                                     # healthy pair on seed 7 reached a
                                     # ratio of 2.23 on a mean disagreement
                                     # of 0.29 % -- a ratio between two
                                     # negligible numbers is noise, not
                                     # evidence.

# The reference pair must have a measurable disagreement of its own before
# it can be used as a yardstick. Every false positive in the first pass
# divided by a baseline of 0.07-0.08 % -- two robots agreeing almost
# perfectly because they had not yet covered enough common ground for
# their agreement to mean anything, which makes any ratio against it
# enormous. Every true detection had a baseline of 0.22 % or more.
BYZANTINE_MIN_BASELINE_RATE = 0.0015
BYZANTINE_MIN_OVERLAP_CELLS = 30000  # ~300 m2 of genuinely shared ground.
                                     # Raised from 15,000 in Session 11.
                                     # Below this the rate is a small-sample
                                     # artefact: control runs convicted
                                     # healthy robots 10 s into the mission,
                                     # on an overlap of a few hundred cells
                                     # at the fringe of both sensors.
                                     # SUPERSEDED IN SESSION 13 by the floor
                                     # below. Kept because it records why a
                                     # floor is needed at all, and because
                                     # contribution_conflict() -- which it
                                     # guarded -- is still the method the
                                     # Chapter 4 explanation is written
                                     # against.

# THE SAMPLE THE BYZANTINE TEST ACTUALLY RUNS ON, SINCE SESSION 13.
# M3: all three pairwise rates are now measured over the cells all three
# robots have seen, so that the ratio compares like with like. That region
# is necessarily smaller than any one pair's overlap, so it needs its own
# floor rather than inheriting the pairwise one.
#
# Measured over 575 checks on six healthy C1 seeds (the probe is recorded in
# Session 13's log):
#
#     triple-overlap size   p10  8,223   p25 13,366   median 15,812
#     pairwise overlap      median 31,969
#
# so the common region runs about half a pair's overlap, and the old 30,000
# would admit only 10 % of checks -- a detector that is silent for nine
# tenths of the round is safe in the same way an unplugged one is.
#
# 10,000 admits 87 % of checks while excluding the thin tail below p25. The
# false-positive work is not being done by this number: on healthy C1 seeds
# the detector's full rule flags 21 judgements of 187 admitted checks on the
# pairwise basis and 1 of 575 on the triple basis, and that holds for every
# floor from 0 to 30,000. The floor is here to stop a small sample, not to
# carry the fix.
BYZANTINE_MIN_TRIPLE_OVERLAP_CELLS = 10000

# THESE THREE GATES EXIST BECAUSE THE DETECTOR WAS OVERFITTED.
# Tuned on three seeds it scored 3/3 with no false positives, and on the
# wider sample it quarantined the wrong robot in 5 of 15 runs and cost a
# healthy squad 14 good inspections. Three seeds is not a sample, it is an
# anecdote, and every number downstream inherited the error.
#
# All three ask the same question in different ways: is this disagreement
# large enough, sustained enough, and measured on enough evidence to be
# worth destroying a robot's work over?
BYZANTINE_MIN_CHECKS = 5             # sustained across five checks, 50 s.
                                     # Transient drift does not survive
                                     # that; a 6 m displacement does,
                                     # because it does not go away. Costs
                                     # detection latency, which is the
                                     # right thing to spend here.
BYZANTINE_MIN_PROGRESS = 0.25        # ...and not before a quarter of the
                                     # round is done. Early on two robots
                                     # have covered almost no common
                                     # ground, so the conflict rate is a
                                     # ratio between two tiny numbers.
                                     # Measured as the fraction of points
                                     # the accuser believes are finished --
                                     # a robot knows that about itself,
                                     # whereas it cannot know how long the
                                     # mission will turn out to be.

# --- immobilised ------------------------------------------------------
# Asking for motion and not moving. Both odometers are in the heartbeat,
# so a peer compares the commanded distance against the achieved one.
IMMOBILE_COMMANDED_M = 1.5           # must have asked for at least this
                                     # much WHEEL travel (distance plus
                                     # rotation converted at the radius)
IMMOBILE_ACHIEVED_FRACTION = 0.10    # and delivered less than this much
                                     # of it, between two heartbeats
IMMOBILE_MIN_HEARTBEATS = 3          # for three reports running. A robot
                                     # grinding against a wall also
                                     # commands motion it does not achieve,
                                     # but only until the escape behaviour
                                     # reverses it out a second or two
                                     # later. Dead motors do not recover.

# --- battery drain, the predictive one --------------------------------
# Two separate signals, and the second is the interesting one.
#
# 1. RATE. A healthy robot's battery falls by exactly the energy it spends,
#    so charge_used / energy_spent is 1.0. The fault multiplies it. Above
#    this ratio something is wrong with the cell, not with the work.
BATTERY_DRAIN_RATIO_MAX = 2.0
#
# 2. PROJECTION. Estimate what the robot's remaining assignment will cost
#    from what its work has cost so far, and compare against the charge it
#    has left. If it cannot finish, say so WHILE IT IS STILL ALIVE. Every
#    other detector here reports a robot that has already failed; this one
#    reports a robot that is going to, which is the difference between
#    reallocating its work and losing it.
BATTERY_PROJECTION_MARGIN = 1.15     # flag when the projected need exceeds
                                     # the remaining charge by this factor
BATTERY_MIN_POINTS_DONE = 2          # need some history before projecting

