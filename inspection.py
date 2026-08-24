"""
inspection.py
=============
The mission expressed as a list of PLACES TO VISIT rather than an area to
cover. Design Change 01, section 3.

WHY THE MISSION IS DISCRETE
---------------------------
"Cover as much area as possible" is hard to report and harder to
reallocate: when a robot fails, its share of an area is a fuzzy thing to
hand to somebody else. A list of inspection points is not. Chapter 4 can
say "robot 2 failed with 6 of its 11 points done, and points 7-11 were
reallocated in 4.2 s" -- a countable claim about a discrete quantity.

It also matches the hardware demonstrator, which already drives taped
lanes with numbered stations. Zones in simulation, lanes in hardware;
inspection points in simulation, tick-marked stations in hardware.

STRATIFICATION
--------------
The facility divides into 11 functional zones (config.ZONES). Every zone
gets between 2 and 5 points and the site gets exactly 40, so no seed can
produce a mission where all the work sits in one corner. Positions inside
a zone are random and seeded, so each seed is a different mission of
comparable difficulty.

THE THREE PLACEMENT RULES, AND WHY EACH ONE EXISTS
--------------------------------------------------
  1. CLEARANCE  -- the robot must be able to stand on the point. A point
                   inside a pump is not an inspection point.
  2. SEPARATION -- two points 30 cm apart are one point pretending to be
                   two, and would flatter the "points visited" metric.
  3. REACHABLE  -- the point must be routable from the start position on
                   the DOCUMENTED layout, using the same planner the robot
                   will use. A point the drawings already say is
                   unreachable is a mistake in the mission, not a finding.
                   (Points made unreachable later, by a deviation, ARE a
                   finding -- see deviations.py.)
"""

import numpy as np

import config
from planner import WavefrontPlanner


class MapView:
    """
    A fully-confident belief built from a known 0/1 grid.

    The planner only ever reads `.L`, so this is all it needs to route over
    a grid we already know -- the documented layout, or ground truth. It
    exists so that "is this point reachable?" is answered by exactly the
    same traversability rules the robot will use in the mission (0.4 m
    coarse cells, 0.8 m obstacle inflation) rather than by a second, subtly
    different definition of reachable.

    It is an analysis tool. No robot is ever given one.
    """

    def __init__(self, grid_array):
        self.L = np.where(grid_array == 1,
                          config.LOG_ODDS_CLAMP,
                          -config.LOG_ODDS_CLAMP).astype(np.float32)


class Zone:
    """One functional area of the facility, as an axis-aligned rectangle."""

    def __init__(self, code, name, x0, y0, x1, y1):
        self.code = code
        self.name = name
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1

    @property
    def centre(self):
        return (0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1))

    @property
    def area_m2(self):
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def contains(self, x, y):
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def __repr__(self):
        return f"<Zone {self.code} {self.name}>"


class InspectionPoint:
    """
    One place the squad has been sent to look at -- a valve, a gauge, a
    flange, a pump seal.

    The mission-state flags live here because the mission IS this list:
    `points_visited / points_total` is the headline number, and
    `unreachable` is reported separately from ordinary failure so that a
    scaffolded-over valve does not count against the squad the way an
    unvisited reachable point does.
    """

    def __init__(self, index, zone_code, x, y):
        self.index = index
        self.zone_code = zone_code
        self.x = float(x)
        self.y = float(y)

        self.visited = False
        self.visit_step = None      # when the robot declared it visited
        self.visited_by = None      # which robot got there (squad missions)
        self.visit_error_m = None   # true distance at that moment -- the
                                    # honest check on navigating by odometry
        self.unreachable = False    # ground truth says no route exists
        self.abandoned = False      # the robot gave up trying

    @property
    def xy(self):
        return (self.x, self.y)

    def __repr__(self):
        state = "visited" if self.visited else (
            "unreachable" if self.unreachable else "pending")
        return f"<P{self.index:02d} {self.zone_code} ({self.x:.1f},{self.y:.1f}) {state}>"


# ---------------------------------------------------------------------
def build_zones():
    """The 11 zones, straight out of config."""
    return [Zone(*row) for row in config.ZONES]


def reachability_planner(facility):
    """
    A planner tuned to answer "can a robot physically get there?" rather
    than "what route should it drive?".

    Same wavefront, less obstacle inflation -- see
    config.REACHABILITY_INFLATE_CELLS for why the two differ.
    """
    return WavefrontPlanner(facility,
                            inflate_cells=config.REACHABILITY_INFLATE_CELLS)


def reachable_field(planner, grid_array, start_xy):
    """
    BFS travel distance from `start_xy` to every coarse cell of a KNOWN
    grid. -1 means no route exists.

    Used for two questions that must be answered the same way:
      * during placement -- can the robot get to this candidate point?
      * after injecting deviations -- has a deviation cut a point off?
    """
    view = MapView(grid_array)
    blocked = planner._coarse_blocked(view)
    sr, sc = planner._to_coarse(*start_xy)
    if blocked[sr, sc]:
        snapped = planner._nearest_free(~blocked, sr, sc)
        if snapped is None:
            raise RuntimeError("start position is not on a traversable cell")
        sr, sc = snapped
    return planner._flood(blocked, [(sr, sc)])


def is_reachable(planner, dist_field, x, y):
    """
    Is (x, y) inside a coarse cell the flood managed to reach?

    The strict test: can the robot stand exactly here. Used when PLACING
    points, because a point the robot cannot stand on is a bad point.
    """
    r, c = planner._to_coarse(x, y)
    if not (0 <= r < planner.c_rows and 0 <= c < planner.c_cols):
        return False
    return bool(dist_field[r, c] >= 0)


def inspection_distance(planner, dist_field, x, y, radius_m=None):
    """
    Travel distance to the nearest cell this point can be INSPECTED from,
    in coarse cells, or None if no such cell can be reached.

    A DIFFERENT QUESTION FROM is_reachable, and the distinction matters.
    The robot counts a point as visited once it believes it is within
    INSPECTION_REACHED_M of it, so a gauge with scaffolding erected on top
    of it can still be inspected from just outside the scaffolding. Only
    when no reachable cell lies within inspection range is the point
    genuinely cut off.

    Getting this wrong produced a contradictory report -- 40 of 40 points
    visited, and one of them simultaneously listed as unreachable.
    """
    if radius_m is None:
        radius_m = config.INSPECTION_REACHED_M
    r, c = planner._to_coarse(x, y)
    cell_m = planner.ds * planner.res
    span = int(np.ceil(radius_m / cell_m))

    best = None
    for rr in range(r - span, r + span + 1):
        for cc in range(c - span, c + span + 1):
            if not (0 <= rr < planner.c_rows and 0 <= cc < planner.c_cols):
                continue
            d = int(dist_field[rr, cc])
            if d < 0:
                continue
            wx, wy = planner._to_world(rr, cc)
            if np.hypot(wx - x, wy - y) <= radius_m:
                if best is None or d < best:
                    best = d
    return best


def can_be_inspected(planner, dist_field, x, y, radius_m=None):
    """Can the robot get close enough to inspect this point at all?"""
    return inspection_distance(planner, dist_field, x, y, radius_m) is not None


# ---------------------------------------------------------------------
def _allocate_counts(rng, n_zones):
    """
    Decide how many points each zone gets: a random 2-5 each, then nudged
    until the total is exactly INSPECTION_POINTS_TARGET.

    Why fix the total rather than just take whatever the draws give: the
    number of points is the denominator of the headline metric
    (points_visited / points_total). If it moved between seeds, a mission
    that visited 35 of 38 would look better than one that visited 37 of 40,
    which is nonsense. Randomise the positions, hold the workload fixed.
    """
    lo, hi = config.INSPECTION_MIN_PER_ZONE, config.INSPECTION_MAX_PER_ZONE
    target = config.INSPECTION_POINTS_TARGET
    if not (n_zones * lo <= target <= n_zones * hi):
        raise ValueError(
            f"INSPECTION_POINTS_TARGET={target} is not reachable with "
            f"{n_zones} zones holding {lo}..{hi} points each")

    counts = [int(c) for c in rng.integers(lo, hi + 1, size=n_zones)]

    # Walk the zones in a shuffled order so the correction does not always
    # fall on Z1. The order is drawn from the seeded stream, so it is
    # reproducible.
    order = [int(i) for i in rng.permutation(n_zones)]
    i = 0
    while sum(counts) != target:
        z = order[i % n_zones]
        i += 1
        if sum(counts) < target and counts[z] < hi:
            counts[z] += 1
        elif sum(counts) > target and counts[z] > lo:
            counts[z] -= 1
    return counts


def generate_inspection_points(facility, seed=config.DEFAULT_SEED,
                               planner=None, verbose=False):
    """
    Place the mission's inspection points on the DOCUMENTED layout.

    Placed before deviations are injected, and against the documented grid,
    because that is the order things happen in reality: an inspection round
    is planned from the drawings, and only then does the robot go out and
    discover that the drawings are out of date.

    `planner` is the REACHABILITY planner (see reachability_planner), not
    the one the robot navigates with.

    Returns a list of InspectionPoint, ordered by zone.
    """
    rng = np.random.default_rng([seed, config.RNG_STREAM_INSPECTION])
    if planner is None:
        planner = reachability_planner(facility)

    zones = build_zones()
    counts = _allocate_counts(rng, len(zones))

    dist = reachable_field(planner, facility.documented_grid,
                           config.START_POSE_XY)

    points = []
    sep2 = config.INSPECTION_MIN_SEPARATION_M ** 2

    for zone, wanted in zip(zones, counts):
        placed_here = 0
        for _ in range(wanted):
            for _attempt in range(config.INSPECTION_PLACEMENT_ATTEMPTS):
                x = float(rng.uniform(zone.x0, zone.x1))
                y = float(rng.uniform(zone.y0, zone.y1))

                # 1. the robot must fit here
                if not facility._has_clearance(x, y, config.INSPECTION_CLEARANCE_M):
                    continue
                # 2. not on top of a point we already placed
                if any((x - p.x) ** 2 + (y - p.y) ** 2 < sep2 for p in points):
                    continue
                # 3. the drawings must offer a route to it
                if not is_reachable(planner, dist, x, y):
                    continue

                points.append(InspectionPoint(len(points), zone.code, x, y))
                placed_here += 1
                break
            else:
                # Rejection sampling gave up. Report it rather than silently
                # shipping a smaller mission -- a zone that cannot hold its
                # quota is a fact about the layout worth knowing.
                if verbose:
                    print(f"  [warning] {zone.code} {zone.name}: could not "
                          f"place point {placed_here + 1} of {wanted}")

    # Renumber so the indices are contiguous even if a zone fell short.
    for i, p in enumerate(points):
        p.index = i
    return points


def points_by_zone(points):
    """{zone_code: [InspectionPoint, ...]} -- for reporting per zone."""
    out = {}
    for p in points:
        out.setdefault(p.zone_code, []).append(p)
    return out


def zone_coverage_fraction(facility):
    """
    Fraction of the facility's free area that falls inside some zone.

    Reported once, in the verification output, so "11 zones covering the
    facility" is a measured statement rather than an assertion.
    """
    inside = np.zeros((facility.n_rows, facility.n_cols), dtype=bool)
    for z in build_zones():
        r0, c0 = facility.world_to_grid(z.x0, z.y0)
        r1, c1 = facility.world_to_grid(z.x1, z.y1)
        inside[max(0, r0):r1 + 1, max(0, c0):c1 + 1] = True
    free = facility.grid == 0
    return float((inside & free).sum() / free.sum())


# =====================================================================
# Verification: python inspection.py
# =====================================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from environment import Facility

    facility = Facility()
    planner = reachability_planner(facility)
    points = generate_inspection_points(facility, config.DEFAULT_SEED,
                                        planner, verbose=True)

    zones = build_zones()
    grouped = points_by_zone(points)

    print("\n" + "=" * 62)
    print("  ZONE DECOMPOSITION AND INSPECTION POINTS")
    print(f"  seed {config.DEFAULT_SEED}")
    print("=" * 62)
    print(f"  {'zone':<5s} {'name':<28s} {'area m2':>8s} {'points':>7s}")
    print("-" * 62)
    for z in zones:
        n = len(grouped.get(z.code, []))
        print(f"  {z.code:<5s} {z.name:<28s} {z.area_m2:8.0f} {n:7d}")
    print("-" * 62)
    print(f"  {'TOTAL':<34s} {'':>8s} {len(points):7d}")
    print(f"  Free area inside a zone      : "
          f"{zone_coverage_fraction(facility)*100:5.1f} %")
    print(f"  Minimum separation achieved  : "
          f"{min(np.hypot(a.x - b.x, a.y - b.y) for i, a in enumerate(points) for b in points[i+1:]):5.2f} m")
    print("=" * 62 + "\n")

    # --- determinism check: the same seed must give the same points ------
    again = generate_inspection_points(facility, config.DEFAULT_SEED, planner)
    identical = (len(again) == len(points) and
                 all(a.x == b.x and a.y == b.y and a.zone_code == b.zone_code
                     for a, b in zip(again, points)))
    other = generate_inspection_points(facility, config.DEFAULT_SEED + 1, planner)
    differs = any(a.x != b.x for a, b in zip(other, points))
    print(f"  same seed reproduces exactly : {identical}")
    print(f"  a different seed moves them  : {differs}\n")

    # --- figure ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 9.5))
    ext = [0, facility.width_m, 0, facility.height_m]
    ax.imshow(facility.render_rgb(dim_free=True), origin="lower", extent=ext,
              interpolation="nearest")

    for z in zones:
        ax.add_patch(Rectangle((z.x0, z.y0), z.x1 - z.x0, z.y1 - z.y0,
                               fill=False, ec="#1D4E89", lw=1.4, ls="--",
                               alpha=0.9))
        ax.text(z.x0 + 0.3, z.y1 - 1.0, z.code, fontsize=9, weight="bold",
                color="#1D4E89")

    px = [p.x for p in points]
    py = [p.y for p in points]
    ax.scatter(px, py, s=52, c="#C1442E", marker="o", edgecolors="white",
               linewidths=1.1, zorder=5, label="inspection points")
    for p in points:
        ax.annotate(str(p.index), (p.x, p.y), fontsize=6, color="white",
                    ha="center", va="center", zorder=6)

    sx, sy = config.START_POSE_XY
    ax.scatter([sx], [sy], s=150, c="#1D9E75", marker="*", edgecolors="black",
               linewidths=0.8, zorder=7, label="charging station (start)")

    ax.set_title(f"11 zones, {len(points)} inspection points "
                 f"(seed {config.DEFAULT_SEED})")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig("inspection_points.png", dpi=130)
    print("Saved inspection_points.png")
