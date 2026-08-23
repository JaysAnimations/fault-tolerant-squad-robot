"""
deviations.py
=============
The things the drawings do not show. Design Change 01, section 4.

THE IDEA
--------
Robots are issued the facility's documented layout, but the facility has
moved on since the drawing was last revised: scaffolding has gone up, a
skid has been taken out for maintenance, an aisle has been closed off.
This module makes ground truth disagree with the drawings, on purpose and
reproducibly.

  environment.Facility.documented_grid  -- what the drawings say
  environment.Facility.grid             -- what is actually there

Only the second one is edited here. That gap is the whole design: it is
what the robots have to discover, and "the squad detected 6 of the 7
deviations" is a genuine industrial deliverable.

THREE TYPES
-----------
  added    a new obstacle in open ground -- scaffolding, a parked trailer
  removed  equipment gone from where the drawings put it
  blocked  a corridor closed off, chosen from a list of real corridors

Blocked routes matter most: they change the COST of the mission and not
just the map, because the robot has to re-plan and drive round.

THE UNREACHABILITY CAP
----------------------
A deviation can cut an inspection point off from the rest of the site.
That is allowed -- a scaffolded-over valve is a real thing that happens,
and the squad should recognise it rather than burn the mission trying --
but it is capped at ONE point per run (decision 3). Any candidate that
would push it past the cap is undone and another is drawn. Without the
cap a bad seed could seal the tank farm and decide the experiment.

DETECTION LATENCY IS PREDICTED, NOT DISCOVERED (section 11)
-----------------------------------------------------------
The log-odds update is deliberately asymmetric: a beam ending on a cell
adds +0.85, a beam passing through adds only -0.40, because a hit is
strong evidence of an obstacle while a pass-through is weak evidence of
emptiness. So from a prior of +/-2.0, an ADDED obstacle needs about 4
observations to confirm and a REMOVED one about 8. Removed obstacles
should take roughly twice as long to detect. That is a consequence of the
sensor model, not a bug, and it is worth reporting as detection latency
broken down by type.
"""

import numpy as np

import config
from environment import L_DEVIATION
from inspection import (build_zones, reachability_planner, reachable_field,
                        is_reachable)


class Deviation:
    """One discrepancy between the drawings and reality."""

    def __init__(self, kind, name, rect, zone_code):
        self.kind = kind                 # "added" / "removed" / "blocked"
        self.name = name
        self.rect = tuple(float(v) for v in rect)   # x0, y0, x1, y1
        self.zone_code = zone_code

        # Cells the robot could actually gather evidence from. Filled in by
        # _attach_evidence once every deviation has been placed.
        self.rows = None
        self.cols = None

        self.cut_off = []                # inspection point indices isolated
        self.detected = False
        self.detected_step = None
        self.best_fraction = 0.0         # highest contradiction seen so far

    @property
    def centre(self):
        x0, y0, x1, y1 = self.rect
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))

    @property
    def area_m2(self):
        x0, y0, x1, y1 = self.rect
        return (x1 - x0) * (y1 - y0)

    @property
    def n_evidence_cells(self):
        return 0 if self.rows is None else len(self.rows)

    def __repr__(self):
        return (f"<Deviation {self.kind} '{self.name}' {self.zone_code} "
                f"{self.area_m2:.1f} m2 "
                f"{'detected' if self.detected else 'undetected'}>")


# ---------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------
def _pick_kinds(rng, n):
    """Draw n deviation types according to the configured weights."""
    kinds = list(config.DEVIATION_TYPE_WEIGHTS.keys())
    weights = np.array([config.DEVIATION_TYPE_WEIGHTS[k] for k in kinds],
                       dtype=float)
    weights = weights / weights.sum()      # normalise here so the config
                                           # values can be plain weights
    return [str(k) for k in rng.choice(kinds, size=n, p=weights)]


def _shuffled(rng, items):
    """A reproducibly shuffled copy of a list."""
    order = rng.permutation(len(items))
    return [items[int(i)] for i in order]


def _zone_for(zones, x, y):
    for z in zones:
        if z.contains(x, y):
            return z.code
    return "--"          # between zones: the perimeter, the east yard, etc.


def _propose_added(rng, facility, planner, dist_documented):
    """
    A new obstacle in open ground.

    Two conditions. The footprint must be entirely free RIGHT NOW, so it is
    genuinely an addition and does not silently merge with equipment or
    with an earlier deviation. And its centre must be somewhere a robot
    could have driven to on the drawings, because an obstacle in a corner
    no route passes is undetectable by construction, which would make the
    detection-rate metric meaningless.
    """
    w = float(rng.uniform(config.DEVIATION_ADDED_MIN_SIZE_M,
                          config.DEVIATION_ADDED_MAX_SIZE_M))
    h = float(rng.uniform(config.DEVIATION_ADDED_MIN_SIZE_M,
                          config.DEVIATION_ADDED_MAX_SIZE_M))
    x = float(rng.uniform(0.0, facility.width_m - w))
    y = float(rng.uniform(0.0, facility.height_m - h))
    rect = (x, y, x + w, y + h)

    r0, r1, c0, c1 = facility.region_bounds(*rect)
    if r1 <= r0 or c1 <= c0:
        return None
    if facility.grid[r0:r1, c0:c1].any():
        return None
    cx, cy = x + w / 2.0, y + h / 2.0
    if not is_reachable(planner, dist_documented, cx, cy):
        return None
    return f"obstacle at ({cx:.0f}, {cy:.0f}) m", rect


def _propose_removed(pool):
    """Take one piece of equipment out, from the list environment.py kept."""
    if not pool:
        return None
    rect = pool.pop()
    cx, cy = 0.5 * (rect[0] + rect[2]), 0.5 * (rect[1] + rect[3])
    return f"equipment at ({cx:.0f}, {cy:.0f}) m", tuple(rect)


def _propose_blocked(pool):
    """Close one of the corridors listed in config."""
    if not pool:
        return None
    name, x0, y0, x1, y1 = pool.pop()
    return name, (x0, y0, x1, y1)


def _unreachable_points(facility, points, planner):
    """Which inspection points ground truth has cut off from the start."""
    dist = reachable_field(planner, facility.grid, config.START_POSE_XY)
    return [p.index for p in points
            if not is_reachable(planner, dist, p.x, p.y)]


def inject_deviations(facility, points, seed=config.DEFAULT_SEED,
                      planner=None, verbose=False):
    """
    Edit ground truth so it disagrees with the drawings, and return the
    list of Deviations injected.

    Call AFTER generate_inspection_points: the points are planned from the
    drawings, and only then does reality get in the way -- which is the
    order these things happen in a real facility.
    """
    rng = np.random.default_rng([seed, config.RNG_STREAM_DEVIATIONS])
    if planner is None:
        planner = reachability_planner(facility)

    n_target = int(rng.integers(config.DEVIATIONS_MIN_PER_RUN,
                                config.DEVIATIONS_MAX_PER_RUN + 1))
    if n_target == 0:
        return []                        # condition C0: perfect drawings

    zones = build_zones()
    kinds = _pick_kinds(rng, n_target)

    # Pools are shuffled once and then popped, so the same candidate is
    # never used twice in a run.
    removable_pool = _shuffled(rng, [tuple(r) for r in
                                     facility.features["removable"]])
    blocked_pool = _shuffled(rng, list(config.BLOCKED_ROUTE_CANDIDATES))

    # Reachability on the DRAWINGS, used to judge where an added obstacle
    # is worth putting. Computed once: the drawings do not change.
    dist_documented = reachable_field(planner, facility.documented_grid,
                                      config.START_POSE_XY)

    injected = []
    rejected = []
    cut_off_so_far = []

    for kind in kinds:
        placed = False
        for _attempt in range(config.DEVIATION_PLACEMENT_ATTEMPTS):
            if kind == "added":
                proposal = _propose_added(rng, facility, planner,
                                          dist_documented)
            elif kind == "removed":
                proposal = _propose_removed(removable_pool)
            else:
                proposal = _propose_blocked(blocked_pool)

            if proposal is None:
                # "added" just missed; the pooled kinds are exhausted.
                if kind == "added":
                    continue
                break

            name, rect = proposal
            occupied = (kind != "removed")
            label = L_DEVIATION if occupied else None

            # Snapshot before editing, so a rejected candidate leaves no
            # trace. Reverting by re-drawing the original shape would be
            # wrong where two deviations overlap.
            r0, r1, c0, c1 = facility.region_bounds(*rect)
            before_grid = facility.grid[r0:r1, c0:c1].copy()
            before_labels = facility.labels[r0:r1, c0:c1].copy()

            facility.set_region(rect[0], rect[1], rect[2], rect[3],
                                occupied, label=label)

            cut_off = _unreachable_points(facility, points, planner)
            if len(cut_off) <= config.MAX_UNREACHABLE_POINTS:
                cx = 0.5 * (rect[0] + rect[2])
                cy = 0.5 * (rect[1] + rect[3])
                dev = Deviation(kind, name, rect, _zone_for(zones, cx, cy))
                dev.cut_off = [i for i in cut_off if i not in cut_off_so_far]
                cut_off_so_far = cut_off
                injected.append(dev)
                placed = True
                break

            # Over the cap: undo it and try another candidate.
            facility.grid[r0:r1, c0:c1] = before_grid
            facility.labels[r0:r1, c0:c1] = before_labels
            facility._boundary = None
            rejected.append((kind, name, len(cut_off)))

        if not placed and verbose:
            print(f"  [warning] could not place a '{kind}' deviation")

    _attach_evidence(facility, injected)
    mark_unreachable_points(facility, points, planner)

    if verbose:
        for kind, name, n in rejected:
            print(f"  [reverted] {kind:<8s} {name:<32s} would have cut off "
                  f"{n} inspection point(s)")

    return injected


def mark_unreachable_points(facility, points, planner=None):
    """
    Flag the inspection points ground truth has cut off.

    ANALYSIS ONLY. The robot is never shown these flags -- it has to work
    out for itself that a point cannot be reached, which is the behaviour
    decision 3 asks for. They exist so the report can separate "the squad
    failed to visit this point" from "nobody could have".
    """
    if planner is None:
        planner = reachability_planner(facility)
    cut_off = set(_unreachable_points(facility, points, planner))
    for p in points:
        p.unreachable = p.index in cut_off
    return [p for p in points if p.unreachable]


# ---------------------------------------------------------------------
# Evidence and detection
# ---------------------------------------------------------------------
def _attach_evidence(facility, deviations):
    """
    Work out which cells the robot can actually gather evidence from.

    THIS IS THE SAME ARGUMENT AS SURFACE-SCORING IN mapping.py, and it is
    worth being able to state at the defence. A 2D LiDAR sees the OUTER
    BOUNDARY of a solid object and never its inside, so for an added
    obstacle the evidence is its visible surface -- scoring against its
    filled footprint would cap detection at a low value however perfect
    the mapping was.

    A removed obstacle is the opposite case: the beams now pass straight
    through where the equipment used to be, so every cell of the old
    footprint is evidence.
    """
    surface = facility.boundary_mask()
    for dev in deviations:
        r0, r1, c0, c1 = facility.region_bounds(*dev.rect)
        truth = facility.grid[r0:r1, c0:c1]
        drawn = facility.documented_grid[r0:r1, c0:c1]

        if dev.kind == "removed":
            mask = (drawn == 1) & (truth == 0)
        else:
            mask = (truth == 1) & surface[r0:r1, c0:c1]
            if not mask.any():
                # Should not happen -- an obstacle in open ground always has
                # a visible surface -- but an empty evidence set would make
                # the deviation undetectable by arithmetic rather than by
                # the robot's performance.
                mask = truth == 1

        rows, cols = np.nonzero(mask)
        dev.rows = rows + r0
        dev.cols = cols + c0


def contradiction_fraction(dev, grid, masks=None):
    """
    How much of this deviation the robot's own map now contradicts.

    Reads ONLY the robot's map and the prior it was issued, through
    OccupancyGrid.contradicts_prior(). It never looks at ground truth --
    the deviation's footprint tells us WHERE to look, and the robot's own
    belief decides WHAT it found. That distinction is what keeps the
    detection metric honest.

    `masks` is the (now_solid, now_open) pair from contradicts_prior. Pass
    it in when scoring several deviations at once: building it dilates two
    full-size masks, so doing it once per check rather than once per
    deviation is worth the extra argument.
    """
    if dev.n_evidence_cells == 0:
        return 0.0
    if masks is None:
        masks = grid.contradicts_prior()
    now_solid, now_open = masks

    # Read the mask that matches what the drawings claim about each cell.
    # The two masks are disjoint before dilation but overlap after it, so
    # this must select rather than OR -- otherwise a removed obstacle
    # nearby could be credited as evidence for an added one.
    drawn_solid = grid.prior_L[dev.rows, dev.cols] > 0
    contradicts = np.where(drawn_solid,
                           now_open[dev.rows, dev.cols],
                           now_solid[dev.rows, dev.cols])
    return float(contradicts.mean())


def update_detection(deviations, grid, step):
    """
    Check every deviation against the robot's current map and record the
    first step at which each one is confirmed.

    Detection is latching: once a deviation has been seen it stays
    reported, even if the robot later drives away and its map softens.
    That matches what a real inspection round produces -- a finding, filed.
    """
    masks = grid.contradicts_prior()
    for dev in deviations:
        frac = contradiction_fraction(dev, grid, masks)
        dev.best_fraction = max(dev.best_fraction, frac)
        if not dev.detected and frac >= config.DEVIATION_DETECT_FRACTION:
            dev.detected = True
            dev.detected_step = step


def detection_summary(deviations):
    """{total, detected, rate, and per-type counts and mean latency}."""
    out = {"total": len(deviations), "detected": 0, "rate": 0.0, "by_kind": {}}
    for dev in deviations:
        k = out["by_kind"].setdefault(dev.kind, {"total": 0, "detected": 0,
                                                 "steps": []})
        k["total"] += 1
        if dev.detected:
            out["detected"] += 1
            k["detected"] += 1
            if dev.detected_step is not None:
                k["steps"].append(dev.detected_step)
    if deviations:
        out["rate"] = out["detected"] / len(deviations)
    return out


# =====================================================================
# Verification: python deviations.py
# =====================================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch

    from environment import Facility
    from inspection import generate_inspection_points

    seed = config.DEFAULT_SEED

    facility = Facility()
    planner = reachability_planner(facility)
    points = generate_inspection_points(facility, seed, planner)
    devs = inject_deviations(facility, points, seed, planner, verbose=True)

    print("\n" + "=" * 74)
    print(f"  DEVIATIONS INJECTED -- seed {seed}")
    print("=" * 74)
    print(f"  {'#':<3s} {'type':<8s} {'zone':<5s} {'where':<32s} "
          f"{'m2':>6s} {'cells':>6s}")
    print("-" * 74)
    for i, d in enumerate(devs):
        print(f"  {i:<3d} {d.kind:<8s} {d.zone_code:<5s} {d.name:<32s} "
              f"{d.area_m2:6.1f} {d.n_evidence_cells:6d}")
    print("-" * 74)

    counts = {}
    for d in devs:
        counts[d.kind] = counts.get(d.kind, 0) + 1
    print(f"  Total                 : {len(devs)} "
          f"(config allows {config.DEVIATIONS_MIN_PER_RUN}-"
          f"{config.DEVIATIONS_MAX_PER_RUN})")
    print(f"  By type               : {counts}")

    cut = [p for p in points if p.unreachable]
    print(f"  Points cut off        : {len(cut)} "
          f"(cap is {config.MAX_UNREACHABLE_POINTS})"
          + (f"  -> {[p.index for p in cut]}" if cut else ""))
    for d in devs:
        if d.cut_off:
            print(f"      by deviation '{d.name}' ({d.kind}): points {d.cut_off}")

    # --- the point of the whole stage --------------------------------
    pristine = Facility()
    prior_untouched = np.array_equal(facility.documented_grid, pristine.grid)
    truth_changed = int((facility.grid != facility.documented_grid).sum())
    added_cells = int(((facility.grid == 1) &
                       (facility.documented_grid == 0)).sum())
    removed_cells = int(((facility.grid == 0) &
                         (facility.documented_grid == 1)).sum())

    print("-" * 74)
    print(f"  Prior identical to a facility with no deviations : {prior_untouched}")
    print(f"  Ground-truth cells that differ from the drawings : {truth_changed}")
    print(f"      new obstacle where the drawings say open     : {added_cells}")
    print(f"      open where the drawings say obstacle         : {removed_cells}")

    # --- determinism --------------------------------------------------
    f2 = Facility()
    p2 = generate_inspection_points(f2, seed, reachability_planner(f2))
    d2 = inject_deviations(f2, p2, seed, reachability_planner(f2))
    same = (len(d2) == len(devs) and
            all(a.kind == b.kind and a.rect == b.rect
                for a, b in zip(d2, devs)) and
            np.array_equal(f2.grid, facility.grid))
    f3 = Facility()
    p3 = generate_inspection_points(f3, seed + 1, reachability_planner(f3))
    d3 = inject_deviations(f3, p3, seed + 1, reachability_planner(f3))
    differs = not (len(d3) == len(devs) and
                   all(a.rect == b.rect for a, b in zip(d3, devs)))

    print(f"  Same seed reproduces byte-identical ground truth : {same}")
    print(f"  A different seed gives different deviations      : {differs}")
    print("=" * 74 + "\n")

    # --- figure --------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.2))
    ext = [0, facility.width_m, 0, facility.height_m]

    documented = Facility()      # a facility with the deviations left out
    axes[0].imshow(documented.render_rgb(), origin="lower", extent=ext,
                   interpolation="nearest")
    axes[0].set_title("What the drawings say\n(the prior the robots are issued)")

    axes[1].imshow(facility.render_rgb(), origin="lower", extent=ext,
                   interpolation="nearest")
    for d in devs:
        x0, y0, x1, y1 = d.rect
        axes[1].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                    ec="#C1442E", lw=2.0))
    axes[1].scatter([p.x for p in points], [p.y for p in points], s=18,
                    c="#1D4E89", marker="o", edgecolors="white", linewidths=0.6)
    axes[1].set_title(f"What is actually there\n({len(devs)} deviations, "
                      f"inspection points in blue)")

    diff = np.zeros((facility.n_rows, facility.n_cols, 3), dtype=np.float32)
    diff[:] = (0.97, 0.96, 0.94)
    diff[(facility.grid == 1) & (facility.documented_grid == 0)] = (0.86, 0.15, 0.15)
    diff[(facility.grid == 0) & (facility.documented_grid == 1)] = (0.15, 0.35, 0.80)
    diff[(facility.grid == 1) & (facility.documented_grid == 1)] *= 0.80
    axes[2].imshow(diff, origin="lower", extent=ext, interpolation="nearest")
    axes[2].legend(handles=[
        Patch(facecolor=(0.86, 0.15, 0.15), label="added / blocked (not on the drawings)"),
        Patch(facecolor=(0.15, 0.35, 0.80), label="removed (drawn, but gone)")],
        loc="upper right", fontsize=8)
    axes[2].set_title("The difference the robots must discover")

    for ax in axes:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig("deviations.png", dpi=120)
    print("Saved deviations.png")
