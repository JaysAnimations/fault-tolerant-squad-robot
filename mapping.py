"""
mapping.py
==========
Log-odds occupancy grid mapping -- the standard formulation from
Thrun, Burgard & Fox, *Probabilistic Robotics* (cited as [3] in HENRY's
Chapter 2, so quote it directly in Chapter 3).

WHY LOG-ODDS RATHER THAN PROBABILITIES:
  Bayesian updating of a probability requires multiplication and
  renormalisation on every observation. Working in log-odds space turns
  that into simple ADDITION, which is fast, numerically stable, and lets a
  cell be "corrected" later if it was wrongly observed. That last property
  is what makes fault recovery possible: when we detect a Byzantine robot,
  we subtract its contributions back out.

WHAT EACH BEAM TEACHES US:
  A beam of length r in direction a tells us two different things:
    1. every cell the beam passed THROUGH is probably free  (negative update)
    2. the cell where the beam STOPPED is probably occupied (positive update)
  A beam with no return only teaches us (1).
"""

import numpy as np
import config


class OccupancyGrid:
    """One robot's private belief about the world."""

    def __init__(self, facility, owner_id=None):
        self.res = facility.res
        self.n_rows = facility.n_rows
        self.n_cols = facility.n_cols
        self.owner_id = owner_id

        # log-odds accumulator. 0.0 everywhere = "completely unknown".
        self.L = np.zeros((self.n_rows, self.n_cols), dtype=np.float32)

        # The documented layout the robot was issued, kept separately from
        # what it has since observed. None until seed_prior() is called --
        # a robot with no prior is the emergency-response scenario, where
        # the drawings cannot be trusted and it must map from nothing.
        self.prior_L = None

        # Provenance ledger: how much log-odds each robot contributed.
        # This is what makes ROLLBACK possible when a robot is later found
        # to be faulty. It is the software equivalent of the "immutable
        # registry of spatial transactions" described in Moroncelli et al.
        self._contributions = {}

        self.step_m = config.LIDAR_RAY_STEP_M

    # -----------------------------------------------------------------
    def seed_prior(self, documented_grid, log_odds=config.PRIOR_LOG_ODDS):
        """
        Issue this robot the facility's documented layout before it sets
        off. Design Change 01, section 5.

        WHY A MODERATE CONFIDENCE AND NOT CERTAINTY
          The drawings are seeded at +/- 2.0 rather than at the +/- 8.0
          clamp, so they are BELIEVED BUT OVERTURNABLE. The classification
          threshold is +/- 1.0, a beam ending on a cell is worth +0.85 and
          a beam passing through is worth -0.40, so:

              a cell drawn as OPEN needs   ~4 hits          to become solid
              a cell drawn as SOLID needs  ~8 pass-throughs to become open

          which is the detection latency predicted in section 11 of the
          design note. Seed it at the clamp instead and the robot would
          argue with reality for the whole mission.

        WHY THE PRIOR IS KEPT SEPARATELY
          Two reasons, and both matter.
          1. Deviation detection compares what the drawings said against
             what was observed. That comparison needs the drawings to
             still be available afterwards, not folded away into a single
             number per cell.
          2. It is deliberately NOT written to self._contributions. The
             prior is infrastructure issued by the operator, not a
             robot's contribution, so rollback() must never take it away.
             Quarantining a Byzantine robot removes what that robot said;
             it does not un-issue the plot plan.
        """
        self.prior_L = np.where(documented_grid == 1,
                                log_odds, -log_odds).astype(np.float32)
        self.L += self.prior_L
        np.clip(self.L, -config.LOG_ODDS_CLAMP, config.LOG_ODDS_CLAMP,
                out=self.L)

    # -----------------------------------------------------------------
    def integrate_scan(self, x, y, theta, ranges, angles, valid,
                       source_id=None, weight=1.0):
        """
        Fold one LiDAR scan into the map.

        source_id : which robot produced this data (for rollback)
        weight    : 1.0 = full trust. Reduce to down-weight a suspect robot
                    instead of rejecting it outright.
        """
        delta = np.zeros_like(self.L)

        max_steps = int(np.ceil(ranges.max() / self.step_m)) + 1
        dists = (np.arange(1, max_steps + 1) * self.step_m)[None, :]   # (1,S)

        cos_a = np.cos(angles)[:, None]
        sin_a = np.sin(angles)[:, None]

        px = x + cos_a * dists
        py = y + sin_a * dists

        cols = np.floor(px / self.res).astype(np.int32)
        rows = np.floor(py / self.res).astype(np.int32)

        inb = (rows >= 0) & (rows < self.n_rows) & \
              (cols >= 0) & (cols < self.n_cols)

        # --- FREE-SPACE UPDATE: samples strictly before the endpoint ---
        # We stop 1.5 grid cells short. Stopping only one RAY STEP short is
        # not enough: because the ray step is finer than the grid, the last
        # free sample can land in the SAME cell as the obstacle and cancel
        # out the occupied update. This was a real bug -- obstacles simply
        # failed to appear in the map. Margin must exceed the cell size.
        free_margin = 1.5 * self.res
        free_mask = (dists < (ranges[:, None] - free_margin)) & inb
        if free_mask.any():
            np.add.at(delta,
                      (rows[free_mask], cols[free_mask]),
                      config.LOG_ODDS_FREE * weight)

        # --- OCCUPIED UPDATE: the endpoint of each beam that returned ---
        if valid.any():
            ex = x + np.cos(angles[valid]) * ranges[valid]
            ey = y + np.sin(angles[valid]) * ranges[valid]
            er = np.floor(ey / self.res).astype(np.int32)
            ec = np.floor(ex / self.res).astype(np.int32)
            ok = (er >= 0) & (er < self.n_rows) & (ec >= 0) & (ec < self.n_cols)
            if ok.any():
                np.add.at(delta, (er[ok], ec[ok]),
                          config.LOG_ODDS_OCCUPIED * weight)

        self.L += delta
        np.clip(self.L, -config.LOG_ODDS_CLAMP, config.LOG_ODDS_CLAMP,
                out=self.L)

        if source_id is not None:
            if source_id not in self._contributions:
                self._contributions[source_id] = np.zeros_like(self.L)
            self._contributions[source_id] += delta

    # -----------------------------------------------------------------
    def merge_from(self, other, source_id=None, weight=1.0):
        """
        Fuse another robot's map into this one.

        Because both maps are in log-odds, fusion is simply addition --
        this is the mathematical reason decentralised mapping works at all,
        and it is worth one slide in your presentation.
        """
        contrib = other.L * weight
        self.L += contrib
        np.clip(self.L, -config.LOG_ODDS_CLAMP, config.LOG_ODDS_CLAMP,
                out=self.L)
        if source_id is not None:
            if source_id not in self._contributions:
                self._contributions[source_id] = np.zeros_like(self.L)
            self._contributions[source_id] += contrib

    def rollback(self, source_id):
        """
        Remove everything a given robot ever contributed.

        THIS IS YOUR FAULT-RECOVERY MECHANISM. When cross-agent consistency
        checking flags robot 2 as Byzantine, we call rollback(2) and the
        global map is restored to what it would have been without it.
        Demonstrating this live is a very strong defence moment.

        Note what this does NOT remove: the prior from seed_prior(). Only
        entries in _contributions are subtracted, and the prior was never
        written there. Quarantining a robot withdraws what that robot
        said, not the plot plan the operator issued.
        """
        if source_id in self._contributions:
            self.L -= self._contributions[source_id]
            np.clip(self.L, -config.LOG_ODDS_CLAMP, config.LOG_ODDS_CLAMP,
                    out=self.L)
            del self._contributions[source_id]
            return True
        return False

    # -----------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------
    def classified(self):
        """
        Convert log-odds to a 3-state map.
          1 = occupied, 0 = free, -1 = still unknown
        """
        out = np.full(self.L.shape, -1, dtype=np.int8)
        out[self.L >= config.LOG_ODDS_OCC_THRESHOLD] = 1
        out[self.L <= config.LOG_ODDS_FREE_THRESHOLD] = 0
        return out

    def known_mask(self):
        return self.L != 0.0

    # -----------------------------------------------------------------
    # Metrics -- these feed Chapter 4
    # -----------------------------------------------------------------
    def coverage_fraction(self, facility):
        """Fraction of genuinely-free cells the squad has established are free."""
        truth_free = (facility.grid == 0)
        believed_free = (self.L <= config.LOG_ODDS_FREE_THRESHOLD)
        return float((truth_free & believed_free).sum() / truth_free.sum())

    def occupied_iou(self, facility):
        """
        Intersection-over-Union of predicted vs true obstacle SURFACES.

        WHY SURFACES AND NOT SOLID OBSTACLES  (explain this in Chapter 3 --
        it is a genuine methodological point, not a fudge):
          A 2D LiDAR observes the outer boundary of an object. It can never
          see the inside of a storage tank. Scoring the robot's map against
          the filled ground-truth tank would therefore cap IoU at a low
          value no matter how perfect the mapping is. We instead score
          against the OBSERVABLE BOUNDARY: obstacle cells that touch free
          space. That is the set the sensor is physically able to report.
        """
        pred = (self.L >= config.LOG_ODDS_OCC_THRESHOLD)
        truth = facility.boundary_mask()
        inter = (pred & truth).sum()
        union = (pred | truth).sum()
        return float(inter / union) if union > 0 else 0.0

    @staticmethod
    def _dilate(mask, n=1):
        """Grow a boolean mask by n cells (4-connectivity). No scipy needed."""
        out = mask.copy()
        for _ in range(n):
            g = out.copy()
            g[1:, :] |= out[:-1, :]
            g[:-1, :] |= out[1:, :]
            g[:, 1:] |= out[:, :-1]
            g[:, :-1] |= out[:, 1:]
            out = g
        return out

    def surface_scores(self, facility, tolerance_cells=2):
        """
        Precision / recall / F1 for mapped obstacle surfaces, with a spatial
        tolerance.

        WHY A TOLERANCE (a panel may probe this, so know the answer):
          Ground truth obstacle boundaries are one cell thick. A real LiDAR
          reading carries range noise, and the pose it is integrated at
          carries odometry error, so a correctly-detected wall lands within
          a cell or two of its true position rather than exactly on it.
          Scoring with zero tolerance measures grid quantisation, not
          mapping quality. We allow +/- 2 cells = 20 cm, which is on the
          order of the combined sensor noise and pose error.

          Report the tolerance explicitly in Chapter 3. An undeclared
          tolerance looks like cheating; a declared one looks like rigour.

        Returns (precision, recall, f1).
        """
        pred = (self.L >= config.LOG_ODDS_OCC_THRESHOLD)
        truth = facility.boundary_mask()

        truth_tol = self._dilate(truth, tolerance_cells)
        pred_tol = self._dilate(pred, tolerance_cells)

        n_pred = pred.sum()
        n_truth = truth.sum()
        if n_pred == 0 or n_truth == 0:
            return 0.0, 0.0, 0.0

        precision = float((pred & truth_tol).sum() / n_pred)
        recall = float((truth & pred_tol).sum() / n_truth)
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        return precision, recall, f1

    def contradicts_prior(self, tolerance_cells=None):
        """
        This robot's deviation report, as two masks:

            now_solid : the drawings show open ground, the robot has since
                        established that something is standing there
            now_open  : the drawings show something solid, the robot has
                        since driven or looked straight through it

        Reads only the robot's own map and the prior it was issued. No
        ground truth is involved, so this is a report a real robot could
        produce and radio home.

        WHY THERE IS A SPATIAL TOLERANCE
          Scans are integrated at the BELIEVED pose, so an obstacle the
          robot has correctly seen lands a cell or two from where it truly
          is. Comparing cell-for-cell would therefore score odometry drift
          rather than detection. This is the same argument -- and the same
          tolerance -- as surface_scores(), and Chapter 3 should declare it
          once for both.

          WHAT MUST BE DILATED IS THE CONTRADICTION, NOT THE BELIEF. An
          earlier version grew the "occupied" and "free" masks first and
          compared afterwards. That reported deviations at step 0, before
          the robot had moved: growing the free mask by two cells reaches
          the surface of every wall the drawings show, and growing the
          occupied mask reaches the open ground beside it, so the prior
          contradicted itself. Overturn the cell first, then allow the
          evidence to spread. A cell still sitting at its prior value now
          contributes nothing, whatever its neighbours say.
        """
        if tolerance_cells is None:
            tolerance_cells = config.DEVIATION_TOLERANCE_CELLS
        if self.prior_L is None:
            raise RuntimeError("this map was never issued a prior, so there "
                               "is nothing for an observation to contradict")

        drawn_solid = self.prior_L > 0
        now_solid = (~drawn_solid) & (self.L >= config.LOG_ODDS_OCC_THRESHOLD)
        now_open = drawn_solid & (self.L <= config.LOG_ODDS_FREE_THRESHOLD)

        if tolerance_cells > 0:
            now_solid = self._dilate(now_solid, tolerance_cells)
            now_open = self._dilate(now_open, tolerance_cells)
        return now_solid, now_open

    def error_rate(self, facility):
        """Fraction of CLASSIFIED cells that are classified wrongly."""
        cls = self.classified()
        decided = cls >= 0
        if decided.sum() == 0:
            return 0.0
        truth = (facility.grid == 1).astype(np.int8)
        wrong = (cls[decided] != truth[decided]).sum()
        return float(wrong / decided.sum())

    def disagreement_with(self, other):
        """
        Fraction of cells where two robots CONFIDENTLY disagree.

        This is the cross-agent geometric consistency check. If robot A
        insists a cell is occupied while robots B and C insist it is free,
        somebody is faulty -- and majority vote tells you who. It is the
        practical implementation of the "geometric constraint" half of the
        Byzantine detection scheme in your Chapter 2.
        """
        a = self.classified()
        b = other.classified()
        both = (a >= 0) & (b >= 0)
        if both.sum() == 0:
            return 0.0
        return float((a[both] != b[both]).sum() / both.sum())


# =====================================================================
# Verification: python mapping.py
#   Confirms a robot starts the mission holding the documented layout,
#   and that the deviations are absent from that starting belief.
# =====================================================================
if __name__ == "__main__":
    from environment import Facility
    from inspection import generate_inspection_points, reachability_planner
    from deviations import inject_deviations, contradiction_fraction

    seed = config.DEFAULT_SEED

    facility = Facility()
    planner = reachability_planner(facility)
    points = generate_inspection_points(facility, seed, planner)
    devs = inject_deviations(facility, points, seed, planner)

    grid = OccupancyGrid(facility, owner_id=0)
    unknown_before = int((grid.classified() == -1).sum())
    grid.seed_prior(facility.documented_grid)

    cls = grid.classified()
    drawn = facility.documented_grid
    matches_drawings = bool(np.array_equal(cls == 1, drawn == 1))
    unknown_after = int((cls == -1).sum())
    levels = sorted(set(np.unique(grid.L).tolist()))

    print("\n" + "=" * 66)
    print(f"  PRIOR MAP SEEDING -- seed {seed}")
    print("=" * 66)
    print(f"  Prior confidence             : +/- {config.PRIOR_LOG_ODDS} log-odds")
    print(f"  Distinct log-odds values     : {levels}")
    print(f"  Unknown cells before seeding : {unknown_before:,} "
          f"({100*unknown_before/grid.L.size:.1f} %)")
    print(f"  Unknown cells after seeding  : {unknown_after:,}")
    print(f"  Belief matches the drawings  : {matches_drawings}")
    print(f"  Prior stored separately      : {grid.prior_L is not None}")

    # --- the prior is not a robot contribution, so rollback keeps it ---
    grid.integrate_scan(*config.START_POSE_XY, 0.0,
                        np.array([1.0]), np.array([0.0]),
                        np.array([True]), source_id=0)
    rolled = grid.rollback(0)
    prior_survived = bool(np.array_equal(grid.L, grid.prior_L))
    print(f"  rollback() ran               : {rolled}")
    print(f"  Prior survives rollback()    : {prior_survived}")

    # --- the deviations are NOT in the robot's starting belief ---------
    print("-" * 66)
    print("  The robot starts believing the drawings, so at step 0 it")
    print("  contradicts none of the deviations that are actually there:")
    print(f"  {'#':<3s} {'type':<8s} {'where':<32s} {'contradicted':>12s}")
    for i, d in enumerate(devs):
        print(f"  {i:<3d} {d.kind:<8s} {d.name:<32s} "
              f"{contradiction_fraction(d, grid)*100:11.1f} %")

    # --- how many observations it takes to overturn the prior ----------
    # Section 11 of the design note predicts this rather than discovering
    # it. Recomputed here from the actual config values, so the table in
    # the write-up cannot drift away from the code.
    hits_to_flip = int(np.ceil((config.LOG_ODDS_OCC_THRESHOLD +
                                config.PRIOR_LOG_ODDS) /
                               config.LOG_ODDS_OCCUPIED))
    passes_to_flip = int(np.ceil((-config.LOG_ODDS_FREE_THRESHOLD +
                                  config.PRIOR_LOG_ODDS) /
                                 -config.LOG_ODDS_FREE))
    print("-" * 66)
    print("  Observations needed to overturn the drawings:")
    print(f"    added obstacle   (drawn open, now solid) : {hits_to_flip}  hits")
    print(f"    removed obstacle (drawn solid, now open) : {passes_to_flip}  pass-throughs")
    print("    -- removed obstacles take about twice as long to confirm,")
    print("       because a beam hit is stronger evidence than a beam that")
    print("       passes through. That is the sensor model, not a bug.")
    print("=" * 66 + "\n")
