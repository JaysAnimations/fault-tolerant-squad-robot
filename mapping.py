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


def classify(L):
    """
    Log-odds to a 3-state map: 1 occupied, 0 free, -1 still unknown.

    Free function because Step 4 needs to classify beliefs that are not
    self.L -- what the map would say on one robot's evidence alone. Kept
    here so the two thresholds are written down exactly once.
    """
    out = np.full(L.shape, -1, dtype=np.int8)
    out[L >= config.LOG_ODDS_OCC_THRESHOLD] = 1
    out[L <= config.LOG_ODDS_FREE_THRESHOLD] = 0
    return out


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

        # How much RAW observation we have already taken from each source,
        # before any trust weight was applied. merge_from needs this to
        # work out what is NEW in a packet; _contributions cannot answer
        # that question once the weight has changed, because it holds the
        # weighted total rather than the evidence. See merge_from.
        self._raw_merged = {}

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
    def own_observations(self):
        """
        What THIS robot has seen with its own sensor.

        Excludes two things, and both exclusions matter when robots start
        sharing maps:

          the PRIOR -- every robot was issued the same drawings. Sending
          them to each other adds information neither of them gathered,
          and after a few meetings every cell would sit at the clamp with
          the deviations buried under it.

          ANYTHING MERGED FROM OTHERS -- if A tells B and B then tells C
          what A said, C would hold A's evidence twice once it meets A
          directly. Log-odds addition cannot tell that two numbers came
          from the same observation. Robots therefore share only what they
          saw themselves; evidence travels one hop, on direct contact.
        """
        own = self._contributions.get(self.owner_id)
        return own if own is not None else np.zeros_like(self.L)

    def merge_from(self, other, source_id=None, weight=1.0):
        """
        Fuse another robot's observations into this one.

        Because both maps are in log-odds, fusion is simply addition --
        this is the mathematical reason decentralised mapping works at all,
        and it is worth one slide in your presentation.

        IDEMPOTENT ON PURPOSE. Robots meet again and again, and addition is
        not idempotent: merging the same map twice would count the same
        evidence twice. We therefore apply only the part not already taken
        from this source, using the provenance ledger rollback() already
        keeps. Merging an unchanged map is then a no-op, and the ledger
        still holds the total contributed by that robot, so quarantining it
        later removes exactly what it gave us.

        THE WEIGHT APPLIES TO THE INCREMENT, NOT THE TOTAL. This is M1, and
        it was a real bug. The old version stored `raw * weight` as the
        source's total and replaced it on every merge, so the first packet
        arriving after a robot's trust dropped 1.0 -> 0.25 applied
        `0.25*raw_now - 1.0*raw_earlier`. That does not mean "count this
        robot's new evidence less"; it means "retroactively scale down
        everything this robot has ever said", including all the correct
        mapping it did BEFORE its sensor degraded. It was the whole cause
        of the sensor_degradation collapse -- C3 surface F1 0.530 against
        C2's 0.686, observed error 8.56 % against 0.95 %.

        Two ledgers are therefore kept, and the distinction is the fix:

          _raw_merged[source]    what that robot has told us, unweighted.
                                 Only ever used to work out what is new.
          _contributions[source] what we actually added to self.L on its
                                 behalf, accumulated across merges at
                                 whatever weight was current each time.
                                 rollback() subtracts this, so it still
                                 removes the full accumulated contribution.

        Evidence already merged keeps the weight it was applied under. A
        robot that maps well for 450 steps and then goes blind keeps the
        450 steps.
        """
        raw = other.own_observations()
        if source_id is None:
            # Untracked merge: no provenance, so no rollback and no
            # protection against double counting. Present for completeness;
            # the squad always passes a source_id.
            self.L += raw * weight
        else:
            seen_before = self._raw_merged.get(source_id)
            increment = raw if seen_before is None else raw - seen_before
            payload = increment * weight

            self.L += payload
            if source_id in self._contributions:
                self._contributions[source_id] += payload
            else:
                self._contributions[source_id] = payload
            self._raw_merged[source_id] = raw.copy()
        np.clip(self.L, -config.LOG_ODDS_CLAMP, config.LOG_ODDS_CLAMP,
                out=self.L)

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
            # Forget what we had taken from it as well, or a later merge
            # from the same robot would add only the delta since the
            # rollback on top of a contribution we have just removed.
            # Un-quarantining therefore re-applies the whole map at the
            # weight current at that time, which is the correct "start
            # again from nothing" reading.
            self._raw_merged.pop(source_id, None)
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
        return classify(self.L)

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

    def contribution_conflict(self, source_a, source_b):
        """
        How badly two robots' OWN observations contradict each other.

        Returns (conflicting_cells, overlapping_cells), counting only cells
        where BOTH sources have actually seen something. Two robots that
        have never been to the same place have nothing to disagree about,
        and dividing by the whole grid would bury a real conflict in
        400,000 cells of untouched prior.

        WHY THIS EXISTS AT ALL, AND WHY IT DOES NOT READ self.L. Comparing
        the merged maps is the obvious approach and it is worse than
        useless: robots merge before anyone compares, so a robot with a
        displaced pose has already had its observations folded into
        everybody else's map by the time a detector could look. Session 6
        measured the merged maps *converging* under that fault -- 492 cells
        in dispute against 2397 for a healthy squad. The faulty squad looks
        healthier than the healthy one.

        The provenance ledger is what makes an honest comparison possible.
        It keeps each source's contribution separate, so we can reconstruct
        what the map WOULD say on one robot's evidence alone -- the prior
        it was issued, plus that robot's observations and nothing else --
        and set two of those side by side.
        """
        a = self._contributions.get(source_a)
        b = self._contributions.get(source_b)
        if a is None or b is None:
            return 0, 0

        base = self.prior_L if self.prior_L is not None else 0.0
        seen = (a != 0.0) & (b != 0.0)
        overlap = int(seen.sum())
        if overlap == 0:
            return 0, 0

        belief_a = classify(base + a if self.prior_L is not None else a)
        belief_b = classify(base + b if self.prior_L is not None else b)
        decided = seen & (belief_a >= 0) & (belief_b >= 0)
        conflict = int((belief_a[decided] != belief_b[decided]).sum())
        return conflict, overlap

    def contribution_conflicts_common(self, sources):
        """
        Every pairwise conflict rate for a group of robots, all measured
        over the cells that EVERY one of them has evidence for.

        WHY THE COMMON REGION AND NOT EACH PAIR'S OWN OVERLAP -- this is M3,
        and it was convicting healthy robots.

        contribution_conflict() above measures a pair over whatever ground
        that pair happens to share. The Byzantine test then divides one such
        rate by another, so it compares two numbers computed on DIFFERENT
        PATCHES OF THE FACILITY. Mapping difficulty is not uniform across
        the site: a pair that overlapped mostly on the open perimeter road
        shows a low rate, and any robot whose overlap fell among the pipe
        racks looks displaced by comparison.

        Deviations are what switch this on, and not for the obvious reason.
        They do not create the conflict -- only 1.2 to 3.6 % of conflicting
        cells lie anywhere near one. What they change is ROUTING: robots
        detour around obstacles the drawings do not show, and therefore who
        ends up sharing ground with whom. Seed 42 with no fault injected at
        all: adding deviations collapsed the r1-r2 overlap from 81,723 cells
        to 56,913 while r0-r1 grew to 108,731, and the ratio against a
        perfectly healthy robot 0 went from 0.79 to 3.28 against a threshold
        of 1.4. The test was reading a difference in WHERE THE ROBOTS MET as
        evidence of a pose fault.

        Restricting all three rates to the common region removes that by
        construction. Every numerator and the single shared denominator now
        describe the same cells, so the ratio compares like with like. This
        changes WHAT IS SAMPLED, not any threshold.

        The cost is sample size -- the three-way intersection is smaller
        than any pairwise overlap -- which is why the caller applies
        BYZANTINE_MIN_TRIPLE_OVERLAP_CELLS rather than the pairwise floor.

        Returns (rates, overlap): rates keyed by frozenset({a, b}), and the
        size of the common region, which is the same for every pair. That
        shared denominator is the entire point.
        """
        held = {}
        for source in sources:
            contribution = self._contributions.get(source)
            if contribution is None:
                return {}, 0        # somebody has told us nothing yet
            held[source] = contribution

        seen_all = None
        for contribution in held.values():
            mine = (contribution != 0.0)
            seen_all = mine if seen_all is None else (seen_all & mine)

        overlap = int(seen_all.sum())
        if overlap == 0:
            return {}, 0

        # What the map would say on each robot's evidence alone -- its own
        # observations on top of the drawings everybody was issued.
        base = self.prior_L
        belief = {}
        for source, contribution in held.items():
            belief[source] = classify(base + contribution if base is not None
                                      else contribution)

        rates = {}
        ids = sorted(held)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = belief[ids[i]], belief[ids[j]]
                # A cell only counts against somebody if both robots have
                # actually made their minds up about it.
                decided = seen_all & (a >= 0) & (b >= 0)
                conflict = int((a[decided] != b[decided]).sum())
                rates[frozenset((ids[i], ids[j]))] = conflict / overlap
        return rates, overlap

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
