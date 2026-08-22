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

        # Provenance ledger: how much log-odds each robot contributed.
        # This is what makes ROLLBACK possible when a robot is later found
        # to be faulty. It is the software equivalent of the "immutable
        # registry of spatial transactions" described in Moroncelli et al.
        self._contributions = {}

        self.step_m = config.LIDAR_RAY_STEP_M

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
