"""
frontier.py
===========
Frontier-based exploration -- how the robot decides where to go next.

WHAT A FRONTIER IS
------------------
A frontier is the boundary between what the robot has mapped and what it
has not: a cell it knows is free, sitting next to a cell it knows nothing
about. Standing on a frontier and looking outward is, by definition, the
only place new information can come from.

The algorithm is therefore embarrassingly simple:

    while frontiers remain:
        pick the most worthwhile one
        drive to it
        the map grows, old frontiers vanish, new ones appear

When no frontiers remain, every reachable place has been seen and the
mission is complete. Nobody wrote a route. That is the difference between
exploration and following a tour, and it is what Step 1 delivers.

WHY WE FLOOD FROM THE ROBOT
---------------------------
The planner floods outward from a goal to find a path to it. Here we flood
outward from the ROBOT instead, which gives the travel distance to every
frontier on the map in a single pass. Ranking twenty candidate frontiers
therefore costs one flood, not twenty path searches.

This is the payoff for choosing a wavefront over A*, and it is worth saying
out loud in Chapter 3.

CHOOSING BETWEEN FRONTIERS
--------------------------
Nearest-first is the classic choice and it is greedy: the robot ping-pongs
between two mediocre frontiers, paying travel cost each time. Largest-first
sends it across the site chasing big unexplored regions and wastes energy.

We score both together:

    score = size - w * distance

so a large frontier is worth travelling for and a small one is not. Since
energy is the metric this project reports, a selection rule that trades
information against travel is the honest one.

HYSTERESIS
----------
The robot commits to a frontier until it arrives, the frontier is consumed,
or it becomes unreachable. Re-deciding every step makes it oscillate between
two similar candidates and burn the mission in turning energy. Commitment is
not laziness here -- it is what makes the behaviour stable.
"""

import numpy as np
from collections import deque

import config


class FrontierExplorer:
    """
    Chooses exploration goals from the robot's own occupancy map.

    Owns no map and no robot -- you hand it both each time you ask for a
    goal. That keeps it usable unchanged when three robots exist in Step 2.
    """

    def __init__(self, planner,
                 min_size_cells=config.FRONTIER_MIN_SIZE_CELLS,
                 distance_weight=config.FRONTIER_DISTANCE_WEIGHT,
                 unknown_fraction=config.FRONTIER_UNKNOWN_FRACTION):
        self.p = planner
        self.min_size = min_size_cells
        self.w = distance_weight
        self.unknown_frac = unknown_fraction

        self.goal_xy = None          # the frontier currently committed to
        self.goal_rc = None
        self.exhausted = False       # True once no frontiers remain

        # Frontiers the robot tried and could not reach. Without this it
        # re-selects the same unreachable frontier forever: the frontier is
        # still there, still looks best, and the robot never learns.
        self.banned = {}             # (row, col) -> step it expires
        self.committed_at = -10**9   # step the current goal was chosen

    # -----------------------------------------------------------------
    def coarse_state(self, occupancy_map):
        """
        Classify every coarse cell as blocked / known-free / unknown.

        Working on the planner's coarse grid rather than the full-resolution
        map keeps this cheap enough to run repeatedly: 137 x 200 cells
        instead of 550 x 800.
        """
        p = self.p
        L = occupancy_map.L
        r, c = p.c_rows * p.ds, p.c_cols * p.ds

        occ = (L[:r, :c] >= config.LOG_ODDS_OCC_THRESHOLD)
        unk = ((L[:r, :c] > config.LOG_ODDS_FREE_THRESHOLD) &
               (L[:r, :c] < config.LOG_ODDS_OCC_THRESHOLD))

        shape = (p.c_rows, p.ds, p.c_cols, p.ds)
        blocked = occ.reshape(shape).any(axis=(1, 3))
        unk_frac = unk.reshape(shape).mean(axis=(1, 3))

        unknown = unk_frac >= self.unknown_frac
        known_free = (~blocked) & (~unknown)
        return blocked, known_free, unknown

    # -----------------------------------------------------------------
    def find_frontiers(self, occupancy_map):
        """
        Return [(row, col, size), ...] -- one entry per frontier region,
        giving its centroid on the coarse grid and how many cells it spans.

        Size matters because a one-cell frontier is usually sensor noise at
        the edge of the map, while a twenty-cell frontier is a corridor the
        robot has not entered yet.
        """
        blocked, known_free, unknown = self.coarse_state(occupancy_map)

        # A frontier cell is known-free and touches unknown space.
        nb_unknown = np.zeros_like(unknown)
        nb_unknown[1:, :] |= unknown[:-1, :]
        nb_unknown[:-1, :] |= unknown[1:, :]
        nb_unknown[:, 1:] |= unknown[:, :-1]
        nb_unknown[:, :-1] |= unknown[:, 1:]
        frontier = known_free & nb_unknown

        # --- group touching frontier cells into regions (BFS labelling) ---
        regions = []
        seen = np.zeros_like(frontier)
        rows, cols = np.nonzero(frontier)
        for r0, c0 in zip(rows, cols):
            if seen[r0, c0]:
                continue
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            cells = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if (0 <= rr < frontier.shape[0]
                                and 0 <= cc < frontier.shape[1]
                                and frontier[rr, cc] and not seen[rr, cc]):
                            seen[rr, cc] = True
                            q.append((rr, cc))
            if len(cells) >= self.min_size:
                arr = np.array(cells)
                cy, cx = arr[:, 0].mean(), arr[:, 1].mean()
                # Use the frontier cell CLOSEST to the centroid, not the
                # centroid itself. THIS WAS A BUG: the centroid of a curved
                # or L-shaped frontier region need not lie on the region at
                # all, so the chosen goal was often not a frontier cell. It
                # then failed its own validity check on the very next step,
                # the robot re-selected, and a 43 ms selection ran ten times
                # a second for the whole mission.
                j = int(np.argmin((arr[:, 0] - cy) ** 2 + (arr[:, 1] - cx) ** 2))
                regions.append((int(arr[j, 0]), int(arr[j, 1]), len(cells)))
        return regions

    # -----------------------------------------------------------------
    def ban(self, rc, until_step):
        """Temporarily rule out a frontier the robot failed to reach."""
        if rc is not None:
            self.banned[tuple(rc)] = until_step

    def _is_banned(self, r, c, step):
        for (br, bc), expiry in self.banned.items():
            if step < expiry and abs(br - r) <= 2 and abs(bc - c) <= 2:
                return True
        return False

    def select(self, occupancy_map, robot_xy, step=0):
        """
        Pick the best frontier and return its world position, or None when
        exploration is complete.
        """
        regions = self.find_frontiers(occupancy_map)
        if not regions:
            self.exhausted = True
            self.goal_xy = None
            return None

        # ONE flood from the robot gives travel distance to every candidate.
        blocked = self.p._coarse_blocked(occupancy_map)
        sr, sc = self.p._to_coarse(*robot_xy)
        if not (0 <= sr < self.p.c_rows and 0 <= sc < self.p.c_cols):
            return None
        if blocked[sr, sc]:
            snapped = self.p._nearest_free(~blocked, sr, sc)
            if snapped is None:
                return None
            sr, sc = snapped
        dist = self.p._flood(blocked, [(sr, sc)])

        # Two passes. First consider only frontiers worth travelling to;
        # if none qualify, accept the close ones so exploration does not
        # terminate early.
        #
        # WHY A MINIMUM DISTANCE EXISTS AT ALL: the LiDAR sees 8 m. A
        # frontier two metres away will be consumed by simply looking in its
        # direction, so driving to it accomplishes nothing. Without this
        # filter the robot crawled from one nearby frontier to the next,
        # re-selecting every few steps and never committing to a real goal.
        for min_travel in (config.FRONTIER_MIN_TRAVEL_CELLS, 0):
            best, best_score = None, -np.inf
            for (r, c, size) in regions:
                if self._is_banned(r, c, step):
                    continue
                d = dist[r, c]
                if d < 0:
                    # The frontier cell may sit where the flood cannot
                    # reach. Fall back to the nearest reachable cell to it.
                    d = self._nearest_reachable_distance(dist, r, c)
                    if d is None:
                        continue
                if d < min_travel:
                    continue
                score = size - self.w * d
                if score > best_score:
                    best, best_score = (r, c), score
            if best is not None:
                break

        if best is None:
            self.exhausted = True
            self.goal_xy = None
            return None

        self.goal_rc = best
        self.goal_xy = self.p._to_world(*best)
        self.committed_at = step
        return self.goal_xy

    @staticmethod
    def _nearest_reachable_distance(dist, r, c, max_cells=6):
        reach = np.argwhere(dist >= 0)
        if len(reach) == 0:
            return None
        d2 = (reach[:, 0] - r) ** 2 + (reach[:, 1] - c) ** 2
        j = int(np.argmin(d2))
        if d2[j] > max_cells ** 2:
            return None
        return int(dist[reach[j, 0], reach[j, 1]])

    # -----------------------------------------------------------------
    def goal_still_valid(self, occupancy_map, robot_xy, step=0):
        """
        A committed goal stays valid until the robot arrives or the frontier
        is consumed by the growing map. This is the hysteresis that stops
        the robot oscillating between similar candidates.
        """
        if self.goal_xy is None:
            return False
        if np.hypot(self.goal_xy[0] - robot_xy[0],
                    self.goal_xy[1] - robot_xy[1]) < config.FRONTIER_REACHED_M:
            return False
        # Minimum commitment. Selection is expensive and re-deciding
        # immediately makes the robot oscillate; hold the goal briefly even
        # if it looks marginal.
        if step - self.committed_at < config.FRONTIER_MIN_COMMIT_STEPS:
            return True

        _, known_free, unknown = self.coarse_state(occupancy_map)
        r, c = self.goal_rc
        nb = unknown[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
        return bool(nb.any())
