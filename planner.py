"""
planner.py
==========
Path planning by BFS wavefront over the robot's OWN occupancy map.

WHY A PLANNER IS NEEDED (this is a result, not just plumbing)
------------------------------------------------------------
A purely reactive controller steers toward the goal and swerves around
whatever the LiDAR sees. It cannot escape a local minimum: if a wall lies
between it and the goal, it pushes against that wall indefinitely. In an
open room this rarely matters. In a real process facility -- buildings with
a single door, a tank farm inside a bund wall with one access gap, aisles
two metres wide -- it fails constantly.

Measured on this facility: reactive control alone reached ~74 % coverage and
could not enter any enclosed area. Quote that number in Chapter 4 as the
motivation for the planner.

HOW THE WAVEFRONT WORKS
-----------------------
Flood outward from the GOAL one cell at a time, recording how many steps it
took to reach each cell. That gives a distance field over the whole map in a
single pass. To travel to the goal from anywhere, repeatedly step to the
neighbouring cell with the lower value -- like water running downhill.

Two properties matter for this project:

  1. It cannot get stuck. If a route exists the wavefront finds it, because
     it explores every reachable cell before giving up.
  2. Flooding from the goal gives distance-to-goal from EVERY cell at once.
     In Step 2 we flood from the ROBOT instead, which gives the travel cost
     to every frontier in one pass -- so ranking frontiers becomes free.
     That is why we use a wavefront rather than A*.

OPTIMISM ABOUT UNKNOWN SPACE
----------------------------
Unknown cells are treated as traversable. A robot exploring an unknown
facility must be willing to drive into unmapped space or it would never
explore at all. If the space turns out to be blocked, the next replan
routes around it.
"""

import numpy as np
from collections import deque

import config


class WavefrontPlanner:

    def __init__(self, facility, downsample=4,
                 inflate_cells=config.PLANNER_INFLATE_CELLS):
        """
        downsample     : plan on a coarser grid for speed. 4 turns an
                         800x550 grid into 200x137, ~16x less work. The path
                         is still followed at full resolution.
        inflate_cells  : grow obstacles by this many coarse cells so the
                         planned path keeps clear of walls. Two coarse cells
                         at downsample 4 is 0.8 m. Measured on this facility:
                         inflate=1 gave 0.4 m clearance and 3026 collisions;
                         inflate=2 keeps 99.2 % of the site reachable and
                         cuts collisions sharply; inflate=3 fragments the
                         site into 5 disconnected regions. One coarse cell
                         at downsample 4 is 0.4 m, comfortably more than the
                         0.25 m robot radius.
        """
        self.ds = downsample
        self.inflate = inflate_cells
        self.res = facility.res
        self.c_rows = facility.n_rows // downsample
        self.c_cols = facility.n_cols // downsample

    # -----------------------------------------------------------------
    def _coarse_blocked(self, occupancy_map, inflate=None):
        """
        Build the coarse traversability grid from the robot's own belief.

        A coarse cell is blocked if ANY fine cell inside it is believed
        occupied. Being pessimistic about obstacles while optimistic about
        unknown space is the correct bias for a mapping robot.

        `inflate` overrides the instance's margin. plan() uses that to
        retry a failed route with a tighter margin.
        """
        occ = occupancy_map.L >= config.LOG_ODDS_OCC_THRESHOLD

        r = self.c_rows * self.ds
        c = self.c_cols * self.ds
        blocks = occ[:r, :c].reshape(self.c_rows, self.ds, self.c_cols, self.ds)
        blocked = blocks.any(axis=(1, 3))

        if inflate is None:
            inflate = self.inflate
        for _ in range(inflate):
            g = blocked.copy()
            g[1:, :] |= blocked[:-1, :]
            g[:-1, :] |= blocked[1:, :]
            g[:, 1:] |= blocked[:, :-1]
            g[:, :-1] |= blocked[:, 1:]
            blocked = g
        return blocked

    def _to_coarse(self, x_m, y_m):
        return (int(y_m / self.res) // self.ds, int(x_m / self.res) // self.ds)

    def _to_world(self, row, col):
        return ((col + 0.5) * self.ds * self.res,
                (row + 0.5) * self.ds * self.res)

    # -----------------------------------------------------------------
    def _nearest_free(self, free, r, c, max_cells=8):
        """Nearest traversable coarse cell to (r, c), or None if too far."""
        cand = np.argwhere(free)
        if len(cand) == 0:
            return None
        d2 = (cand[:, 0] - r) ** 2 + (cand[:, 1] - c) ** 2
        j = int(np.argmin(d2))
        if d2[j] > max_cells ** 2:
            return None
        return int(cand[j, 0]), int(cand[j, 1])

    # -----------------------------------------------------------------
    def _coarse_solid(self, occupancy_map):
        """
        Coarse cells containing a believed obstacle, with NO margin added.

        The difference between this and _coarse_blocked is the difference
        between "there is a wall here" and "I would rather not drive here".
        A robot can stand in the second; it cannot stand in the first.
        """
        occ = occupancy_map.L >= config.LOG_ODDS_OCC_THRESHOLD
        r = self.c_rows * self.ds
        c = self.c_cols * self.ds
        return occ[:r, :c].reshape(self.c_rows, self.ds,
                                   self.c_cols, self.ds).any(axis=(1, 3))

    def _snap_start(self, occupancy_map, blocked, dist, sr, sc, max_cells=6):
        """
        Move the start to a nearby cell the goal flood actually reached.

        THE SNAP MUST BE SOMEWHERE THE ROBOT CAN WALK TO. An earlier version
        took the nearest reachable cell by straight-line distance, and that
        is wrong in a way that took a three-robot run to expose: a robot
        standing just inside the control building had its start snapped to a
        cell two coarse cells away -- on the far side of the north wall,
        because at the wide margin the sealed doorway meant nothing inside
        the room had been reached by the flood. The planner then returned a
        route that began outside the building, the robot drove at a wall for
        the rest of the mission, and it inspected nothing.

        So the search is a bounded flood outward from the robot's own cell
        rather than a nearest-neighbour lookup. It may pass through cells
        that are blocked only by the safety margin -- the robot is
        demonstrably standing in one of those -- but never through a cell
        that actually contains an obstacle. If it cannot get out, it returns
        None, and plan() then retries at the tighter margin, which is what
        opens the doorway.
        """
        solid = self._coarse_solid(occupancy_map)

        q = deque([(sr, sc, 0)])
        seen = {(sr, sc)}
        while q:
            r, c, d = q.popleft()
            if dist[r, c] >= 0:
                return r, c
            if d >= max_cells:
                continue
            # Eight-connected on purpose. The search this replaced measured
            # straight-line distance and accepted anything within 6 cells,
            # so a four-connected flood of the same radius would reach less
            # ground than before and reject snaps that used to work -- which
            # cost the scripted tour 0.2 of its surface F1 before it was
            # caught. Eight-connectivity restores the old reach; the useful
            # restriction is the wall, not the radius.
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.c_rows and 0 <= nc < self.c_cols):
                    continue
                if (nr, nc) in seen or solid[nr, nc]:
                    continue
                seen.add((nr, nc))
                q.append((nr, nc, d + 1))
        return None

    # -----------------------------------------------------------------
    def _flood(self, blocked, sources):
        """BFS flood over an already-computed blocked grid."""
        dist = np.full((self.c_rows, self.c_cols), -1, dtype=np.int32)
        q = deque()
        for (r, c) in sources:
            if 0 <= r < self.c_rows and 0 <= c < self.c_cols and not blocked[r, c]:
                dist[r, c] = 0
                q.append((r, c))
        while q:
            r, c = q.popleft()
            d = dist[r, c] + 1
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.c_rows and 0 <= nc < self.c_cols
                        and not blocked[nr, nc] and dist[nr, nc] < 0):
                    dist[nr, nc] = d
                    q.append((nr, nc))
        return dist

    # -----------------------------------------------------------------
    def distance_field(self, occupancy_map, sources):
        """
        BFS flood from one or more source cells.

        Returns an int array of step counts, -1 where unreachable. Passing
        several sources floods from all of them at once -- used in Step 2
        for multi-frontier ranking.
        """
        return self._flood(self._coarse_blocked(occupancy_map), sources)

    # -----------------------------------------------------------------
    def plan(self, occupancy_map, start_xy, goal_xy):
        """
        Plan a route from start to goal.

        Returns a list of (x, y) waypoints in metres, or None if the goal is
        unreachable given what the robot currently believes.

        TWO MARGINS, TRIED IN ORDER
        ---------------------------
        The comfortable margin is tried first. If it finds no route at all,
        the tight margin is tried before giving up.

        This exists because of a specific, measurable failure. Obstacles are
        inflated by 0.8 m so paths do not scrape walls -- but that is 0.8 m
        from EACH side, so a 2.0 m doorway is reduced to 0.4 m and the 0.4 m
        coarse grid rounds it shut. The planner then reported the control
        building unreachable, although demo_single.py drives into it. The
        caller's only recourse was to steer straight at the goal and hope,
        which meant pushing against the outside wall.

        A margin is a preference, not a law. Where the geometry leaves no
        comfortable route, take the tight one; 0.4 m still clears a 0.25 m
        robot. Where a comfortable route exists this changes nothing, since
        the second attempt only runs after the first returns None.
        """
        path = self._plan_at(occupancy_map, start_xy, goal_xy, self.inflate)
        if path is None and self.inflate > config.PLANNER_TIGHT_INFLATE_CELLS:
            path = self._plan_at(occupancy_map, start_xy, goal_xy,
                                 config.PLANNER_TIGHT_INFLATE_CELLS)
        return path

    def _plan_at(self, occupancy_map, start_xy, goal_xy, inflate):
        """
        One planning attempt at a given obstacle margin.

        We flood from the GOAL, then walk downhill from the start, so the
        field stays valid wherever the robot drifts to.
        """
        blocked = self._coarse_blocked(occupancy_map, inflate)

        gr, gc = self._to_coarse(*goal_xy)
        sr, sc = self._to_coarse(*start_xy)
        if not (0 <= gr < self.c_rows and 0 <= gc < self.c_cols):
            return None
        if not (0 <= sr < self.c_rows and 0 <= sc < self.c_cols):
            return None

        # --- snap the GOAL to a traversable cell -------------------------
        # Coarse cells are 0.4 m and a cell counts as blocked if ANY fine
        # cell inside it is occupied, so a goal placed near a wall often
        # lands in a blocked cell even though the robot could stand there.
        if blocked[gr, gc]:
            snapped = self._nearest_free(~blocked, gr, gc)
            if snapped is None:
                return None
            gr, gc = snapped

        dist = self._flood(blocked, [(gr, gc)])

        # --- snap the START likewise -------------------------------------
        # THIS WAS A REAL BUG AND IT COST AN ENTIRE RUN. The robot spends
        # most of its life driving close to walls, which puts its own coarse
        # cell in the blocked set. The planner then returned None, the caller
        # saw an empty path, and asked for a replan on the very next step --
        # so a 40 ms planner ran 10 times a second, for the whole mission.
        #
        # A robot that is briefly inside an inflated obstacle is not lost.
        # Snap it to the nearest reachable cell and carry on. Only give up if
        # it is genuinely stranded.
        if dist[sr, sc] < 0:
            snapped = self._snap_start(occupancy_map, blocked, dist, sr, sc)
            if snapped is None:
                return None
            sr, sc = snapped

        # --- walk downhill from start to goal ---
        path = []
        r, c = sr, sc
        guard = self.c_rows * self.c_cols
        while dist[r, c] > 0 and guard > 0:
            guard -= 1
            best, best_d = None, dist[r, c]
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.c_rows and 0 <= nc < self.c_cols
                        and dist[nr, nc] >= 0 and dist[nr, nc] < best_d):
                    best, best_d = (nr, nc), dist[nr, nc]
            if best is None:
                break
            r, c = best
            path.append(self._to_world(r, c))

        if not path:
            return [goal_xy]
        path.append(goal_xy)
        return self._simplify(path)

    @staticmethod
    def _simplify(path, keep_every=3):
        """
        Thin the path. A cell-by-cell path makes the controller twitch; a
        node every few cells gives it room to drive smoothly.
        """
        if len(path) <= 2:
            return path
        out = [path[0]]
        out += [p for i, p in enumerate(path[1:-1], 1) if i % keep_every == 0]
        out.append(path[-1])
        return out
