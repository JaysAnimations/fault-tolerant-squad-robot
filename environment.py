"""
environment.py
==============
Ground truth layout of the simulated facility.

FACILITY TYPE
-------------
An onshore GAS PROCESSING UNIT with an adjacent tank farm and utility block.
Chosen because it is (a) at grade, so a ground robot can traverse it,
(b) subdivided into blocks by roads, which is what industry standards
actually prescribe, and (c) rich in the specific geometry that makes
multi-robot mapping hard.

The layout follows published plant-layout practice rather than being drawn
by eye. Cite these in Chapter 3:

  * Block layout -- the site is subdivided into blocks with roads all round
    each process unit and each dyked tank enclosure, for access and safety.
  * Central pipe rack -- the preferred arrangement places the pipe rack at
    the centre of a process unit, with large vessels outboard of it and
    pumps at the outer limits of the process area.
  * Road widths -- major roads at least 6 m, minor roads at least 4 m.
  * Pipe rack width -- typically 6-12 m for a process unit rack.
  * Equipment spacing -- adjacent process vessels typically 1.5-3 m apart.
  * Siting -- buildings at the periphery, process area central, tank farm
    adjacent to process and utilities, within a bund (dyke) wall.

WHY THE GEOMETRY MATTERS
------------------------
The map is not scenery. Each feature exists to create a condition the
fault-tolerance experiment needs:

  Feature              Creates
  -------------------  --------------------------------------------------
  Pipe rack columns    a LONG ROW OF IDENTICAL OBJECTS -- the direct cause
                       of perceptual aliasing, our Byzantine fault
  Pump row             a second row of identical objects
  Storage tanks        large radio shadows and hard sensor occlusion
  Bund wall            a near-enclosed area with a single access gap
  Buildings            enclosed rooms reachable through one door
  Roads vs aisles      corridor widths from 6 m down to ~2 m

COORDINATE CONVENTION
---------------------
World coordinates in METRES, x right, y up.
Grid indices [row, col], row = y / GRID_RES_M, col = x / GRID_RES_M.
Cell 1 = solid obstacle, 0 = free space.

A 2D LiDAR mounted low on a ground robot sees the SUPPORT COLUMNS of an
elevated pipe rack, not the pipes overhead. Rack columns are therefore
modelled as discrete posts with clear space between them, which is both
physically correct and the source of the repetitive signature above.
"""

import numpy as np
import config

# --- feature classes, used ONLY for rendering -------------------------
# The physics grid stays a plain 0/1 occupancy array. These labels are a
# parallel array so the map can be drawn in colour without changing what
# the robots actually collide with or sense.
L_FREE, L_FENCE, L_RACK, L_VESSEL, L_EXCHANGER = 0, 1, 2, 3, 4
L_PUMP, L_BUILDING, L_BUND, L_TANK, L_SKID = 5, 6, 7, 8, 9

FEATURE_COLOURS = {
    L_FREE:      (0.97, 0.96, 0.94),
    L_FENCE:     (0.25, 0.27, 0.30),
    L_RACK:      (0.29, 0.45, 0.62),
    L_VESSEL:    (0.13, 0.55, 0.52),
    L_EXCHANGER: (0.85, 0.52, 0.20),
    L_PUMP:      (0.90, 0.72, 0.20),
    L_BUILDING:  (0.47, 0.36, 0.60),
    L_BUND:      (0.55, 0.42, 0.32),
    L_TANK:      (0.20, 0.42, 0.28),
    L_SKID:      (0.52, 0.54, 0.56),
}

FEATURE_NAMES = {
    L_FENCE: "perimeter fence", L_RACK: "pipe rack",
    L_VESSEL: "vessels / columns", L_EXCHANGER: "heat exchangers",
    L_PUMP: "pumps", L_BUILDING: "buildings", L_BUND: "bund wall",
    L_TANK: "storage tanks", L_SKID: "skids / packages",
}


class Facility:
    """2D occupancy grid of a gas processing unit."""

    def __init__(self):
        self.res = config.GRID_RES_M
        self.width_m = config.WORLD_WIDTH_M
        self.height_m = config.WORLD_HEIGHT_M

        self.n_cols = int(round(self.width_m / self.res))
        self.n_rows = int(round(self.height_m / self.res))

        self.grid = np.zeros((self.n_rows, self.n_cols), dtype=np.uint8)
        self.labels = np.zeros((self.n_rows, self.n_cols), dtype=np.uint8)
        self._boundary = None

        # Recorded for figures and for Chapter 3. Populated by _build_layout.
        self.features = {}

        self._build_layout()

    # =================================================================
    # Coordinate helpers
    # =================================================================
    def world_to_grid(self, x_m, y_m):
        col = np.floor(np.asarray(x_m) / self.res).astype(np.int32)
        row = np.floor(np.asarray(y_m) / self.res).astype(np.int32)
        return row, col

    def in_bounds(self, row, col):
        return (row >= 0) & (row < self.n_rows) & (col >= 0) & (col < self.n_cols)

    def is_occupied_world(self, x_m, y_m):
        row, col = self.world_to_grid(x_m, y_m)
        if not self.in_bounds(row, col):
            return True
        return bool(self.grid[row, col])

    # =================================================================
    # Shape primitives
    # =================================================================
    def _rect(self, x0, y0, x1, y1, value=1, label=None):
        r0, c0 = self.world_to_grid(x0, y0)
        r1, c1 = self.world_to_grid(x1, y1)
        r0, r1 = max(0, min(r0, r1)), min(self.n_rows, max(r0, r1))
        c0, c1 = max(0, min(c0, c1)), min(self.n_cols, max(c0, c1))
        self.grid[r0:r1, c0:c1] = value
        # labels are cosmetic only -- they never touch collision or sensing
        self.labels[r0:r1, c0:c1] = (label if label is not None else L_SKID) if value else L_FREE

    def _circle(self, cx, cy, radius_m, value=1, label=None):
        rows = np.arange(self.n_rows)[:, None]
        cols = np.arange(self.n_cols)[None, :]
        ys = (rows + 0.5) * self.res
        xs = (cols + 0.5) * self.res
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius_m ** 2
        self.grid[mask] = value
        self.labels[mask] = (label if label is not None else L_VESSEL) if value else L_FREE

    def _hollow_box(self, x0, y0, x1, y1, wall_t=0.4, label=None):
        """A building: solid walls, open interior."""
        self._rect(x0, y0, x1, y1, 1, label=label or L_BUILDING)
        self._rect(x0 + wall_t, y0 + wall_t, x1 - wall_t, y1 - wall_t, 0)

    def _column_row(self, x_start, x_end, y, spacing, size=0.6):
        """
        A row of evenly spaced square columns -- a pipe rack seen by a
        ground-level 2D LiDAR. Returns the column centres.

        THE REPETITION IS THE POINT. Every column is geometrically
        identical to every other, which is exactly the condition under
        which a robot mis-associates its position by a whole bay.
        """
        centres = []
        x = x_start
        while x <= x_end:
            self._rect(x - size / 2, y - size / 2, x + size / 2, y + size / 2, label=L_RACK)
            centres.append((x, y))
            x += spacing
        return centres

    # =================================================================
    # Layout
    # =================================================================
    def _build_layout(self):
        W, H = self.width_m, self.height_m

        # --- perimeter fence -----------------------------------------
        t = 0.4
        self._rect(0, 0, W, t, label=L_FENCE)
        self._rect(0, H - t, W, H, label=L_FENCE)
        self._rect(0, 0, t, H, label=L_FENCE)
        self._rect(W - t, 0, W, H, label=L_FENCE)

        # A 6 m perimeter road runs inside the fence, so nothing is built
        # closer than 6.4 m to the boundary. Major roads must be >= 6 m.

        # --- MAIN PIPE RACK ------------------------------------------
        # East-west across the whole unit at mid height, splitting the
        # site into a north process half and a south utility/storage half.
        # Columns at 6 m centres, so robots pass freely between them.
        self.features["rack_columns"] = self._column_row(
            x_start=9.0, x_end=71.0, y=25.5, spacing=6.0, size=0.6)

        # A branch rack serving the process unit, running north.
        self.features["rack_branch"] = self._column_row(
            x_start=33.0, x_end=33.0, y=25.5, spacing=6.0)
        for yy in (31.0, 37.0, 43.0):
            self._rect(32.7, yy - 0.3, 33.3, yy + 0.3, label=L_RACK)

        # --- SECONDARY PIPE RACK (north) -----------------------------
        # Links the process unit to the compressor house. A second row of
        # identical columns, which doubles the opportunity for a robot to
        # mis-associate one bay with another.
        self.features["rack_columns_north"] = self._column_row(
            x_start=24.0, x_end=72.0, y=49.5, spacing=6.0, size=0.6)

        # --- STORE / WORKSHOP (north-west periphery) -----------------
        self._hollow_box(7.0, 45.0, 19.0, 52.5)
        self._rect(12.0, 45.0, 14.5, 45.4, 0)          # south door
        self.features["store"] = (7.0, 45.0, 19.0, 52.5)

        # --- CONTROL BUILDING (periphery, per siting practice) --------
        self._hollow_box(7.0, 32.0, 13.5, 40.0)
        self._rect(13.1, 35.0, 13.5, 37.0, 0)          # doorway, east wall
        self.features["control_building"] = (7.0, 32.0, 13.5, 40.0)

        # --- PROCESS UNIT (north-west block) -------------------------
        # Large vessels outboard of the rack; pumps at the outer limit.
        self._circle(20.0, 43.0, 2.2, label=L_VESSEL)  # fractionation column
        self._circle(26.5, 43.0, 2.0, label=L_VESSEL)  # contactor
        self._circle(32.0, 44.0, 1.4, label=L_VESSEL)  # separator
        self._circle(36.5, 44.0, 1.4, label=L_VESSEL)  # separator

        for (x0, y0) in [(17.0, 36.0), (23.0, 36.0), (29.0, 36.0)]:
            self._rect(x0, y0, x0 + 4.0, y0 + 2.2, label=L_EXCHANGER)

        # Pump row at the outer limit of the process area -- a second
        # sequence of identical objects.
        self.features["pump_row"] = []
        for x in np.arange(17.0, 39.0, 3.0):
            self._rect(x - 0.5, 31.5, x + 0.5, 32.5, label=L_PUMP)
            self.features["pump_row"].append((x, 32.0))

        # --- COMPRESSOR HOUSE (north-east block) ---------------------
        self._hollow_box(45.0, 33.0, 70.0, 46.0)
        self._rect(54.0, 33.0, 57.0, 33.4, 0)          # south door
        self._rect(45.0, 38.5, 45.4, 41.0, 0)          # west door
        self._rect(50.0, 36.0, 56.0, 39.0, label=L_SKID)   # compressor skid A
        self._rect(60.0, 36.0, 66.0, 39.0, label=L_SKID)   # compressor skid B
        self.features["compressor_house"] = (45.0, 33.0, 70.0, 46.0)

        # --- TANK FARM (south-east block, inside a bund) -------------
        # Dyked enclosure with a single access gap, per storage practice.
        bx0, by0, bx1, by1 = 43.0, 6.5, 73.0, 21.0
        self._rect(bx0, by0, bx1, by1, label=L_BUND)
        self._rect(bx0 + 0.5, by0 + 0.5, bx1 - 0.5, by1 - 0.5, 0)
        self._rect(bx0, 12.5, bx0 + 0.5, 15.5, 0)      # access gap, west wall
        self.features["bund"] = (bx0, by0, bx1, by1)

        self.features["tanks"] = []
        for (cx, cy) in [(51.0, 13.75), (59.0, 13.75), (67.0, 13.75)]:
            self._circle(cx, cy, 3.0, label=L_TANK)
            self.features["tanks"].append((cx, cy, 3.0))

        # --- UTILITY AREA (south-west block) -------------------------
        for (x0, y0) in [(9.0, 9.0), (15.5, 9.0), (22.0, 9.0)]:
            self._rect(x0, y0, x0 + 4.5, y0 + 3.0, label=L_SKID)

        self._rect(9.0, 15.5, 14.0, 18.5, label=L_SKID)   # air compressor package
        self._rect(18.0, 15.5, 23.0, 18.5, label=L_SKID)  # cooling water skid

        # Electrical substation -- enclosed, one door
        self._hollow_box(29.0, 8.5, 38.0, 15.0)
        self._rect(32.5, 14.6, 35.0, 15.0, 0)          # north door
        self.features["substation"] = (29.0, 8.5, 38.0, 15.0)

        # Firewater pumphouse
        self._rect(29.0, 18.0, 35.0, 21.5, label=L_SKID)

    # =================================================================
    # Metrics support
    # =================================================================
    def render_rgb(self, dim_free=False):
        """
        Colour image of the facility for figures and video.

        PURELY COSMETIC. The robots sense and collide against self.grid,
        which is untouched by this. If a panel asks whether the colours
        affect the simulation, the answer is no -- self.labels is a parallel
        array built only for drawing.
        """
        img = np.zeros((self.n_rows, self.n_cols, 3), dtype=np.float32)
        for lab, rgb in FEATURE_COLOURS.items():
            img[self.labels == lab] = rgb
        if dim_free:
            img[self.labels == L_FREE] *= 0.93

        # Subtle edge shading so adjacent equipment reads as separate
        # objects rather than one blob.
        occ = self.grid == 1
        edge = np.zeros_like(occ)
        edge[1:, :] |= occ[1:, :] & ~occ[:-1, :]
        edge[:, 1:] |= occ[:, 1:] & ~occ[:, :-1]
        img[edge] *= 0.72
        return img

    def legend_handles(self):
        """matplotlib patches for a facility legend."""
        from matplotlib.patches import Patch
        return [Patch(facecolor=FEATURE_COLOURS[k], edgecolor="0.3", label=v)
                for k, v in FEATURE_NAMES.items()]

    def boundary_mask(self):
        """Obstacle cells touching free space -- the surfaces a LiDAR can see."""
        if self._boundary is None:
            occ = self.grid == 1
            free = ~occ
            nb = np.zeros_like(free)
            nb[1:, :] |= free[:-1, :]
            nb[:-1, :] |= free[1:, :]
            nb[:, 1:] |= free[:, :-1]
            nb[:, :-1] |= free[:, 1:]
            self._boundary = occ & nb
        return self._boundary

    def validate_waypoints(self, waypoints, clearance_m=None):
        """Fail loudly if any waypoint sits inside or too near an obstacle."""
        if clearance_m is None:
            clearance_m = config.ROBOT_RADIUS_M + 0.15
        bad = [(i, x, y) for i, (x, y) in enumerate(waypoints)
               if not self._has_clearance(x, y, clearance_m)]
        if bad:
            lines = "\n".join(f"    waypoint[{i}] = ({x}, {y}) is inside an "
                              f"obstacle or too close to one" for i, x, y in bad)
            raise ValueError("Invalid waypoints:\n" + lines)
        return True

    def free_cell_count(self):
        return int((self.grid == 0).sum())

    def free_area_m2(self):
        return self.free_cell_count() * self.res ** 2

    def find_free_pose(self, rng, margin_m=0.6):
        for _ in range(10000):
            x = rng.uniform(margin_m, self.width_m - margin_m)
            y = rng.uniform(margin_m, self.height_m - margin_m)
            if self._has_clearance(x, y, config.ROBOT_RADIUS_M + 0.3):
                return x, y, rng.uniform(-np.pi, np.pi)
        raise RuntimeError("Could not find a free spawn pose")

    def _has_clearance(self, x, y, radius_m):
        r0, c0 = self.world_to_grid(x - radius_m, y - radius_m)
        r1, c1 = self.world_to_grid(x + radius_m, y + radius_m)
        if r0 < 0 or c0 < 0 or r1 >= self.n_rows or c1 >= self.n_cols:
            return False
        return not self.grid[r0:r1 + 1, c0:c1 + 1].any()


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fac = Facility()
    print(f"Extent          : {fac.width_m:.0f} m x {fac.height_m:.0f} m")
    print(f"Grid            : {fac.n_rows} rows x {fac.n_cols} cols")
    print(f"Free area       : {fac.free_area_m2():.0f} m^2")
    print(f"Obstacle ratio  : {fac.grid.mean()*100:.1f} %")
    print(f"Rack columns    : {len(fac.features['rack_columns'])}")
    print(f"Pumps           : {len(fac.features['pump_row'])}")
    print(f"Storage tanks   : {len(fac.features['tanks'])}")

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.imshow(fac.render_rgb(), origin="lower",
              extent=[0, fac.width_m, 0, fac.height_m], interpolation="nearest")

    labels = [
        (10.2, 36.0, "CONTROL\nBUILDING"),
        (26.0, 43.5, "PROCESS UNIT"),
        (27.0, 33.5, "pump row"),
        (40.0, 24.0, "MAIN PIPE RACK"),
        (57.0, 44.0, "COMPRESSOR HOUSE"),
        (58.0, 20.0, "TANK FARM (bunded)"),
        (16.0, 20.5, "UTILITY AREA"),
        (33.5, 11.5, "SUBSTATION"),
    ]
    for x, y, s in labels:
        ax.text(x, y, s, fontsize=8, ha="center", va="center",
                color="tab:blue", weight="bold")

    ax.set_title("Simulated gas processing unit — ground truth")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(handles=fac.legend_handles(), loc="upper left",
              bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig("facility_ground_truth.png", dpi=130)
    print("\nSaved facility_ground_truth.png")
