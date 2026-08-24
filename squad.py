"""
squad.py
========
One robot's private world, and the auction that stops three of them doing
the same job. Step 2.

WHAT "PRIVATE" MEANS HERE
------------------------
Each SquadMember owns:

    its robot        physics, energy, health flags
    its OccupancyGrid  its own belief about the facility
    its inbox        messages that reached it
    its claim table  what it has heard other robots say they are doing
    its done set     the points it believes are finished

Nothing in this file, or anywhere else in the project, holds a shared map
or a shared task list. Two robots disagree about the state of the mission
whenever they have been out of radio range of each other, and that is
correct rather than a bug -- it is the condition the fault-tolerance work
in Steps 3 to 5 has to survive.

The inspection point objects ARE shared, because the operator issued the
same list to every robot. But a robot never reads `point.visited` to make
a decision: that flag is the mission record, written for the report. A
robot decides from its own `done` set, which only grows when it inspects
something itself or hears that somebody else did.

THE AUCTION
-----------
Before committing to a point, a robot works out what that point would
cost it -- travel distance over its OWN map -- and broadcasts it. If it
hears that another robot can reach the same point more cheaply, it leaves
that point alone and takes the next best.

Claims are LEASES. The holder re-broadcasts while it is still driving
there, and a claim that stops being refreshed lapses after
CLAIM_TIMEOUT_STEPS. Without that, a robot that died holding six claims
would block those six points for the rest of the mission -- which is
exactly the failure Step 5 exists to prevent, so it must not be baked
into Step 2.
"""

import numpy as np

import config
from sensors import Lidar2D
from mapping import OccupancyGrid
from robot import Robot
from planner import WavefrontPlanner
from control import choose_velocity
from inspection import reachability_planner, inspection_distance
from comms import claim_message, visited_message, map_message


def status_code(robot):
    """
    Health flags packed into one byte, so the trajectory trace stays small.

        0        dead
        bit 0    alive
        bit 1    motors failed
        bit 2    sensor failed
        bit 3    radio failed

    A healthy robot is 1.
    """
    if not robot.alive:
        return 0
    code = 1
    if not robot.mobile:
        code |= 2
    if not robot.sensing:
        code |= 4
    if not robot.connected:
        code |= 8
    return code


class TrajectoryTrace:
    """
    Just enough of a mission to replay it as video later, without paying
    to run the simulation again.

    One sample per robot per second (TRACE_EVERY_N_STEPS at 10 Hz). Stored
    as columns of float32 and written with np.savez_compressed, so a
    fifteen-minute three-robot mission is tens of kilobytes rather than
    megabytes. Deliberately thin: position, heading, believed position,
    health, battery, energy. Anything a replay cannot draw does not belong
    in here.

    Believed position is included alongside true position because the gap
    between the two IS the story in a video -- it is what odometry drift
    looks like, and under the wrong-position fault it is what the audience
    is meant to see.
    """

    COLUMNS = ("step", "id", "x", "y", "theta", "bx", "by",
               "status", "battery_j", "energy_j")

    def __init__(self):
        self.rows = []

    def record(self, step, squad):
        for m in squad:
            r = m.robot
            self.rows.append((step, r.id, r.x, r.y, r.theta, r.bx, r.by,
                              status_code(r), r.battery_j, r.total_energy_j))

    def to_arrays(self):
        data = np.asarray(self.rows, dtype=np.float32)
        return {name: data[:, i] for i, name in enumerate(self.COLUMNS)}

    def save(self, path):
        np.savez_compressed(path, **self.to_arrays())
        return path

    def __len__(self):
        return len(self.rows)


class SquadMember:
    """One robot and everything it privately owns."""

    def __init__(self, robot_id, pose, facility, seed, use_prior=True,
                 use_auction=True):
        x, y, theta = pose
        # use_auction=False makes the robot ignore what everybody else says
        # they are doing, while still sharing maps. It exists as the control
        # case: without it there is no way to show that the auction is doing
        # anything, only that three robots are faster than one.
        self.use_auction = use_auction

        # Each robot draws its noise from its own stream, so that changing
        # what one robot does cannot shift another robot's sensor readings.
        rng = np.random.default_rng([seed, config.RNG_STREAM_ROBOT, robot_id])

        self.id = robot_id
        self.robot = Robot(robot_id=robot_id, x=x, y=y, theta=theta, rng=rng)
        self.rng = rng

        # --- private belief ------------------------------------------
        self.grid = OccupancyGrid(facility, owner_id=robot_id)
        if use_prior:
            self.grid.seed_prior(facility.documented_grid)

        # --- its own sensor and its own planners ---------------------
        self.lidar = Lidar2D()
        self.planner = WavefrontPlanner(facility)
        self.ranker = reachability_planner(facility)

        # --- what it has heard ---------------------------------------
        self.inbox = []
        self.claims = {}      # point index -> {"by", "cost", "heard_at"}
        self.done = set()     # points it believes are finished
        self.given_up = set()  # points it tried and could not reach

        # --- navigation state ----------------------------------------
        self.target = None
        self.target_cost = None
        self.claimed_at = -10 ** 9
        self.path = []
        self.path_index = 0
        self.replan_countdown = 0
        self.plan_cooldown = 0
        self.best_dist = float("inf")
        self.steps_since_progress = 0
        self.steps_on_point = 0
        self.last_map_broadcast = -10 ** 9

        self.ranges = self.angles = self.valid = None
        self.trail_x = [self.robot.x]
        self.trail_y = [self.robot.y]

        # Why the last bid found nothing, for the report. Not used by the
        # robot for anything.
        self._no_route = []
        self._deferred = 0

        # Getting unstuck: how long motion has been commanded but refused,
        # and how much longer to keep reversing.
        self.blocked_steps = 0
        self.escape_steps = 0

    # =================================================================
    # 1. Sense and map
    # =================================================================
    def sense_and_map(self, facility, step):
        """
        Scan the world and fold the result into this robot's own map.

        The scan is taken from the TRUE pose, because that is where the
        sensor physically is. It is integrated at the BELIEVED pose,
        because that is where the robot thinks it is. Everything the robot
        does downstream flows from the second one.
        """
        if step % config.INSPECTION_SCAN_EVERY_N_STEPS != 0:
            return
        if not self.robot.sensing or not self.robot.alive:
            return

        r = self.robot
        self.ranges, self.angles, self.valid = self.lidar.scan(
            facility, r.x, r.y, r.theta, self.rng)
        r.pay_sensing(config.DT_S * config.INSPECTION_SCAN_EVERY_N_STEPS)
        self.grid.integrate_scan(r.bx, r.by, r.btheta,
                                 self.ranges, self.angles, self.valid,
                                 source_id=self.id)

    # =================================================================
    # 2. Listen
    # =================================================================
    def process_inbox(self, step):
        """
        Drain everything that arrived this step.

        Claims are only believed if they are cheaper than what we already
        hold, or if what we hold has expired. Nothing here trusts a message
        further than that -- which is the hook Step 4's Byzantine detector
        needs.
        """
        merged = 0
        for msg in self.inbox:
            kind = msg["kind"]

            if kind == "claim":
                held = self.claims.get(msg["point"])
                if (held is None
                        or self._claim_expired(held, step)
                        or msg["cost"] < held["cost"]
                        or held["by"] == msg["from"]):
                    self.claims[msg["point"]] = {"by": msg["from"],
                                                 "cost": msg["cost"],
                                                 "heard_at": step}

                # STAND DOWN IF OUTBID WHILE ALREADY DRIVING THERE.
                # Checking claims only at selection time is not enough: two
                # robots that bid in the same step both set off, and neither
                # learns better until one of them arrives. Yielding here cut
                # duplicated ground substantially. Strictly cheaper, so a
                # tie leaves the incumbent in place and two robots cannot
                # yield to each other.
                if (self.use_auction
                        and self.target is not None
                        and msg["point"] == self.target.index
                        and msg["from"] != self.id
                        and msg["cost"] < self.target_cost):
                    self.claims.pop(msg["point"], None)
                    self.claims[msg["point"]] = {"by": msg["from"],
                                                 "cost": msg["cost"],
                                                 "heard_at": step}
                    self._clear_target()

            elif kind == "visited":
                self.done.add(msg["point"])
                self.claims.pop(msg["point"], None)
                # Somebody else finished the point we were driving to.
                # Drop it and pick another rather than arriving at a job
                # already done.
                if self.target is not None and self.target.index == msg["point"]:
                    self._clear_target()

            elif kind == "map":
                self.grid.merge_from(msg["grid"], source_id=msg["from"])
                merged += 1

        self.inbox.clear()
        return merged

    @staticmethod
    def _claim_expired(claim, step):
        return (step - claim["heard_at"]) > config.CLAIM_TIMEOUT_STEPS

    # =================================================================
    # 3. Bid
    # =================================================================
    def choose_target(self, points, step, radio, squad):
        """
        Pick the cheapest point nobody else can reach more cheaply, and say
        so out loud.

        One BFS flood from this robot gives the travel cost to every
        candidate at once -- the same trick frontier.py uses, and the
        reason this project plans with a wavefront rather than A*.

        Returns True if a new target was claimed.
        """
        if self.target is not None or not self.robot.alive:
            return False

        pending = [p for p in points
                   if p.index not in self.done and p.index not in self.given_up]
        if not pending:
            return False

        me = (self.robot.bx, self.robot.by)
        blocked = self.ranker._coarse_blocked(self.grid)
        sr, sc = self.ranker._to_coarse(*me)
        if not (0 <= sr < self.ranker.c_rows and 0 <= sc < self.ranker.c_cols):
            return False

        # The robot is standing in this cell, so it is traversable whatever
        # the safety margin says about it. Flooding from it directly is both
        # cheaper and safer than snapping to the nearest unblocked cell:
        # that snap measured straight-line distance, so a robot beside a
        # wall could have its flood started on the far side of it. When that
        # happened inside a building the flood covered only that room, every
        # point outside looked unreachable, and the robot wrote off
        # twenty-eight of them in one step.
        blocked[sr, sc] = False
        dist = self.ranker._flood(blocked, [(sr, sc)])

        best, best_cost = None, None
        no_route = []
        deferred = 0
        for p in pending:
            cost = inspection_distance(self.ranker, dist, p.x, p.y)
            if cost is None:
                no_route.append(p.index)   # my own map says there is no way in
                continue

            if self.use_auction:
                held = self.claims.get(p.index)
                if (held is not None
                        and held["by"] != self.id
                        and not self._claim_expired(held, step)
                        and held["cost"] <= cost):
                    deferred += 1
                    continue       # somebody else can get there cheaper

            if best is None or cost < best_cost:
                best, best_cost = p, cost

        if best is None:
            # Nothing worth bidding for: either other robots hold everything
            # that is left, or this robot's map shows no route to any of it.
            #
            # Deliberately NOT giving up on those points here. A robot that
            # cannot route anywhere is usually wedged against a wall for a
            # few steps, not permanently defeated, and an earlier version
            # that wrote them off cost one robot twenty-eight points in a
            # single step. Points are given up only after actually trying
            # and failing to reach them; a squad that genuinely has nothing
            # left to do is caught by the stall check in demo_squad.
            self._no_route = list(no_route)
            self._deferred = deferred
            return False

        self.target = best
        self.target_cost = best_cost
        self._announce_claim(step, radio, squad)
        self.path, self.path_index = [], 0
        self.replan_countdown, self.plan_cooldown = 0, 0
        self.best_dist = float("inf")
        self.steps_since_progress, self.steps_on_point = 0, 0
        return True

    def _announce_claim(self, step, radio, squad):
        self.claims[self.target.index] = {"by": self.id,
                                          "cost": self.target_cost,
                                          "heard_at": step}
        radio.broadcast(self, claim_message(self.id, self.target.index,
                                            self.target_cost, step),
                        squad, config.COMMS_CLAIM_PACKET_KB)
        self.claimed_at = step

    def refresh_claim(self, step, radio, squad):
        """
        Renew the lease on the point being driven to.

        A claim that is never renewed lapses, which is what frees a dead
        robot's work for somebody else to pick up.
        """
        if self.target is None or not self.robot.alive:
            return
        if step - self.claimed_at < config.CLAIM_REFRESH_EVERY_N_STEPS:
            return
        self._announce_claim(step, radio, squad)

    def offer_map(self, step, radio, squad):
        """Periodically broadcast what this robot has seen for itself."""
        if not self.robot.alive:
            return
        if step - self.last_map_broadcast < config.COMMS_EXCHANGE_EVERY_N_STEPS:
            return
        radio.broadcast(self, map_message(self.id, self.grid, step),
                        squad, config.COMMS_MAP_PACKET_KB)
        self.last_map_broadcast = step

    # =================================================================
    # 4. Drive
    # =================================================================
    def _clear_target(self):
        self.target = None
        self.target_cost = None
        self.path, self.path_index = [], 0

    def drive(self, facility, step, radio, squad):
        """
        Plan to the current target, follow the plan, and decide whether the
        point has been reached or should be given up on.
        """
        if not self.robot.alive:
            return
        if self.target is None:
            # Nothing assigned: hold station. It still costs idle power,
            # which is the honest price of having no work to do.
            self.robot.step_motion(0.0, 0.0, facility)
            return

        me = (self.robot.bx, self.robot.by)
        goal = self.target.xy

        # --- plan ----------------------------------------------------
        self.replan_countdown -= 1
        self.plan_cooldown -= 1
        need_plan = (not self.path
                     or self.path_index >= len(self.path)
                     or self.replan_countdown <= 0)
        if need_plan and self.plan_cooldown <= 0:
            new_path = self.planner.plan(self.grid, me, goal)
            self.replan_countdown = config.INSPECTION_REPLAN_EVERY_N_STEPS
            if new_path:
                self.path, self.path_index = new_path, 0
            else:
                self.plan_cooldown = config.INSPECTION_PLAN_FAIL_COOLDOWN

        # --- follow --------------------------------------------------
        advanced = False
        node = None
        if self.path and self.path_index < len(self.path):
            node = self.path[self.path_index]
            if np.hypot(node[0] - self.robot.bx,
                        node[1] - self.robot.by) < config.INSPECTION_PATH_NODE_TOLERANCE_M:
                self.path_index += 1
                advanced = True
                node = self.path[min(self.path_index, len(self.path) - 1)]

        # --- progress ------------------------------------------------
        dist = float(np.hypot(goal[0] - self.robot.bx, goal[1] - self.robot.by))
        self.steps_on_point += 1
        if advanced or dist < self.best_dist - 0.05:
            self.best_dist = min(self.best_dist, dist)
            self.steps_since_progress = 0
        else:
            self.steps_since_progress += 1

        arrived = dist < config.INSPECTION_REACHED_M
        stuck = (self.steps_since_progress > config.INSPECTION_NO_PROGRESS_STEPS
                 or self.steps_on_point > config.INSPECTION_MAX_STEPS_PER_POINT)

        if arrived:
            self._mark_visited(step, radio, squad)
        elif stuck:
            self.given_up.add(self.target.index)
            self.claims.pop(self.target.index, None)
            self._clear_target()

        # --- move ----------------------------------------------------
        if self.escape_steps > 0:
            # Backing out of a wedge. Deliberately ignores the controller:
            # the controller is what drove into it.
            self.escape_steps -= 1
            v = -config.SQUAD_ESCAPE_SPEED_FRACTION * config.MAX_LINEAR_SPEED_MPS
            w = config.SQUAD_ESCAPE_TURN_RPS
        elif node is None:
            v, w = 0.0, 0.0
        else:
            v, w, _ = choose_velocity(self.robot, self.ranges, self.angles, node)

        moved = self.robot.step_motion(v, w, facility)

        # Motion was commanded and refused. Count it, and once the robot has
        # been going nowhere for long enough, reverse out.
        if moved or abs(v) < 1e-9:
            self.blocked_steps = 0
        else:
            self.blocked_steps += 1
            if self.blocked_steps > config.SQUAD_ESCAPE_AFTER_BLOCKED_STEPS:
                self.escape_steps = config.SQUAD_ESCAPE_STEPS
                self.blocked_steps = 0

    def _mark_visited(self, step, radio, squad):
        p = self.target
        p.visited = True                 # the mission record, for the report
        p.visit_step = step
        p.visited_by = self.id
        p.visit_error_m = float(np.hypot(p.x - self.robot.x,
                                         p.y - self.robot.y))
        self.done.add(p.index)
        self.claims.pop(p.index, None)
        radio.broadcast(self, visited_message(self.id, p.index, step),
                        squad, config.COMMS_CLAIM_PACKET_KB)
        self._clear_target()

    # =================================================================
    def record_trail(self):
        self.trail_x.append(self.robot.x)
        self.trail_y.append(self.robot.y)

    def observed_mask(self):
        """Cells this robot has seen for itself -- not merged from others."""
        return self.grid.own_observations() != 0.0

    def __repr__(self):
        return (f"<SquadMember {self.id} {self.robot.status()} "
                f"done={len(self.done)} target="
                f"{self.target.index if self.target else None}>")
