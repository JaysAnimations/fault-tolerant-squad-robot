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
from comms import (claim_message, visited_message, map_message,
                   gave_up_message, heartbeat_message, suspicion_message)
from detection import FaultDetector
import recovery


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
                 use_auction=True, reallocation=None):
        x, y, theta = pose
        # Whether one robot's unfinished work can become another's. Gates
        # all three routes -- claim expiry, a cheaper bid taking a claim,
        # and the give-up broadcast. See REALLOCATION_ENABLED.
        self.reallocation = (config.REALLOCATION_ENABLED
                             if reallocation is None else reallocation)
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

        # Sensor health, as the three parameters sensors.py already takes.
        # Healthy values here; faults.py makes them worse. The robot itself
        # never reads these to make a decision -- it cannot tell that its
        # own LiDAR has gone bad, which is precisely why detecting that
        # needs the other robots.
        self.sensor_noise_std = config.LIDAR_NOISE_STD_M
        self.sensor_range_scale = 1.0
        self.sensor_dropout_prob = 0.0

        # Faults that have been applied to this robot, as (step, name).
        # Written by the injector, read only by the report. Nothing in the
        # robot's own decision-making may consult it.
        self.faults = []

        # What this robot believes about its team-mates. Private, like
        # everything else here -- two robots routinely hold different
        # opinions because they heard different things.
        self.detector = FaultDetector(robot_id)
        self.last_heartbeat = -10 ** 9

        # --- what it has decided to DO about its team-mates (Step 5) ---
        # accuser_id -> {suspect: {fault: step}}. Conclusions other robots
        # have reached and told us about; corroboration is counted from
        # this plus our own detector.
        self.heard_suspicions = {}
        self.trust = {}               # peer_id -> weight on its map data
        self.quarantined = set()      # peers whose contributions we removed
        self.ignore_claims_from = set()   # peers whose work we have taken
        self.released = set()         # points freed for re-auction
        # Its lane: the points it was given at mission start, and any it
        # has since picked up from a robot the squad wrote off. `assigned`
        # never changes; `extra` only grows, and only when reallocation is
        # enabled.
        self.assigned = set()
        self.extra = set()
        # The whole division of the round, robot id -> its lane. Issued by
        # the operator along with the point list, exactly as the drawings
        # are, so every robot knows whose lane is whose. It is a plan, not
        # a live shared state -- nothing writes to it during the mission.
        self.lane_of = {}
        self.recovery_actions = []
        self.recovery_applied = set()  # (suspect, fault) already acted on
        self._announced = set()       # suspicions already broadcast

        # Rolling scan statistics, summed since the last heartbeat and
        # reported as means. These describe the robot's own sensor, and it
        # reports them honestly -- a degraded LiDAR does not lie about
        # itself, it is simply bad.
        self._scans = 0
        self._valid_sum = 0.0
        self._range_var_sum = 0.0

        # --- what it has heard ---------------------------------------
        self.inbox = []
        self.claims = {}      # point index -> {"by", "cost", "heard_at"}
        self.done = set()     # points it believes are finished
        # ...and WHO it believes finished each one. Needed because a
        # quarantine invalidates everything the quarantined robot reported,
        # and a bare set of indices cannot say which those are.
        self.done_by = {}
        self.given_up = set()  # points it tried and could not reach
        # ...and what each of them cost when it took them on, so the
        # finding can be passed to somebody who has not tried yet.
        self.given_up_cost = {}
        # Points OTHER robots have reported failing at, and what it cost
        # them. Kept apart from `given_up` because these are second-hand:
        # this robot may still be much better placed than the one that
        # failed, and choose_target is where that gets decided.
        self.gave_up_elsewhere = {}

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
            facility, r.x, r.y, r.theta, self.rng,
            noise_std=self.sensor_noise_std,
            range_scale=self.sensor_range_scale,
            dropout_prob=self.sensor_dropout_prob)
        r.pay_sensing(config.DT_S * config.INSPECTION_SCAN_EVERY_N_STEPS)
        self.grid.integrate_scan(r.bx, r.by, r.btheta,
                                 self.ranges, self.angles, self.valid,
                                 source_id=self.id)

        # Sensor health, accumulated for the next heartbeat. Two numbers:
        # how many beams came back at all, and how erratic the ranges were.
        # A dropping sensor moves the first; a noisy one moves the second.
        self._scans += 1
        self._valid_sum += float(self.valid.mean())
        self._range_var_sum += float(np.var(self.ranges))

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
                # A robot whose work we have taken over does not get to
                # claim anything else. Without this it keeps re-claiming
                # the points we just released and the reallocation undoes
                # itself on the next broadcast.
                if msg["from"] in self.ignore_claims_from:
                    continue
                held = self.claims.get(msg["point"])
                # With permanent claims the first one heard stands; a later
                # cheaper bid does not displace it, because displacing it
                # is how work moves off a failed robot.
                overtakes = (msg["cost"] < held["cost"]
                             if (held is not None and self.reallocation)
                             else False)
                if (held is None
                        or self._claim_expired(held, step)
                        or overtakes
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
                        and self.reallocation
                        and self.target is not None
                        and msg["point"] == self.target.index
                        and msg["from"] != self.id
                        and msg["cost"] < self.target_cost):
                    self.claims.pop(msg["point"], None)
                    self.claims[msg["point"]] = {"by": msg["from"],
                                                 "cost": msg["cost"],
                                                 "heard_at": step}
                    self._clear_target()

            elif kind == "gave_up":
                # Somebody tried this point and could not reach it. Record
                # what it cost them; choose_target decides whether that
                # finding applies to us, because only there do we know what
                # the point would cost US.
                prev = self.gave_up_elsewhere.get(msg["point"])
                if prev is None or msg["cost"] < prev["cost"]:
                    self.gave_up_elsewhere[msg["point"]] = {
                        "by": msg["from"], "cost": msg["cost"],
                        "heard_at": step}
                self.claims.pop(msg["point"], None)

            elif kind == "visited":
                if msg["from"] in self.quarantined:
                    continue    # we do not credit a quarantined robot
                self.done.add(msg["point"])
                self.done_by[msg["point"]] = msg["from"]
                self.claims.pop(msg["point"], None)
                # A success overrides an earlier failure: if somebody got
                # there, it is reachable after all.
                self.gave_up_elsewhere.pop(msg["point"], None)
                # Somebody else finished the point we were driving to.
                # Drop it and pick another rather than arriving at a job
                # already done.
                if self.target is not None and self.target.index == msg["point"]:
                    self._clear_target()

            elif kind == "heartbeat":
                self.detector.note_heartbeat(msg, step)

            elif kind == "suspicion":
                self.heard_suspicions.setdefault(msg["from"], {}) \
                    .setdefault(msg["suspect"], {}) \
                    .setdefault(msg["fault"], msg["step"])

            elif kind == "map":
                # Merge at whatever this robot is now worth to us. A
                # quarantined robot is worth nothing and is skipped
                # entirely -- merging at weight zero would still write an
                # empty contribution into the ledger and undo the rollback
                # bookkeeping.
                weight = self.trust.get(msg["from"], 1.0)
                if weight > 0.0:
                    self.grid.merge_from(msg["grid"], source_id=msg["from"],
                                         weight=weight)
                    merged += 1

                for accuser, suspects in msg["suspicions"].items():
                    held = self.heard_suspicions.setdefault(accuser, {})
                    for suspect, faults in suspects.items():
                        entry = held.setdefault(suspect, {})
                        for fault, when in faults.items():
                            entry.setdefault(fault, when)

                # Reconcile the round as well as the map. A point somebody
                # has inspected is simply done -- that is a fact, not a
                # judgement, so it is taken as given.
                #
                # UNLESS WE HAVE QUARANTINED WHOEVER DID IT. A robot we no
                # longer trust to have mapped correctly is a robot we no
                # longer trust to have inspected correctly either, and
                # without this filter a team-mate's next map exchange
                # simply hands the invalidated completions straight back
                # and undoes the re-inspection.
                for idx, who in msg["done_by"].items():
                    if who in self.quarantined:
                        continue
                    self.done.add(idx)
                    self.done_by.setdefault(idx, who)
                    self.gave_up_elsewhere.pop(idx, None)
                    self.claims.pop(idx, None)

                # A failure is second-hand and stays that way: record what
                # it cost them and let choose_target decide whether we are
                # better placed. Keep the cheapest report, since that is the
                # strongest evidence anyone has that the point is hopeless.
                for idx, cost in msg["gave_up"].items():
                    if idx in self.done:
                        continue
                    prev = self.gave_up_elsewhere.get(idx)
                    if prev is None or cost < prev["cost"]:
                        self.gave_up_elsewhere[idx] = {
                            "by": msg["from"], "cost": cost,
                            "heard_at": step}

                if self.target is not None and self.target.index in self.done:
                    self._clear_target()

        self.inbox.clear()
        return merged

    def _claim_expired(self, claim, step):
        """
        Has this claim lapsed?

        Never, when claims are deeds rather than leases. That is the naive
        condition: whoever claimed a point keeps it, alive or dead, and if
        it dies the point simply never gets visited.
        """
        if not self.reallocation:
            return False
        return (step - claim["heard_at"]) > config.CLAIM_TIMEOUT_STEPS

    # =================================================================
    # 2b. Judge the neighbours
    # =================================================================
    def run_detectors(self, peer_ids, points, step):
        """
        Run every detector this robot can, on the evidence it actually has.

        Everything here works from heartbeats that had to survive the radio
        to arrive, plus this robot's OWN provenance ledger. No detector
        reads another robot's map, another robot's true pose, or any
        ground truth -- if one did, every detection latency reported in
        Chapter 4 would be fiction.
        """
        if step % config.DETECTOR_EVERY_N_STEPS == 0:
            me = (self.robot.bx, self.robot.by)
            self.detector.check_comms_loss(me, peer_ids, step)

            # How much of the round this robot believes is still
            # outstanding, for the predictive battery check. Its own view,
            # which may be out of date -- that is the honest input, not a
            # global count.
            outstanding = max(0, len(points) - len(self.done))

            for peer_id in peer_ids:
                if peer_id == self.id:
                    continue
                self.detector.check_sensor_degradation(peer_id, step)
                self.detector.check_immobilised(peer_id, step)
                self.detector.check_battery(peer_id, step, outstanding,
                                            len(peer_ids))

        # Throttled separately: this one rebuilds and compares two
        # full-size beliefs per peer. See BYZANTINE_CHECK_EVERY_N_STEPS.
        if step % config.BYZANTINE_CHECK_EVERY_N_STEPS == 0:
            progress = len(self.done) / max(len(points), 1)
            self.detector.check_wrong_position(self.grid, peer_ids, step,
                                               progress)

    def act_on_suspicions(self, points, step, radio, squad, enabled=True):
        """
        Tell the others what we have concluded, then act on anything the
        squad now agrees about.

        Announcing happens even with recovery disabled, so that condition
        C2 still pays the comms energy of a squad that talks about its
        problems without doing anything about them. What C2 does not do is
        act.
        """
        self.announce_suspicions(step, radio, squad)
        if not enabled:
            return []
        return recovery.consider(self, points, step)

    # =================================================================
    # 3. Bid
    # =================================================================
    def _candidates(self, points):
        """
        Which points this robot is allowed to go for.

        Its own lane, always. Points handed to it because the squad wrote
        somebody off, if reallocation is enabled. And, once its own lane is
        finished and reallocation is enabled, whatever is still outstanding
        anywhere -- a robot with nothing left of its own goes and helps.

        THAT LAST CLAUSE IS THE WHOLE OF C2 vs C5. In the naive condition
        it is switched off: a robot finishes its lane and stops, and a
        point belonging to a robot that died is never visited by anybody.
        That is what a system with no reallocation actually does, and it is
        what the hardware demonstrator does when fault tolerance is off --
        the surviving robot finishes its own taped lane and parks.
        """
        open_now = [p for p in points
                    if p.index not in self.done
                    and p.index not in self.given_up]

        mine = [p for p in open_now
                if p.index in self.assigned or p.index in self.extra]
        if mine or not self.reallocation:
            return mine
        return open_now

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

        pending = self._candidates(points)
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
                        # A permanent claim is not open to a better offer.
                        # "Held until it visits it" has to mean exactly
                        # that, or a closer robot simply takes the work off
                        # a dying one and the naive condition quietly
                        # recovers after all.
                        and (not self.reallocation
                             or held["cost"] <= cost)):
                    deferred += 1
                    continue       # somebody else has it

                # ACCEPT SOMEBODY ELSE'S FAILURE unless we are much better
                # placed than they were. Repeating a team-mate's failed
                # attempt costs a full no-progress budget to learn what we
                # were already told, and on seed 2024 doing that three times
                # over made the squad slower than a single robot.
                #
                # Taken into our own `given_up` rather than merely skipped,
                # because accepting the finding is what lets the round end:
                # the mission is over when every robot considers the board
                # clear, and a point nobody will ever bid for again has to
                # count as clear.
                failed = self.gave_up_elsewhere.get(p.index)
                if (failed is not None
                        and cost > failed["cost"] * config.GIVE_UP_OVERRIDE_FRACTION):
                    self.given_up.add(p.index)
                    continue

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

    def telemetry(self, points):
        """
        What this robot can honestly say about itself.

        Everything here is measurable from inside the robot -- its believed
        pose, its ledgers, its odometers, a summary of its own scans.
        Nothing is a diagnosis: the robot is not claiming to be well or
        unwell, it is stating readings. Every judgement is made by whoever
        receives it.

        A robot with a displaced pose reports its displaced pose in perfect
        good faith, which is exactly why that fault needs a different
        detector from all the others.
        """
        r = self.robot
        scans = max(self._scans, 1)
        assigned = len(self.done) + (1 if self.target is not None else 0)
        return {
            "bx": r.bx, "by": r.by,
            # Where it is going, so a peer can predict where it will be and
            # tell "out of range" apart from "failed".
            "target_xy": self.target.xy if self.target is not None else None,
            "battery_j": r.battery_j,
            "energy_j": r.total_energy_j,
            "distance_m": r.distance_travelled_m,
            "commanded_m": r.commanded_distance_m,
            "rotation_rad": r.rotation_travelled_rad,
            "commanded_rad": r.commanded_rotation_rad,
            "valid_ratio": self._valid_sum / scans,
            "range_var": self._range_var_sum / scans,
            "points_done": len(self.done),
            "points_assigned": assigned,
            "points_total": len(points),
        }

    def _suspicion_payload(self):
        """
        Everything this robot believes about who has failed -- its own
        conclusions and the ones it has been told. Passed on so that a
        corroborating pair who never meet directly can still both learn
        of each other through a third robot.
        """
        payload = {self.id: {s: dict(f) for s, f
                             in self.detector.accusations.items() if f}}
        for accuser, suspects in self.heard_suspicions.items():
            payload.setdefault(accuser, {})
            for suspect, faults in suspects.items():
                payload[accuser].setdefault(suspect, {}).update(faults)
        return payload

    def announce_suspicions(self, step, radio, squad):
        """Say out loud anything newly concluded, once each."""
        if not self.robot.alive:
            return
        for suspect, faults in self.detector.accusations.items():
            for fault in faults:
                if (suspect, fault) in self._announced:
                    continue
                self._announced.add((suspect, fault))
                radio.broadcast(self,
                                suspicion_message(self.id, suspect, fault,
                                                  step),
                                squad, config.COMMS_SUSPICION_PACKET_KB)

    def send_heartbeat(self, step, radio, squad, points):
        """Describe ourselves to anyone in range, on a fixed interval."""
        if not self.robot.alive:
            return
        if step - self.last_heartbeat < config.HEARTBEAT_EVERY_N_STEPS:
            return
        radio.broadcast(self,
                        heartbeat_message(self.id, step,
                                          self.telemetry(points)),
                        squad, config.COMMS_HEARTBEAT_PACKET_KB)
        self.last_heartbeat = step
        # Reset the scan window so the next heartbeat describes the next
        # stretch of driving rather than the whole mission to date.
        self._scans = 0
        self._valid_sum = 0.0
        self._range_var_sum = 0.0

    def offer_map(self, step, radio, squad):
        """Periodically broadcast what this robot has seen for itself."""
        if not self.robot.alive:
            return
        if step - self.last_map_broadcast < config.COMMS_EXCHANGE_EVERY_N_STEPS:
            return
        # given_up costs travel only when reallocation is enabled -- they
        # are how a team-mate learns it need not retry somebody else's
        # abandoned point, which is reallocation by another route.
        radio.broadcast(self, map_message(self.id, self.grid, step,
                                          dict(self.done_by),
                                          dict(self.given_up_cost)
                                          if self.reallocation else {},
                                          self._suspicion_payload()),
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
            self._mark_given_up(step, radio, squad)

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
        # Was it actually inspected? Scored against the TRUE pose, which is
        # the simulator's business and never the robot's -- a displaced
        # robot has no way of knowing it is standing in the wrong place.
        if p.visit_error_m <= config.INSPECTION_VERIFY_RADIUS_M:
            p.truly_visited = True
        self.done.add(p.index)
        self.done_by[p.index] = self.id
        self.claims.pop(p.index, None)
        radio.broadcast(self, visited_message(self.id, p.index, step),
                        squad, config.COMMS_CLAIM_PACKET_KB)
        self._clear_target()

    def _mark_given_up(self, step, radio, squad):
        """
        Abandon the current point and tell everyone, so that nobody else
        has to spend a no-progress budget learning the same thing.

        The cost broadcast is what this point cost when it was taken, which
        is the receiver's yardstick for deciding whether it is appreciably
        better placed than we were.
        """
        p = self.target
        cost = self.target_cost
        self.given_up.add(p.index)

        if not self.reallocation:
            # THE NAIVE CASE: it keeps the claim and the point is lost.
            #
            # Giving up and announcing it is not base mission behaviour, it
            # is self-reported failure detection followed by task
            # reallocation -- exactly the capability under test. Leaving it
            # switched on in the naive condition is what produced the
            # immobilised inversion, where C2 beat C3: an immobilised robot
            # gave up on every point it was holding and the healthy robots
            # quietly collected them, with fault tolerance turned off.
            #
            # So here the robot stops trying, says nothing, and keeps the
            # claim that stops anybody else trying either.
            self._clear_target()
            return

        self.given_up_cost[p.index] = cost
        self.claims.pop(p.index, None)
        radio.broadcast(self, gave_up_message(self.id, p.index, cost, step),
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
