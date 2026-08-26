"""
detection.py
============
Noticing that a team-mate has broken. Step 4.

WHO DOES THE NOTICING
---------------------
Peers, always. Every detector in this file runs on evidence about ANOTHER
robot, never about the robot running it. That is not a stylistic choice:
a robot cannot be trusted to detect its own faults, and the wrong-position
fault is the proof. That robot's sensor is perfect, its readings are
clean, its map is internally consistent and its energy accounting is
normal. Nothing it can check about itself is wrong. It is simply drawing a
correct map in the wrong place, and only somebody else can see that.

Each robot therefore keeps its own opinions about its peers, formed only
from messages that actually reached it. Two robots can and do disagree
about who is broken -- one of them may have been out of range while the
evidence arrived. That is a property of a decentralised squad, not a bug.

DETECTION ONLY
--------------
Nothing here quarantines anybody, rolls anything back or reallocates any
work. An accusation is recorded with the step it was made and that is all
that happens. Keeping detection separate from recovery is what makes
detection latency a measured number instead of an assumption, and Step 5
is where accusations start having consequences.

THE TRAP IN THE COMMS DETECTOR
------------------------------
A healthy robot behind a storage tank goes quiet in exactly the way a
failed one does. If silence alone were evidence, every robot that drove
out of radio range would be accused, and false-positive rate is one of the
numbers this project reports.

So silence is only evidence when the robot SHOULD have been heard. Each
robot predicts where its peers are and only starts a timeout against a
peer it believes is close enough to hear. The prediction decays -- a peer
that has been quiet long enough could be anywhere -- and past that point
the gate opens and a long conservative timeout takes over.
"""

import numpy as np

import config


class FaultDetector:
    """
    One robot's opinions about its team-mates.

    Owned by a SquadMember, fed by heartbeats, and consulted by nobody but
    the report. It holds no map and no robot: everything it knows arrived
    as a message.
    """

    def __init__(self, owner_id):
        self.owner_id = owner_id

        # peer_id -> the last heartbeat we heard, plus when we heard it.
        # `previous` keeps the one before, because three of the five
        # detectors work on rates and a rate needs two samples.
        self.peers = {}

        # peer_id -> {fault_name: step first accused}. Latching: an
        # accusation is a finding and findings do not un-happen. Step 5
        # may act on it; nothing here withdraws it.
        self.accusations = {}
        self.reasons = {}      # (peer_id, fault) -> what the evidence was

        # Bookkeeping so the report can explain a miss: how many times the
        # silence gate suppressed a timeout that would otherwise have run.
        self.silence_excused = 0

        # Consecutive bad reports per peer. Two detectors demand a run of
        # them rather than a single sample, because one awkward corner is
        # not a broken robot.
        self._sensor_streak = {}
        self._immobile_streak = {}
        self._byzantine_streak = {}

        # Which heartbeat each streak detector has already counted, so a
        # detector that runs more often than heartbeats arrive does not
        # count the same report twice. Without this, "three reports
        # running" would silently become one and a half.
        self._seen_report = {}

    # =================================================================
    # Hearing from a peer
    # =================================================================
    def note_heartbeat(self, msg, step):
        """Record a peer's self-report, keeping the previous one for rates."""
        peer_id = msg["from"]
        record = self.peers.get(peer_id)
        previous = record["latest"] if record else None
        self.peers[peer_id] = {"latest": dict(msg), "previous": previous,
                               "heard_at": step}

    def accuse(self, peer_id, fault_name, step, reason=""):
        """
        Record that this robot believes `peer_id` has failed in this way.

        First accusation wins: the step recorded is when it was first
        noticed, which is what detection latency measures. `reason` records
        which piece of evidence did it, so a report can say why -- and, on
        a healthy squad, explain what a false positive tripped over.
        """
        held = self.accusations.setdefault(peer_id, {})
        if fault_name not in held:
            held[fault_name] = step
            self.reasons[(peer_id, fault_name)] = reason
            return True
        return False

    def accused(self, peer_id, fault_name=None):
        held = self.accusations.get(peer_id, {})
        return bool(held) if fault_name is None else (fault_name in held)

    def _fresh_report(self, peer_id, latest, tag):
        """
        True the first time a given heartbeat is judged by a given
        detector. The streak detectors count consecutive REPORTS, and they
        run more often than reports arrive, so without this they would
        count the same one repeatedly and convict on a third of the
        evidence they claim to require.
        """
        key = (peer_id, tag)
        if self._seen_report.get(key) == latest["step"]:
            return False
        self._seen_report[key] = latest["step"]
        return True

    # =================================================================
    # The silence gate  (Stage A)
    # =================================================================
    def predict_peer_xy(self, peer_id, step):
        """
        Where do we think this peer is now?

        Dead reckoning on somebody else's behalf: take the last pose it
        broadcast, and walk it toward the inspection point it said it was
        driving to, at the speed it has recently been making. Both facts
        come out of its own heartbeats -- the declared target and the
        distance odometer -- so this is a prediction built entirely from
        what the peer volunteered about itself.

        WHY NOT JUST USE THE LAST KNOWN POSITION. Because it is useless for
        this purpose, and finding that out cost a full run. A peer's last
        known position is ALWAYS within radio range, by construction: being
        in range is the only reason we heard it. Any test of the form
        "last position, minus how far it could have moved, is still beyond
        the radio" can therefore never be true, and the gate built on it
        excused nothing at all -- every healthy robot accused every other
        of a comms failure. A robot that has driven off toward a point 60 m
        away is predicted to be 60 m away, which is the whole point.
        """
        record = self.peers[peer_id]
        last = record["latest"]
        px, py = last["bx"], last["by"]

        goal = last.get("target_xy")
        if goal is None:
            return px, py             # it was not going anywhere

        # Its recent ground speed, from the odometer in two heartbeats. If
        # we have only ever heard one, fall back to a fraction of the top
        # speed rather than assuming it stood still -- a robot that has
        # announced a destination is usually driving to it, and "assume it
        # stayed put" is precisely the assumption that keeps predicting a
        # departed robot is still next to us.
        previous = record["previous"]
        speed = config.MAX_LINEAR_SPEED_MPS * config.PEER_PREDICTION_SPEED_FRACTION
        if previous is not None:
            dt_s = (last["step"] - previous["step"]) * config.DT_S
            if dt_s > 0:
                speed = max(0.0, (last["distance_m"]
                                  - previous["distance_m"]) / dt_s)

        dx, dy = goal[0] - px, goal[1] - py
        remaining = float(np.hypot(dx, dy))
        if remaining < 1e-6 or speed <= 0.0:
            return px, py

        elapsed_s = (step - record["heard_at"]) * config.DT_S
        travelled = min(speed * elapsed_s, remaining)
        return px + dx / remaining * travelled, py + dy / remaining * travelled

    def short_timeout_applies(self, my_xy, peer_id, step):
        """
        Can we positively say this peer should have been heard by now?

        Only then does silence mean anything. Three ways the answer is no:

          we have never heard from it, so there is no estimate at all;
          our estimate has gone stale, past PEER_POSITION_DECAY_S, and a
            peer that has been quiet that long could be anywhere;
          our estimate puts it outside COMMS_RANGE_M, where silence is
            geometry rather than failure.

        THE STALE CASE FALLS BACK, IT DOES NOT ACCUSE. An earlier version
        had this backwards -- it treated a stale estimate as grounds to
        start the short timeout, which is exactly the wrong way round: the
        less we know about where a robot is, the *weaker* the case that its
        silence is its own fault. Staleness hands the job to the hard
        timeout instead, which is deliberately long.

        Both poses are BELIEVED poses -- ours and the one the peer sent.
        Neither robot knows a true position. A peer whose believed pose has
        been displaced will report the displacement in good faith and this
        gate will believe it; that is a real limitation, and the
        wrong-position detector is what covers it.
        """
        record = self.peers.get(peer_id)
        if record is None:
            return False

        elapsed_s = (step - record["heard_at"]) * config.DT_S
        if elapsed_s > config.PEER_POSITION_DECAY_S:
            return False

        px, py = self.predict_peer_xy(peer_id, step)
        predicted_range = float(np.hypot(my_xy[0] - px, my_xy[1] - py))

        # Comfortably inside, not merely inside -- and "comfortably" gets
        # stricter the longer the peer has been quiet, because that is how
        # confident the prediction is. Judging at the boundary with a fixed
        # margin turns ordinary dead-reckoning error straight into
        # accusations against healthy robots.
        margin = (config.COMMS_PREDICTION_MARGIN_M
                  + config.PEER_PREDICTION_DRIFT_MPS * elapsed_s)
        return predicted_range <= (config.COMMS_RANGE_M - margin)

    # =================================================================
    # Detector 3 of 5: comms loss
    # =================================================================
    def check_comms_loss(self, my_xy, peer_ids, step):
        """
        Accuse a peer that has gone quiet when it had no business doing so.

        Two timeouts, and the difference between them is the whole point:

          the SHORT one only runs against a peer we believe is in range.
          Silence from somewhere it could not be heard from is not silence,
          it is geometry.

          the HARD one runs against anybody, however far away we last saw
          them. A robot that has said nothing for three minutes has a
          problem regardless of where we think it is, and without this a
          robot that failed just as it drove out of range would never be
          accused at all.
        """
        for peer_id in peer_ids:
            if peer_id == self.owner_id:
                continue
            record = self.peers.get(peer_id)
            if record is None:
                continue              # never in contact; nothing to judge

            quiet_s = (step - record["heard_at"]) * config.DT_S

            if quiet_s > config.COMMS_SILENCE_HARD_TIMEOUT_S:
                self.accuse(peer_id, "comms_loss", step,
                            f"hard timeout, silent {quiet_s:.0f} s")
                continue

            if quiet_s > config.COMMS_SILENCE_TIMEOUT_S:
                if self.short_timeout_applies(my_xy, peer_id, step):
                    px, py = self.predict_peer_xy(peer_id, step)
                    rng = float(np.hypot(my_xy[0] - px, my_xy[1] - py))
                    self.accuse(peer_id, "comms_loss", step,
                                f"silent {quiet_s:.0f} s while predicted "
                                f"{rng:.0f} m away, inside range")
                else:
                    self.silence_excused += 1

    # =================================================================
    # Detector 1 of 5: sensor degradation
    # =================================================================
    def check_sensor_degradation(self, peer_id, step):
        """
        A LiDAR that has stopped seeing properly.

        Two channels, and BOTH must be bad at once. Neither works alone,
        which the measured distributions make obvious:

          valid-return ratio  healthy runs 0.35 to 0.96, median 0.62;
                              degraded 0.04 to 0.96, median 0.27. The
                              spread overlaps almost completely, because a
                              healthy robot in open ground gets few returns
                              too -- there is simply nothing out there to
                              hit.
          range variance      healthy 1.5 to 10.2, median 6.8; degraded
                              0.1 to 10.2, median 0.8. Note the direction:
                              a degraded sensor's variance goes DOWN, not
                              up, because losing range squashes every
                              reading toward the same short cap.

        Few returns AND uniform ranges together is the signature: it means
        the sensor is both blind and short-sighted, which open ground is
        not. A robot in a corridor sees plenty and varies plenty; a robot
        in the open sees little but what it does see varies.

        Then it must persist. One bad window is a robot in an awkward
        corner; SENSOR_MIN_HEARTBEATS in a row is a broken sensor.
        """
        record = self.peers.get(peer_id)
        if record is None:
            return
        latest = record["latest"]
        if not self._fresh_report(peer_id, latest, "sensor"):
            return

        bad = (latest["valid_ratio"] < config.SENSOR_VALID_RATIO_MIN
               and latest["range_var"] < config.SENSOR_RANGE_VARIANCE_MIN)

        if not bad:
            self._sensor_streak[peer_id] = 0
            return

        streak = self._sensor_streak.get(peer_id, 0) + 1
        self._sensor_streak[peer_id] = streak
        if streak >= config.SENSOR_MIN_HEARTBEATS:
            self.accuse(peer_id, "sensor_degradation", step,
                        f"{streak} consecutive reports at "
                        f"valid={latest['valid_ratio']:.2f}, "
                        f"variance={latest['range_var']:.2f}")

    # =================================================================
    # Detector 2 of 5: wrong position -- the one that needs three robots
    # =================================================================
    def check_wrong_position(self, grid, peer_ids, step):
        """
        A robot drawing a correct map in the wrong place.

        THE VOTE. A disagreement between two robots says one of them is
        wrong and nothing about which. The third robot breaks the tie: if I
        disagree sharply with B while agreeing with C, and C is not
        disagreeing with me, then B is the odd one out. That is the entire
        reason this project uses three robots rather than two, and it is
        worth saying in those words at the defence.

        WHAT IS COMPARED. Each robot's OWN observations, pulled out of this
        robot's provenance ledger and re-based on the prior -- never the
        merged map. See OccupancyGrid.contribution_conflict for why the
        obvious version of this detector reports nothing at all.

        Note what this cannot do: it cannot tell whether the odd one out is
        me. A robot with a displaced pose runs this same check, disagrees
        with both of its peers, and concludes nothing -- because the rule
        below requires somebody to agree with me before I accuse anyone.
        That is correct. A robot cannot be its own judge, and the peers
        will convict it.
        """
        ids = sorted({self.owner_id} | set(peer_ids))
        if len(ids) < 3:
            return          # no third opinion available, so no vote

        # All three pairwise rates, entirely from this robot's own ledger.
        # Comparing the two PEERS against each other is the part that makes
        # the test comparative: it measures how much the rest of the squad
        # agrees among itself, which is this run's drift level.
        rates = {}
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                conflict, overlap = grid.contribution_conflict(ids[i], ids[j])
                if overlap < config.BYZANTINE_MIN_OVERLAP_CELLS:
                    return  # not enough shared ground to judge anybody yet
                rates[frozenset((ids[i], ids[j]))] = conflict / overlap

        # Score every candidate, then act on AT MOST ONE of them.
        #
        # ONLY ONE ROBOT CAN BE THE ODD ONE OUT. That is what "odd one out"
        # means in a squad of three, and enforcing it matters: a genuinely
        # displaced robot inflates every rate it appears in, including the
        # ones used to judge the healthy robot it happens to share most
        # ground with. Scoring each suspect independently, both cleared the
        # threshold and both got quarantined -- on seed 42 the squad
        # correctly quarantined the faulty robot and then quarantined a
        # healthy one alongside it. Taking only the strongest case leaves
        # the minority robot accused and the majority intact.
        scored = []
        for suspect in ids:
            if suspect == self.owner_id:
                continue    # a robot cannot be its own judge
            others = [k for k in ids if k != suspect]
            mean_rate = float(np.mean([rates[frozenset((suspect, o))]
                                       for o in others]))
            baseline = rates[frozenset((others[0], others[1]))]

            # Three ways this comparison is not yet worth making: the
            # suspect barely disagrees with anyone, the pair being used as
            # the yardstick has no measurable disagreement of its own to
            # act as one, or the ratio is simply not large enough.
            ratio = mean_rate / baseline if baseline > 0 else float("inf")
            if (mean_rate < config.BYZANTINE_MIN_MEAN_RATE
                    or baseline < config.BYZANTINE_MIN_BASELINE_RATE
                    or ratio < config.BYZANTINE_RATIO):
                continue
            scored.append((ratio, suspect, mean_rate, baseline))

        for suspect in ids:
            if suspect not in [s for _, s, _, _ in scored]:
                self._byzantine_streak[suspect] = 0

        if not scored:
            return

        ratio, suspect, mean_rate, baseline = max(scored)
        streak = self._byzantine_streak.get(suspect, 0) + 1
        self._byzantine_streak[suspect] = streak
        if streak < config.BYZANTINE_MIN_CHECKS:
            return

        self.accuse(suspect, "wrong_position", step,
                    f"disagrees with the rest of us at {mean_rate*100:.2f} "
                    f"% while they agree with each other at "
                    f"{baseline*100:.2f} % ({ratio:.1f}x)")

    # =================================================================
    # Detector 4 of 5: immobilised
    # =================================================================
    def check_immobilised(self, peer_id, step):
        """
        Asking for motion and delivering none.

        Both odometers ride in the heartbeat, so this is a subtraction: how
        much movement was commanded between two reports, and how much
        actually happened.

        ROTATION IS COUNTED, NOT JUST DISTANCE, and leaving it out very
        nearly hid this fault. An immobilised robot whose next path node is
        behind it commands a pure turn and no forward motion at all -- the
        controller will not drive until it is pointing the right way. It
        cannot turn, so its heading never updates, so it re-issues the same
        turn every step forever while commanding exactly zero distance.
        Measured on seed 42: 3,285 consecutive steps of that.

        Radians become metres at the wheel by multiplying by the robot
        radius, which is what a differential drive actually does: turning
        in place by an angle drives each wheel that far round.

        Requires several consecutive reports, because a robot grinding
        against a wall also commands motion it does not achieve -- for a
        few seconds, until the escape behaviour reverses it out.
        """
        record = self.peers.get(peer_id)
        if record is None or record["previous"] is None:
            return
        latest, previous = record["latest"], record["previous"]
        if not self._fresh_report(peer_id, latest, "immobile"):
            return

        commanded = ((latest["commanded_m"] - previous["commanded_m"])
                     + config.ROBOT_RADIUS_M
                     * (latest["commanded_rad"] - previous["commanded_rad"]))
        achieved = ((latest["distance_m"] - previous["distance_m"])
                    + config.ROBOT_RADIUS_M
                    * (latest["rotation_rad"] - previous["rotation_rad"]))

        if commanded < config.IMMOBILE_COMMANDED_M:
            self._immobile_streak[peer_id] = 0
            return          # it is not asking to go anywhere

        if achieved > config.IMMOBILE_ACHIEVED_FRACTION * commanded:
            self._immobile_streak[peer_id] = 0
            return          # it is asking, and it is moving

        streak = self._immobile_streak.get(peer_id, 0) + 1
        self._immobile_streak[peer_id] = streak
        if streak >= config.IMMOBILE_MIN_HEARTBEATS:
            self.accuse(peer_id, "immobilised", step,
                        f"commanded {commanded:.2f} m of wheel travel, "
                        f"achieved {achieved:.2f} m, {streak} reports "
                        f"running")

    # =================================================================
    # Detector 5 of 5: battery drain -- the predictive one
    # =================================================================
    def check_battery(self, peer_id, step, points_outstanding, squad_size):
        """
        Charge disappearing faster than the work explains, and -- the part
        that matters -- a robot that will not finish what it has been given.

        Two signals:

        RATE. A healthy robot's battery falls by exactly the joules it
        spends, so charge_used / energy_spent is 1.00 to the last decimal.
        The fault multiplies that. Anything meaningfully above 1 is a cell
        problem, not a work problem, and it is unambiguous: the measured
        healthy value is 1.00 in every window of every run.

        PROJECTION, and this is the one worth having. Every other detector
        in this file reports a robot that has ALREADY failed. This one
        reports a robot that is going to: take what its work has cost so
        far per inspection point, multiply by the share of the remaining
        points it can expect to be given, and compare against the charge it
        has left. Flagging that while the robot is still driving is the
        difference between handing its work to somebody else and losing it
        when it dies mid-aisle.

        The estimate is deliberately crude -- past cost per point, an even
        split of what is left. It does not need to be accurate, only early.
        """
        record = self.peers.get(peer_id)
        if record is None or record["previous"] is None:
            return
        latest, previous = record["latest"], record["previous"]

        spent = latest["energy_j"] - previous["energy_j"]
        drained = previous["battery_j"] - latest["battery_j"]
        if spent > 1e-6:
            ratio = drained / spent
            if ratio > config.BATTERY_DRAIN_RATIO_MAX:
                self.accuse(peer_id, "battery_drain", step,
                            f"battery falling {ratio:.1f}x faster than the "
                            f"work it is doing")
                return

        done = latest["points_done"]
        if done < config.BATTERY_MIN_POINTS_DONE:
            return
        share = max(1.0, points_outstanding / max(squad_size, 1))
        needed = (latest["energy_j"] / done) * share
        if needed * config.BATTERY_PROJECTION_MARGIN > latest["battery_j"]:
            self.accuse(peer_id, "battery_drain", step,
                        f"projected {needed:.0f} J needed for its share of "
                        f"the round, {latest['battery_j']:.0f} J left "
                        f"(predictive)")

    # =================================================================
    def summary(self):
        """{peer_id: {fault: step}} -- what this robot currently believes."""
        return {pid: dict(faults) for pid, faults in self.accusations.items()
                if faults}

    def __repr__(self):
        return (f"<FaultDetector owner={self.owner_id} "
                f"accusations={self.summary()}>")


# ---------------------------------------------------------------------
def squad_accusations(squad):
    """
    Collate what the whole squad believes, without building a shared
    object to hold it.

    Returns {accused_id: {fault: (first_step, [accusers])}}. This is a
    REPORTING function -- it reads each robot's private opinions from the
    outside, which no robot can do. The robots never pool their views.
    """
    out = {}
    for member in squad:
        for peer_id, faults in member.detector.accusations.items():
            for fault, step in faults.items():
                entry = out.setdefault(peer_id, {}).setdefault(
                    fault, [None, []])
                if entry[0] is None or step < entry[0]:
                    entry[0] = step
                entry[1].append(member.id)
    return {pid: {f: (s, sorted(who)) for f, (s, who) in faults.items()}
            for pid, faults in out.items()}
