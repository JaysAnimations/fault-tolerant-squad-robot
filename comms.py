"""
comms.py
========
The radio link between robots. Step 2.

WHY THIS FILE EXISTS AT ALL
---------------------------
It would be far easier to let the three robots read each other's
variables directly. That is exactly what we must not do. The claim in
Chapter 1 is a DECENTRALISED squad: no master, no shared map, and every
piece of information a robot holds about another robot arrived as a
message that had to survive distance and interference to get there.

This module is the only route by which anything passes between robots.
If you can delete it and the squad still coordinates, the decentralisation
claim is false.

WHAT LIMITS DELIVERY
--------------------
  range  -- COMMS_RANGE_M, measured between the robots' TRUE positions.
            Radio propagation is physics, not belief, so this is the
            simulator's business and not the robot's. A robot never gets
            to look at this; it only ever notices that a message did or
            did not arrive.
  loss   -- an in-range message still fails with COMMS_PACKET_LOSS_PROB.
            Real plants are full of steel and variable-speed drives.
  power  -- the sender is charged E_COMMS_J_PER_KB for what it sends.

WHY THE SENDER PAYS ONCE PER BROADCAST
--------------------------------------
A broadcast costs the transmitter the same whether nobody hears it or
three robots do -- the energy goes into the antenna, not into each
listener. So the charge is per broadcast, not per delivery. Charging per
recipient would make a robot's energy depend on how many friends happened
to be nearby, which is not how a radio works.
"""

import numpy as np

import config


class Radio:
    """
    Range-limited, lossy broadcast between robots.

    Owns no map and no robot. You hand it the squad each time you want a
    message delivered, which keeps it unaware of anything but geometry.
    """

    def __init__(self, rng, range_m=None, loss_prob=None):
        # Read from config at construction, not as a default argument.
        # A default is evaluated once when this module is imported, so a
        # sweep that sets config.COMMS_RANGE_M between runs would change
        # what the detectors believe about range while leaving the radio
        # itself on whatever value happened to be loaded first -- robots
        # judging each other against a range their own radio does not have.
        self.rng = rng
        self.range_m = (config.COMMS_RANGE_M if range_m is None else range_m)
        self.loss_prob = (config.COMMS_PACKET_LOSS_PROB
                          if loss_prob is None else loss_prob)

        # Counters, for the report. Not used by any robot.
        self.sent = 0
        self.delivered = 0
        self.lost_to_range = 0
        self.lost_to_noise = 0

    # -----------------------------------------------------------------
    def in_range(self, a, b):
        """True separation between two robots, against the radio range."""
        d = np.hypot(a.robot.x - b.robot.x, a.robot.y - b.robot.y)
        return bool(d <= self.range_m)

    def broadcast(self, sender, message, squad, size_kb):
        """
        Send one message to every other member of the squad.

        Delivery is attempted for each of them independently: two robots
        at different distances are not guaranteed the same outcome, and a
        message that reaches one may be lost to the other. That asymmetry
        is what makes the squad's knowledge genuinely partial.

        Returns the number of robots that received it.
        """
        # A robot that is dead, or whose radio has failed, transmits
        # nothing -- and pays nothing.
        if not sender.robot.alive or not sender.robot.connected:
            return 0

        sender.robot.pay_comms(size_kb)
        self.sent += 1

        received = 0
        for other in squad:
            if other is sender:
                continue
            if not other.robot.alive or not other.robot.connected:
                continue
            if not self.in_range(sender, other):
                self.lost_to_range += 1
                continue
            if self.rng.random() < self.loss_prob:
                self.lost_to_noise += 1
                continue
            other.inbox.append(message)
            self.delivered += 1
            received += 1
        return received

    # -----------------------------------------------------------------
    def summary(self):
        return {"sent": self.sent, "delivered": self.delivered,
                "lost_to_range": self.lost_to_range,
                "lost_to_noise": self.lost_to_noise}


# ---------------------------------------------------------------------
# Message constructors. Plain dicts on purpose -- a message is data, and
# giving it a class would invite putting behaviour on it.
# ---------------------------------------------------------------------
def claim_message(sender_id, point_index, cost, step):
    """"I intend to inspect point N, and it will cost me C to get there."""
    return {"kind": "claim", "from": sender_id, "point": point_index,
            "cost": cost, "step": step}


def visited_message(sender_id, point_index, step):
    """"Point N is done, stop bidding for it."""
    return {"kind": "visited", "from": sender_id, "point": point_index,
            "step": step}


def suspicion_message(sender_id, suspect_id, fault_name, step):
    """
    "I believe robot N has failed, in this way."

    A CONCLUSION, NOT EVIDENCE. The sender is not asking anyone to check
    its working; it reached this on its own and is saying so. A receiver
    that has reached the same conclusion independently now has two
    measurements agreeing, which is what Step 5 requires before acting.

    This message is why the squad needs three robots and a radio that
    reaches. Two robots cannot corroborate anything if they are never in
    contact -- which is what the range sweep in sweep_comms.py exists to
    settle.
    """
    return {"kind": "suspicion", "from": sender_id, "suspect": suspect_id,
            "fault": fault_name, "step": step}


def heartbeat_message(sender_id, step, telemetry):
    """
    "I am here, this is how I am, and this is what I have been doing."

    Broadcast on a fixed interval. It carries the sender's BELIEVED pose,
    its energy and battery, its odometers and a summary of its recent
    scans -- everything a peer needs to run the Step 4 detectors, and
    nothing a robot could not honestly report about itself.

    WHY THE POSE IS IN HERE, AND WHY IT IS THE BELIEVED ONE. A peer needs
    it twice over: to decide whether silence from this robot is expected
    (see FaultDetector.silence_is_expected) and, eventually, to notice that
    it disagrees with everybody about where things are. The believed pose
    is what a robot can actually transmit -- no robot knows its true one --
    and a robot with a displaced pose will report its displacement in
    perfect good faith. That is what makes that fault hard.

    Note what a heartbeat is NOT: it is not an accusation, a diagnosis or a
    request. It is a robot describing itself. Every judgement is made by
    the receiver, which is what keeps detection decentralised.
    """
    msg = {"kind": "heartbeat", "from": sender_id, "step": step}
    msg.update(telemetry)
    return msg


def gave_up_message(sender_id, point_index, cost, step):
    """
    "I tried point N and could not get to it. It cost me C when I took it."

    A NEGATIVE RESULT IS STILL A RESULT, and this is the message that says
    so. Without it every robot has to discover the same closed aisle for
    itself, paying its own 500-step no-progress budget to learn what a
    team-mate already knows. On seed 2024 that made three robots slower
    than one.

    The cost travels with it so the receiver can judge whether the finding
    applies to it. A robot approaching from the far side of the site may
    well succeed where the sender failed; a robot no closer than the sender
    was is about to repeat somebody else's mistake.

    Step 5 needs this message anyway -- reallocating a failed robot's work
    is the same announcement with a different reason attached.
    """
    return {"kind": "gave_up", "from": sender_id, "point": point_index,
            "cost": cost, "step": step}


def map_message(sender_id, grid, step, done_by, gave_up, suspicions):
    """
    "Here is what I have seen, and here is where the round stands."

    The payload is the sender's live grid rather than a copy. That is safe
    only because every inbox is drained in the same step it is filled, so
    nothing can mutate in between -- and it avoids copying a megabyte per
    meeting. If message handling is ever deferred to a later step, this
    must become a snapshot.

    WHY MISSION STATE TRAVELS WITH THE MAP. The one-off "visited" and
    "gave up" announcements only reach whoever happens to be within radio
    range at that instant, and on an 80 x 55 m site with a 25 m radio that
    is usually nobody. Robots were therefore driving across the facility to
    inspect points a team-mate had finished an hour earlier, and nobody
    ever heard that one point was unreachable: on seed 2024, 60 % of the
    mission was spent that way.

    So when two robots do meet, they reconcile. `done_by` maps a point
    number to the robot that reported inspecting it, and `gave_up` maps a
    point number to what it cost the robot that failed at it -- a few
    hundred bytes beside a map update, which is why the modelled packet
    size does not change.

    `done_by` carries WHO rather than just WHICH, because a quarantine has
    to be able to take back everything the quarantined robot reported. A
    bare set of point numbers cannot say whose completions to withdraw.

    `suspicions` rides along for the same reason `done` does: a one-off
    announcement only reaches whoever was listening at that instant, and
    corroboration is exactly the thing that must not be lost to a missed
    packet.
    """
    return {"kind": "map", "from": sender_id, "grid": grid, "step": step,
            "done_by": done_by, "gave_up": gave_up,
            "suspicions": suspicions}
