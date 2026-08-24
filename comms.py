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

    def __init__(self, rng, range_m=config.COMMS_RANGE_M,
                 loss_prob=config.COMMS_PACKET_LOSS_PROB):
        self.rng = rng
        self.range_m = range_m
        self.loss_prob = loss_prob

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


def map_message(sender_id, grid, step):
    """
    "Here is what I have seen."

    The payload is the sender's live grid rather than a copy. That is safe
    only because every inbox is drained in the same step it is filled, so
    nothing can mutate in between -- and it avoids copying a megabyte per
    meeting. If message handling is ever deferred to a later step, this
    must become a snapshot.
    """
    return {"kind": "map", "from": sender_id, "grid": grid, "step": step}
