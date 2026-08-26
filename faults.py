"""
faults.py
=========
Breaking a robot on purpose. Step 3.

WHAT THIS FILE IS AND IS NOT
----------------------------
It is the injector: at a scheduled step it reaches into one robot and
damages it. That is all.

It does NOT notice anything, score anything, or tell anybody. No robot is
informed that it has been broken, and no other robot is informed either.
Detection is Step 4 and it has to work from evidence -- a map that
disagrees with everyone else's, a robot that stopped talking, a drain rate
that outruns the work done. If this module so much as set a flag saying
"robot 1 is faulty", every detection latency reported in Chapter 4 would
be a lie.

So the only thing that happens here is damage.

HOW THE FIVE FAULTS ARE IMPLEMENTED
-----------------------------------
Almost entirely by setting values that already existed. sensors.py has
accepted `noise_std`, `range_scale` and `dropout_prob` since Step 0, and
robot.py has carried `mobile`, `connected` and `battery_drain_multiplier`
for just as long. This is wiring, not new machinery, and that was the plan.

    sensor_degradation  the three sensor parameters get worse
    wrong_position      the BELIEVED pose jumps; readings are untouched
    comms_loss          robot.connected = False
    immobilised         robot.mobile = False
    battery_drain       robot.battery_drain_multiplier goes up

THE SECOND ONE IS THE INTERESTING ONE
-------------------------------------
Every other fault makes a robot obviously worse. The wrong-position fault
makes it *look fine*. Its sensor is perfect, its map is internally
consistent, its energy accounting is normal, and it will happily tell you
where it is. It is simply wrong, and it poisons every map it is merged
into. Nothing a robot can check about itself will find it -- which is the
argument for having three robots and taking a vote, and it is the reason
this project is about fault tolerance rather than fault avoidance.
"""

import config


def _apply_sensor_degradation(member):
    """Dirty, short-sighted, unreliable optics."""
    member.sensor_noise_std = config.FAULT_SENSOR_NOISE_STD_M
    member.sensor_range_scale = config.FAULT_SENSOR_RANGE_SCALE
    member.sensor_dropout_prob = config.FAULT_SENSOR_DROPOUT_PROB


def _apply_wrong_position(member):
    """
    Displace what the robot BELIEVES about where it is.

    Note which three variables are touched: bx, by, btheta. The true pose
    is left exactly as it was, because nothing has physically moved -- the
    robot has mis-associated its scan with the wrong pipe-rack bay. From
    this step on it drives to the wrong places for the right reasons and
    writes correct scans into the wrong cells.
    """
    dx, dy = config.FAULT_POSE_OFFSET_M
    member.robot.bx += dx
    member.robot.by += dy
    member.robot.btheta += config.FAULT_POSE_OFFSET_RAD


def _apply_comms_loss(member):
    """Radio dead. The robot is not broken, it is alone -- and it does not
    know that either, since nothing tells a radio that nobody is hearing
    it."""
    member.robot.connected = False


def _apply_immobilised(member):
    """Motors dead. It still senses, still maps, still relays messages."""
    member.robot.mobile = False


def _apply_battery_drain(member):
    """Charge disappears faster than the work being done justifies."""
    member.robot.battery_drain_multiplier = config.FAULT_BATTERY_DRAIN_MULTIPLIER


_HANDLERS = {
    "sensor_degradation": _apply_sensor_degradation,
    "wrong_position": _apply_wrong_position,
    "comms_loss": _apply_comms_loss,
    "immobilised": _apply_immobilised,
    "battery_drain": _apply_battery_drain,
}


class FaultInjector:
    """
    Applies a schedule of faults as the mission runs.

    The schedule is a list of (robot_id, step, fault_name). Held here
    rather than on the robots so that no robot can read its own schedule,
    which keeps the "nobody knows it is broken" property honest.
    """

    def __init__(self, schedule=None):
        if schedule is None:
            schedule = config.FAULT_INJECTIONS
        self.schedule = list(schedule)
        for robot_id, step, name in self.schedule:
            if name not in _HANDLERS:
                raise ValueError(
                    f"unknown fault '{name}' -- expected one of "
                    f"{', '.join(config.FAULT_TYPES)}")
        self.fired = []          # (robot_id, step, name), for the report

    def apply_due(self, squad, step):
        """
        Fire every fault scheduled for this step. Returns what fired.

        Walks the schedule in order rather than the squad, so two faults
        scheduled for the same robot at the same step apply in the order
        they were written -- which keeps the run reproducible.
        """
        fired_now = []
        for robot_id, at_step, name in self.schedule:
            if at_step != step:
                continue
            for member in squad:
                if member.id != robot_id:
                    continue
                _HANDLERS[name](member)
                member.faults.append((step, name))
                self.fired.append((robot_id, step, name))
                fired_now.append((robot_id, step, name))
        return fired_now

    def __repr__(self):
        return f"<FaultInjector {len(self.schedule)} scheduled, {len(self.fired)} fired>"
