"""
recovery.py
===========
What the squad does about a robot it has decided is faulty. Step 5.

THE RESPONSE DEPENDS ON WHAT BROKE
----------------------------------
Throwing every suspect robot away wastes good data and good hardware, and
it is also wrong: three of the five faults leave the robot's map perfectly
trustworthy. Only one of them justifies erasing anything.

    fault                map trust   rollback   reallocate its work
    ------------------   ---------   --------   -------------------
    wrong_position       none        YES        yes
    sensor_degradation   reduced     no         no
    comms_loss           unchanged   no         yes
    immobilised          unchanged   no         yes
    battery_drain        unchanged   no         yes

Reading that table row by row:

  WRONG POSITION is the only quarantine. The robot's map is internally
  consistent and globally misplaced, so every cell it ever contributed is
  wrong and has to come back out. rollback() exists for exactly this.

  SENSOR DEGRADATION is not a lie, it is a poor view. The robot still sees
  walls, just badly. Down-weight its contributions rather than discarding
  them, and leave it working -- it can still drive to inspection points,
  and it can still carry messages.

  COMMS LOSS IS NOT A FAULT IN THE ROBOT AT ALL. It is isolation. That
  robot is healthy, its map is good, and it is still working through its
  assignment alone. Quarantining it would throw away perfectly good data
  for the crime of being behind a storage tank. What the others do is take
  over the work it can no longer coordinate on, and merge its map when it
  comes back. Handling this differently from a genuine fault is the piece
  of engineering judgement the spec singles out.

  IMMOBILISED and BATTERY DRAIN are honest robots that cannot finish. Take
  their work, keep their maps. The battery case is the one where acting
  early actually matters, because the detector fires while the robot is
  still alive.

WHAT THIS FILE WILL NOT DO ON ONE ROBOT'S WORD
----------------------------------------------
Nothing. Every response requires RECOVERY_MIN_ACCUSERS independent robots
to have reached the same conclusion. A robot with a dead radio accuses
everybody -- correctly, from its own evidence -- and if a single
accusation were enough, the one broken robot would quarantine the two
healthy ones and the mission would collapse in the name of fault
tolerance.

A LIMITATION, STATED RATHER THAN HIDDEN
---------------------------------------
The spec asks that a degraded or immobilised robot be kept on as a message
RELAY. The radio in comms.py is single-hop: a message either reaches a
robot directly or it does not, and no robot forwards anything. So "keep it
as a relay" has nothing to attach to here, and what is implemented is the
part that does: the robot stays in the squad, keeps its claims where
appropriate, and keeps contributing what it can. Multi-hop forwarding
would be a genuine addition to the comms model rather than a Step 5
behaviour.
"""

import config


def response_for(fault_name):
    """
    How to react to a corroborated accusation.

    Returns a dict with three keys:
        trust      what this robot's map contributions are now worth
        rollback   whether to remove everything it has already contributed
        reallocate whether to release its claims for somebody else to take

    Read from config at call time rather than baked into a module-level
    table, so an experiment can change RECOVERY_SENSOR_TRUST without
    touching this file.
    """
    if fault_name == "wrong_position":
        # The only true quarantine, and the ONLY fault that puts completed
        # inspections in doubt. This robot was not where it thought it was,
        # so neither its map nor its "inspected" reports mean anything.
        return {"trust": 0.0, "rollback": True, "reallocate": True,
                "invalidate": True}

    if fault_name == "sensor_degradation":
        # A poor view, not a false one. Keep it working and keep its data,
        # worth less.
        return {"trust": config.RECOVERY_SENSOR_TRUST, "rollback": False,
                "reallocate": False, "invalidate": False}

    if fault_name in ("comms_loss", "immobilised", "battery_drain"):
        # Honest robots that cannot finish, or cannot coordinate. Their
        # maps are as good as anyone's -- AND SO ARE THEIR COMPLETED
        # INSPECTIONS. A robot with a dead radio, stuck wheels or a flat
        # battery was exactly where it thought it was when it read that
        # gauge. Discarding that work would be throwing away verified
        # inspections to punish a robot for a fault that never touched
        # them.
        return {"trust": 1.0, "rollback": False, "reallocate": True,
                "invalidate": False}

    raise ValueError(f"no recovery response defined for '{fault_name}'")


def corroborated(member):
    """
    Every (suspect, fault) anybody believes, with how many robots believe
    it. Yields (suspect, fault, accusers).

    NO THRESHOLD IS APPLIED HERE. How many accusers are needed depends on
    what is about to be done, not on what is alleged, so the count is
    reported and the caller decides. Gating here instead was what made
    comms loss unactionable.

    Note what is NOT here: any pooling of evidence. Robots exchange
    conclusions, not observations, and each one reached its own
    independently. Two robots agreeing is two separate measurements
    agreeing, which is what makes the corroboration worth anything.

    TWO THINGS A NAIVE TALLY GETS WRONG -- this is M2, and it cost seed 42
    a healthy robot: accusers [1, 2] against suspect 0, where robot 1 was
    the injected victim. 129,383 cells rolled back, 14 inspections thrown
    away, and the robot that was actually broken never touched.

      1. A ROBOT THAT IS ITSELF ON TRIAL DOES NOT GET A VOTE. The
         two-accuser rule assumes two accusers are two independent healthy
         measurements. When one of them IS the patient, what you have is
         one honest mistake plus the fault itself, agreeing.

      2. AT MOST ONE ROBOT IN THREE IS FAULTY, so accusations pointing in
         several directions cannot all be true. Rather than acting on
         everyone named, the squad resolves to the robot the most peers
         have named and dismisses the rest. Acting on all of them is how a
         single displaced robot took a healthy one down with it.

    Both rules only bite when accusations point in more than one direction.
    On a clean detection -- which is what the arithmetic in
    check_wrong_position produces, because a displaced robot disagrees with
    both peers and so accuses nobody -- there is exactly one suspect, no
    accuser is under suspicion, and nothing here changes the outcome.

    A note on the isolated robot, because it looks like it should break:
    a robot whose radio is dead accuses everybody, but nobody hears it. Its
    accusations never reach anyone else's heard_suspicions, so they cannot
    make its healthy peers look accused, and comms-loss reallocation is
    unaffected.
    """
    tally = {}

    for suspect, faults in member.detector.accusations.items():
        for fault in faults:
            tally.setdefault((suspect, fault), set()).add(member.id)

    for accuser, suspects in member.heard_suspicions.items():
        for suspect, faults in suspects.items():
            for fault in faults:
                tally.setdefault((suspect, fault), set()).add(accuser)

    if not tally:
        return

    # Who is under an accusation of any kind, and how many distinct robots
    # have named each of them. Faults are pooled for this count: the
    # question is "who does the squad think is broken", not "broken how".
    named_by = {}
    for (suspect, fault), accusers in tally.items():
        named_by.setdefault(suspect, set()).update(accusers)
    accused = set(named_by)

    # Rule 2: resolve to the most-accused robot. Ties break to the lowest
    # id purely so the run stays deterministic -- on a tie neither reaches
    # a quarantine quorum anyway, once rule 1 has been applied.
    primary = min(accused, key=lambda s: (-len(named_by[s]), s))

    for (suspect, fault), accusers in tally.items():
        if suspect == member.id:
            continue          # nobody acts on accusations against itself
        if suspect != primary:
            continue          # at most one of us is broken, and it is not this one

        # Rule 1: drop any accuser that is itself under accusation.
        independent = sorted(a for a in accusers if a not in accused)
        if not independent:
            continue
        yield suspect, fault, independent


def _invalidate_completions(member, suspect, points):
    """
    Take back every inspection the quarantined robot reported.

    DISTRUST IS TOTAL, AND IT HAS TO BE INDIRECT. A displaced robot cannot
    know it is displaced, and neither can anybody else know which of its
    reported inspections were real -- checking would need the true
    positions, which no robot has. So the squad cannot pick and choose. It
    can only apply the same rule it applies to the map: if this robot's
    observations are no longer trusted, neither are its completions, and
    all of them go back in the pool.

    This mirrors rollback() exactly. rollback() does not remove the cells a
    Byzantine robot got wrong, it removes everything the robot ever
    contributed, because there is no way to tell the two apart from the
    inside.

    IT COSTS SOMETHING AND THAT COST IS A RESULT. Some of those inspections
    were perfectly good -- everything the robot did before the fault, and
    anything it happened to get right afterwards. Re-inspecting them burns
    energy and time the squad did not need to spend. That is the price of
    not being able to tell good work from bad, and Chapter 4 should quote
    it rather than hide it.
    """
    by_index = {p.index: p for p in points}
    invalidated = []
    for index, who in list(member.done_by.items()):
        if who != suspect:
            continue
        member.done.discard(index)
        member.done_by.pop(index, None)
        invalidated.append(index)

        # The mission record follows the squad's belief. `truly_visited` is
        # deliberately NOT cleared -- whether a gauge was physically within
        # range is a fact about the world, not a claim anybody made about
        # it, and the report needs both numbers.
        p = by_index.get(index)
        if p is not None and p.visited_by == suspect:
            p.visited = False
            p.visited_by = None
            p.visit_step = None
            p.invalidated = True
    return sorted(invalidated)


def consider(member, points, step):
    """
    Act on anything the squad now agrees about.

    Called once per detector tick. Every action is idempotent and taken at
    most once per (suspect, fault), so this is safe to call repeatedly.
    Returns the actions taken this step, for the report.
    """
    if not member.robot.alive:
        return []

    taken = []
    for suspect, fault, accusers in corroborated(member):
        response = response_for(fault)
        n = len(accusers)
        restored = 0
        released = []
        did = []

        # Each part of the response is gated on its own evidence bar, and
        # they are recorded separately so a cheap action can go ahead now
        # while the destructive one waits for a second opinion that may
        # never arrive.

        # --- release its work for somebody else: cheap, one accuser ---
        if (response["reallocate"]
                and n >= config.RECOVERY_REALLOCATE_ACCUSERS
                and (suspect, fault, "release") not in member.recovery_applied):
            member.recovery_applied.add((suspect, fault, "release"))
            member.ignore_claims_from.add(suspect)
            for index, claim in list(member.claims.items()):
                if claim["by"] == suspect:
                    member.claims.pop(index, None)
                    released.append(index)
            member.released.update(released)

            # ...AND TAKE OVER ITS LANE. Under a static partition this is
            # what reallocation actually means: the failed robot's share of
            # the round becomes everybody else's to bid for, rather than
            # sitting unvisited because its owner cannot move. Releasing
            # only the one point it happened to be driving to would leave
            # the other twelve stranded.
            member.extra.update(member.lane_of.get(suspect, set()))
            did.append("released")

        # --- trust its map less: cheap while the data is kept ---------
        if (0.0 < response["trust"] < 1.0
                and n >= config.RECOVERY_REALLOCATE_ACCUSERS
                and (suspect, fault, "trust") not in member.recovery_applied):
            member.recovery_applied.add((suspect, fault, "trust"))
            member.trust[suspect] = response["trust"]
            did.append("down-weighted")

        # --- quarantine and erase: destructive, two accusers ----------
        invalidated = []
        if (response["rollback"]
                and n >= config.RECOVERY_QUARANTINE_ACCUSERS
                and (suspect, fault, "quarantine") not in member.recovery_applied):
            member.recovery_applied.add((suspect, fault, "quarantine"))
            member.trust[suspect] = response["trust"]
            before = member.grid.classified()
            member.grid.rollback(suspect)
            after = member.grid.classified()
            restored = int((before != after).sum())
            member.quarantined.add(suspect)
            did.append("quarantined")
            # Only a displaced robot's completions are in doubt. Gated
            # explicitly rather than left to fall out of `rollback` being
            # true, so that adding a future fault that rolls back the map
            # cannot silently start discarding verified inspections too.
            if response["invalidate"]:
                invalidated = _invalidate_completions(member, suspect,
                                                      points)

        if not did:
            continue

        action = {"step": step, "by": member.id, "suspect": suspect,
                  "fault": fault, "accusers": accusers, "did": did,
                  "trust": member.trust.get(suspect, 1.0),
                  "rolled_back": "quarantined" in did,
                  "cells_restored": restored,
                  "claims_released": released,
                  "invalidated": invalidated,
                  "outstanding": sum(1 for p in points if not p.visited)}
        member.recovery_actions.append(action)
        taken.append(action)

    return taken
