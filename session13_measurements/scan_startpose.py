"""
How many experiment seeds put a deviation on top of a fixed start pose?

Builds the facility and injects the deviations for every seed in
EXPERIMENT_SEEDS, then runs the same clearance test demo_squad.run() does.
No missions are run -- this is seconds, not hours.
"""
import sys

sys.path.insert(0, r"C:\Users\User\Desktop\Final Year Project\Project"
                   r"\Final Year Project\Dummy Simulation_Modified")

import config  # noqa: E402
from environment import Facility  # noqa: E402
from inspection import generate_inspection_points  # noqa: E402
from deviations import inject_deviations  # noqa: E402
from demo_squad import reachability_planner  # noqa: E402

bad = {}
for seed in config.EXPERIMENT_SEEDS:
    for label, with_dev in (("C0", False), ("C1-C5", True)):
        facility = Facility()
        ranker = reachability_planner(facility)
        points = generate_inspection_points(facility, seed, ranker)
        inject_deviations(facility, points, seed, ranker, verbose=False,
                          count=None if with_dev else 0)

        for n_robots in (1, config.SQUAD_SIZE):
            poses = config.SQUAD_START_POSES[:n_robots]
            for i, (x, y, _th) in enumerate(poses):
                if not facility._has_clearance(
                        x, y, config.ROBOT_RADIUS_M + 0.3):
                    bad.setdefault(seed, []).append(
                        (label, n_robots, i, x, y))
    print(f"  seed {seed:>5d} {'BLOCKED' if seed in bad else 'ok'}",
          flush=True)

print()
if not bad:
    print("no seed blocks a start pose")
else:
    print(f"{len(bad)} of {len(config.EXPERIMENT_SEEDS)} seeds block a "
          f"start pose:")
    for seed, hits in sorted(bad.items()):
        where = sorted({(h[0], h[2], h[3], h[4]) for h in hits})
        print(f"  seed {seed}: " + ", ".join(
            f"{w[0]} pose {w[1]} at ({w[2]}, {w[3]})" for w in where))
