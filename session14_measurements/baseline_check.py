"""
The CLAUDE.md three-seed regression baseline, re-measured.

demo_single.py takes no command-line arguments -- it always runs
DEFAULT_SEED -- so the three-seed table has to be driven from here.

Pass criteria (CLAUDE.md): coverage 95.5-99.0 % and waypoints skipped 0 on
every seed. The J/m2 band moves with the energy coefficients and is
re-recorded rather than treated as a failure. F1, collisions, distance and
odometry are informational.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import demo_single
import config

print("BASELINE -- demo_single, three seeds")
print("E_DRIVE=%.2f  E_TURN=%.2f  P_SENSE=%.2f  P_COMPUTE=%.2f"
      % (config.E_DRIVE_J_PER_M, config.E_TURN_J_PER_RAD,
         config.P_SENSE_W, config.P_COMPUTE_W))
print()
print("%6s %9s %8s %10s %7s %11s %8s" %
      ("seed", "Coverage", "J/m2", "Distance", "F1", "Collisions", "Skipped"))

fails = []
for seed in (42, 7, 2024):
    facility, grid, robot, history, trail = demo_single.run(seed=seed,
                                                            verbose=False)
    # Exactly the quantities demo_single.report() prints, so the numbers
    # are comparable with the CLAUDE.md table digit for digit.
    cov = grid.coverage_fraction(facility) * 100
    _, _, f1 = grid.surface_scores(facility)
    area = facility.free_area_m2() * cov / 100
    per_m2 = robot.total_energy_j / max(area, 1e-6)
    skipped = history.get("skipped", 0) if isinstance(history, dict) else 0

    print("%6d %8.2f %% %8.2f %9.1f m %7.3f %11d %8d" %
          (seed, cov, per_m2, robot.distance_travelled_m, f1,
           robot.collisions, skipped))

    if not (95.5 <= cov <= 99.0):
        fails.append(f"seed {seed}: coverage {cov:.2f} % outside 95.5-99.0")
    if skipped != 0:
        fails.append(f"seed {seed}: {skipped} waypoints skipped")

print()
if fails:
    print("FAIL:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("PASS on the two binding criteria (coverage, waypoints skipped).")
