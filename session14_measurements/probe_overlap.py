"""
Why does displacing a robot FURTHER not make it easier to catch?

Stage 4a measured detection of wrong_position at 0/7, 2/7, 2/7, 3/7, 3/7
for displacements of 2, 4, 6, 12 and 24 m. It never exceeds 43 %, even at
24 m -- four pipe-rack bays, and 133x the 0.18 m of odometry drift a
healthy robot accumulates. Stage 4b then showed that loosening
BYZANTINE_RATIO from 1.4 to 1.15 buys no extra detection either. Neither
the fault size nor the decision threshold is what limits the detector.

THE HYPOTHESIS. Session 13's M3 fix computes all three pairwise conflict
rates over the TRIPLE-OVERLAP region -- cells all three robots have
evidence for -- and refuses to judge below
BYZANTINE_MIN_TRIPLE_OVERLAP_CELLS. A displaced robot drives to the wrong
places, so it shares less ground with its peers. If displacing it further
shrinks that shared region, then the very thing that should make the fault
visible is also destroying the evidence the detector runs on, and
detection would saturate no matter how gross the fault.

HOW THIS MEASURES IT. Wraps OccupancyGrid.contribution_conflicts_common
with a recorder -- no production code is modified -- and reports the
overlap sizes seen at each displacement, together with how many checks
cleared the floor.

Three runs on one seed. Cheap.
"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
import mapping
import run_experiments as rx

SEED = 42                       # a wrong_position seed
DISPLACEMENTS = [2.0, 6.0, 24.0]

seen = collections.defaultdict(list)
current = {"d": None}

original = mapping.OccupancyGrid.contribution_conflicts_common


def recording(self, sources):
    rates, overlap = original(self, sources)
    # Only three-way calls are the Byzantine test; ignore anything else.
    if len(list(sources)) >= 3:
        seen[current["d"]].append(overlap)
    return rates, overlap


mapping.OccupancyGrid.contribution_conflicts_common = recording

keep = config.FAULT_POSE_OFFSET_M
floor = config.BYZANTINE_MIN_TRIPLE_OVERLAP_CELLS
print("TRIPLE-OVERLAP SAMPLE AGAINST DISPLACEMENT -- seed %d, C3" % SEED)
print("BYZANTINE_MIN_TRIPLE_OVERLAP_CELLS = %d\n" % floor)

for d in DISPLACEMENTS:
    current["d"] = d
    config.FAULT_POSE_OFFSET_M = (d, 0.0)
    row = rx.run_one("C3", SEED, save_trace=False)
    v = np.array(seen[d], dtype=float)
    admitted = int((v >= floor).sum())
    print("  %5.1f m : %4d checks | overlap mean %8.0f  median %8.0f  "
          "max %8.0f | cleared the floor %4d (%5.1f %%) | detected %s"
          % (d, len(v), v.mean() if len(v) else 0,
             np.median(v) if len(v) else 0, v.max() if len(v) else 0,
             admitted, 100.0 * admitted / max(len(v), 1),
             row["fault_detected"]))

config.FAULT_POSE_OFFSET_M = keep
mapping.OccupancyGrid.contribution_conflicts_common = original

print("\nIf the overlap shrinks as displacement grows, the detector is")
print("losing the evidence it needs at exactly the rate the fault becomes")
print("more obvious, and that -- not the threshold -- is the limit.")
