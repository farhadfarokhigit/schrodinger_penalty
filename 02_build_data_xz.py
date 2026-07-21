"""
Stage 2 of 2: convert the raw penalty records from Stage 1 into the
(N,2) array of on-target (x,z) locations, in metres, that the
fitting/plotting code expects.

Input:  penalties_raw.json  (produced by 01_fetch_penalties.py)
Output: data_xz.npy         (N,2) float array, columns = (x, z) in metres

Coordinate conversion, and why:
  StatsBomb's pitch coordinate system is 120 (length) x 80 (width) units,
  which corresponds to a real pitch in yards (i.e. 1 unit = 1 yard); the
  goal line is at x=120, goal centre at y=40, goal width 8 yards (so the
  frame spans y in [36,44]), and shot end_location gives [x, y, z] with
  z as height in the same yard-based units. We convert to:
    x_metres = (y - 40) * 0.9144      # lateral offset from goal centre
    z_metres = z * 0.9144             # height above the ground
  (0.9144 m/yard is the exact yard-to-metre conversion.)

"On-target" filtering: a shot is kept only if it stayed within the goal
frame (36 <= y <= 44 and 0 <= z <= 2.67 yards, i.e. within the posts and
below the crossbar with a little slack for StatsBomb's own rounding).
Shots that missed the frame entirely are execution errors, not a chosen
target, and are excluded -- this is what turns the full penalty count
(1,481 in our run) into the smaller on-target count (1,401) used for
every fit in this project.
"""

import json
import numpy as np

YARD_TO_M = 0.9144
GOAL_CENTRE_Y = 40.0
GOAL_Y_MIN, GOAL_Y_MAX = 36.0, 44.0
GOAL_Z_MIN, GOAL_Z_MAX = 0.0, 2.67


def penalty_to_xz(p):
    """Return (x_metres, z_metres, on_target_bool) for one raw penalty record,
    or (None, None, False) if the record has no usable end_location."""
    end = p.get('end_location')
    if not end:
        return None, None, False
    y = end[1]
    z = end[2] if len(end) > 2 else 0.0   # a few records omit z; treat as ground level
    on_target = (GOAL_Y_MIN <= y <= GOAL_Y_MAX) and (GOAL_Z_MIN <= z <= GOAL_Z_MAX)
    x_m = (y - GOAL_CENTRE_Y) * YARD_TO_M
    z_m = z * YARD_TO_M
    return x_m, z_m, on_target


def build_data_xz(penalties):
    rows = []
    n_total = len(penalties)
    n_no_location = 0
    for p in penalties:
        x_m, z_m, on_target = penalty_to_xz(p)
        if x_m is None:
            n_no_location += 1
            continue
        if on_target:
            rows.append((x_m, z_m))
    data = np.array(rows, dtype=float)
    print(f"total penalty records: {n_total}")
    print(f"  missing end_location entirely: {n_no_location}")
    print(f"  on-target (kept): {len(data)}")
    print(f"  off-target (excluded, execution error): {n_total - n_no_location - len(data)}")
    return data


if __name__ == '__main__':
    with open('penalties_raw.json') as f:
        penalties = json.load(f)

    data_xz = build_data_xz(penalties)
    np.save('data_xz.npy', data_xz)
    print(f"\nsaved data_xz.npy with shape {data_xz.shape}")
    print(f"mean x = {data_xz[:,0].mean():.3f} m, mean z = {data_xz[:,1].mean():.3f} m")
