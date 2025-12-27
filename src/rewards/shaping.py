import numpy as np
from src.physics.flocking import compute_centroid, sheep_in_gate
from src.env.constants import FIELD_WIDTH, FIELD_HEIGHT


def compute_reward(sheep, dogs, world, prev_centroid=None):
    centroid = compute_centroid(sheep)
    in_gate = sheep_in_gate(sheep, world.gate)

    reward = -0.05  # time pressure

    # Primary objective
    reward += in_gate * 2.0

    # Weak shaping only
    if prev_centroid is not None:
        prev_dist = np.linalg.norm(prev_centroid - world.gate.center)
        curr_dist = np.linalg.norm(centroid - world.gate.center)
        reward += np.clip(prev_dist - curr_dist, -1.0, 1.0) * 0.3

    return reward