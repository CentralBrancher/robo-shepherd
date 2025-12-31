import numpy as np
from src.physics.flocking import compute_centroid, compute_flock_radius, sheep_in_gate
from src.utils.math import normalize
from src.env.constants import FIELD_WIDTH, FIELD_HEIGHT


def compute_reward(sheep, dogs, world, prev_centroid=None):
    centroid = compute_centroid(sheep)
    radius = compute_flock_radius(sheep, centroid)
    gate_pos = world.gate.center

    # ─────────────────────────────
    # Base time penalty
    # ─────────────────────────────
    reward = -0.05

    # ─────────────────────────────
    # 1. Continuous centroid progress toward gate (PRIMARY DRIVER)
    # ─────────────────────────────
    if prev_centroid is not None:
        prev_dist = np.linalg.norm(prev_centroid - gate_pos)
        curr_dist = np.linalg.norm(centroid - gate_pos)
        progress = np.clip(prev_dist - curr_dist, -5.0, 5.0)
        reward += progress * 5.0

        # Penalize stalled flock
        centroid_motion = np.linalg.norm(centroid - prev_centroid)
        reward -= np.exp(-centroid_motion * 4.0) * 0.5

    # ─────────────────────────────
    # 2. Terminal success reward (NOT exploitable)
    # ─────────────────────────────
    in_gate = sheep_in_gate(sheep, world.gate)
    reward += in_gate * 0.2  # shaping only

    # ─────────────────────────────
    # 3. Flock compactness (early pressure)
    # ─────────────────────────────
    reward -= radius * 0.01

    # ─────────────────────────────
    # 4. Penalize dogs blocking gate direction (continuous)
    # ─────────────────────────────
    to_gate = normalize(gate_pos - centroid)

    for dog in dogs:
        dog_vec = normalize(dog.pos - centroid)
        block = np.clip(np.dot(dog_vec, to_gate), 0.0, 1.0)
        reward -= block * 2.0

    # ─────────────────────────────
    # 5. Anti wall-hugging penalty (CRITICAL)
    # ─────────────────────────────
    for dog in dogs:
        wall_dist = min(
            dog.pos[0],
            FIELD_WIDTH - dog.pos[0],
            dog.pos[1],
            FIELD_HEIGHT - dog.pos[1]
        )
        reward -= np.exp(-wall_dist / 50.0) * 2.0

    # ─────────────────────────────
    # 6. Mild sheep-wall interaction penalty
    # (prevents pinning sheep directly)
    # ─────────────────────────────
    for s in sheep:
        wall_dist = min(
            s.pos[0],
            FIELD_WIDTH - s.pos[0],
            s.pos[1],
            FIELD_HEIGHT - s.pos[1]
        )
        reward -= np.exp(-wall_dist / 40.0) * 0.02

    return reward
