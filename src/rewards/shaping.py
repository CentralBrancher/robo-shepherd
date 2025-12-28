import numpy as np
from src.physics.flocking import compute_centroid, compute_flock_radius, sheep_in_gate
from src.utils.math import normalize
from src.env.constants import FIELD_WIDTH, FIELD_HEIGHT

def compute_reward(sheep, dogs, world, prev_centroid=None):
    centroid = compute_centroid(sheep)
    radius = compute_flock_radius(sheep, centroid)

    gate_pos = world.gate.center
    dist_to_gate = np.linalg.norm(centroid - gate_pos)

    # Smooth phase weight:
    # far → 0 (herding phase)
    # near → 1 (gate precision phase)
    gate_focus = np.exp(-dist_to_gate / 200.0)

    reward = -0.1  # time penalty (forces completion)

    # ─────────────────────────────
    # 1. Progress toward gate (dominant late)
    # ─────────────────────────────
    if prev_centroid is not None:
        prev_dist = np.linalg.norm(prev_centroid - gate_pos)
        progress = np.clip(prev_dist - dist_to_gate, -5.0, 5.0)
        reward += gate_focus * progress * 3.0

    # ─────────────────────────────
    # 2. Strong gate success reward
    # ─────────────────────────────
    in_gate = sheep_in_gate(sheep, world.gate)
    reward += in_gate * 40.0

    # ─────────────────────────────
    # 3. Flock compactness (early only)
    # ─────────────────────────────
    reward -= (1.0 - gate_focus) * radius * 0.02

    # ─────────────────────────────
    # 4. Penalize dogs blocking the gate
    # ─────────────────────────────
    to_gate = normalize(gate_pos - centroid)

    for dog in dogs:
        dog_vec = normalize(dog.pos - centroid)

        # Dog in front of flock relative to gate direction
        if np.dot(dog_vec, to_gate) > 0.3:
            reward -= 1.0

    # ─────────────────────────────
    # 5. Mild dog wall penalty (not dominant)
    # ─────────────────────────────
    for dog in dogs:
        if (
            dog.pos[0] < 20 or dog.pos[0] > FIELD_WIDTH - 20 or
            dog.pos[1] < 20 or dog.pos[1] > FIELD_HEIGHT - 20
        ):
            reward -= 0.3

    return reward
