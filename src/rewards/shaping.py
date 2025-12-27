import numpy as np
from src.physics.flocking import compute_centroid, sheep_in_gate
from src.physics.directional import directional_pressure
from src.env.constants import FIELD_WIDTH, FIELD_HEIGHT

def compute_reward(sheep, dogs, world, prev_centroid=None):
    """
    Reward function for shepherding:
    - Primary objective: sheep in gate
    - Shaping: flock centroid approaching gate
    - Dog incentives: stay near flock, reduce distance to gate, apply directional pressure
    """

    reward = -0.05  # small time penalty to encourage faster herding

    # 1. Primary objective: sheep in gate
    in_gate = sheep_in_gate(sheep, world.gate)
    reward += in_gate * 2.0

    # 2. Flock centroid shaping
    centroid = compute_centroid(sheep)
    if prev_centroid is not None:
        prev_dist = np.linalg.norm(prev_centroid - world.gate.center)
        curr_dist = np.linalg.norm(centroid - world.gate.center)
        # Reward reduction in distance to gate
        reward += np.clip(prev_dist - curr_dist, -10.0, 10.0) * 1.0

    # 3. Dog proximity and directional shaping
    for dog in dogs:
        dist_to_centroid = np.linalg.norm(dog.pos - centroid)
        reward -= 0.01 * dist_to_centroid  # penalize being far from flock

        dist_to_gate = np.linalg.norm(dog.pos - world.gate.center)
        reward += 0.001 * (FIELD_WIDTH - dist_to_gate)  # incentive toward gate

        # Directional pressure reward: dogs behind flock
        flock_dir = world.gate.center - centroid
        pressure = directional_pressure(
            sheep_pos=centroid,
            dog_pos=dog.pos,
            gate_pos=world.gate.center
        )
        reward += 0.5 * pressure  # encourage proper positioning behind flock

    # 4. Optional: small shaping for flock spread (penalize very dispersed flock)
    flock_radius = max(np.linalg.norm(s.pos - centroid) for s in sheep)
    reward -= 0.01 * flock_radius  # encourages cohesive flocking

    return reward
