from src.physics.flocking import compute_centroid, compute_flock_radius, sheep_in_gate

def compute_reward(sheep, world, prev_centroid=None):
    centroid = compute_centroid(sheep)
    radius = compute_flock_radius(sheep, centroid)
    in_gate = sheep_in_gate(sheep, world.gate)

    reward = -0.1  # timestep penalty

    if prev_centroid is not None:
        prev_dist = np.linalg.norm(prev_centroid - world.gate.center)
        curr_dist = np.linalg.norm(centroid - world.gate.center)
        reward += (prev_dist - curr_dist) * 0.5

    reward -= radius * 0.02
    reward += in_gate * 1.0
    return reward
