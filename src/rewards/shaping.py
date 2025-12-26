from src.physics.flocking import compute_centroid, compute_flock_radius, sheep_in_gate

def compute_reward(sheep, world):
    centroid = compute_centroid(sheep)
    radius = compute_flock_radius(sheep, centroid)
    in_gate = sheep_in_gate(sheep, world.gate)

    reward = 0.0
    reward += in_gate * 5.0         # main objective
    reward -= radius * 0.01         # compactness penalty

    return reward
