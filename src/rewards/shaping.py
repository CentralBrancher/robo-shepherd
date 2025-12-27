from src.physics.flocking import compute_centroid, compute_flock_radius, sheep_in_gate

def compute_reward(sheep, world):
    centroid = compute_centroid(sheep)
    radius = compute_flock_radius(sheep, centroid)
    in_gate = sheep_in_gate(sheep, world.gate)

    return in_gate * 5.0 - radius * 0.01
