import numpy as np

def compute_centroid(sheep):
    positions = np.array([s.pos for s in sheep])
    return positions.mean(axis=0)

def compute_flock_radius(sheep, centroid):
    return max(np.linalg.norm(s.pos - centroid) for s in sheep)

def sheep_in_gate(sheep, gate):
    return sum(gate.contains(s.pos) for s in sheep)
