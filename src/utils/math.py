import numpy as np

def limit_magnitude(vec, max_val):
    mag = np.linalg.norm(vec)
    if mag > max_val and mag > 0:
        return vec / mag * max_val
    return vec

def normalize(vec):
    mag = np.linalg.norm(vec)
    if mag == 0:
        return vec
    return vec / mag
