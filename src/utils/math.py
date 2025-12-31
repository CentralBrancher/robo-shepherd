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

def soft_wall_force(pos, margin=40.0, strength=1.0):
    fx = fy = 0.0
    if pos[0] < margin:
        fx += (margin - pos[0]) / margin
    elif pos[0] > FIELD_WIDTH - margin:
        fx -= (pos[0] - (FIELD_WIDTH - margin)) / margin

    if pos[1] < margin:
        fy += (margin - pos[1]) / margin
    elif pos[1] > FIELD_HEIGHT - margin:
        fy -= (pos[1] - (FIELD_HEIGHT - margin)) / margin

    return np.array([fx, fy]) * strength
