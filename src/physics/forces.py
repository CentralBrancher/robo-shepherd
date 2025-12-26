import numpy as np
from src.env.constants import FIELD_WIDTH, FIELD_HEIGHT

def wall_repulsion(pos, margin=30, strength=1.5):
    force = np.zeros(2)

    # Left wall
    if pos[0] < margin:
        force[0] += (margin - pos[0]) / margin

    # Right wall
    if pos[0] > FIELD_WIDTH - margin:
        force[0] -= (pos[0] - (FIELD_WIDTH - margin)) / margin

    # Top wall
    if pos[1] < margin:
        force[1] += (margin - pos[1]) / margin

    # Bottom wall
    if pos[1] > FIELD_HEIGHT - margin:
        force[1] -= (pos[1] - (FIELD_HEIGHT - margin)) / margin

    return force * strength
