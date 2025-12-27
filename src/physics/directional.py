import numpy as np
from src.utils.math import normalize

def directional_pressure(sheep_pos, dog_pos, gate_pos):
    """
    Returns a soft pressure value in [0.1, 1.0]:
    - 1.0 = dog directly behind sheep (good pressure)
    - 0.5 = dog to the side
    - 0.1 = dog in front of sheep (bad pressure)
    """
    to_gate = normalize(gate_pos - sheep_pos)
    dog_dir = normalize(sheep_pos - dog_pos)  # direction from dog to sheep

    alignment = np.dot(dog_dir, to_gate)  # [-1, 1]
    # Map [-1,1] -> [0.1, 1.0]
    pressure = np.clip((alignment + 1) / 2 * 0.9 + 0.1, 0.1, 1.0)
    return pressure
