import numpy as np
from stable_baselines3 import PPO
from src.env.gym_shepherd_env import ShepherdGymEnv
from tqdm import tqdm

# -----------------------------
# Parameters
# -----------------------------
NUM_EPISODES = 10
MODEL_PATH = "./models/ppo_shepherd"

# -----------------------------
# Load model
# -----------------------------
model = PPO.load(MODEL_PATH)

# -----------------------------
# Evaluation loop
# -----------------------------
successes = 0
steps_to_success = []

for ep in tqdm(range(NUM_EPISODES), desc="Evaluating", ncols=100):
    env = ShepherdGymEnv(render_mode=None)
    obs, _ = env.reset(seed=ep)
    done = False
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        steps += 1

    if info.get("success", False):
        successes += 1
        steps_to_success.append(steps)

    env.close()

# -----------------------------
# Summary
# -----------------------------
print(f"Success rate: {successes}/{NUM_EPISODES}")
if steps_to_success:
    print(f"Avg steps (successful): {np.mean(steps_to_success):.1f}")
    