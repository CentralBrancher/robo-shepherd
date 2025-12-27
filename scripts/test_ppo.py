import numpy as np
from stable_baselines3 import PPO
from src.env.gym_shepherd_env import ShepherdGymEnv
from tqdm import tqdm

# -----------------------------
# Parameters
# -----------------------------
NUM_EPISODES = 10
MODEL_PATH = "./models/ppo_shepherd_2048"

# -----------------------------
# Load model
# -----------------------------
model = PPO.load(MODEL_PATH)

# -----------------------------
# Evaluation loop
# -----------------------------
episode_durations = []

for ep in tqdm(range(NUM_EPISODES), desc="Evaluating Episodes", ncols=100):
    env = ShepherdGymEnv(render_mode="human")
    obs, _ = env.reset()
    done = False
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    episode_durations.append(steps)
    env.close()

# -----------------------------
# Summary
# -----------------------------
avg_steps = np.mean(episode_durations)
min_steps = np.min(episode_durations)
max_steps = np.max(episode_durations)

print(f"\nEvaluation complete over {NUM_EPISODES} episodes")
print(f"Avg steps to success: {avg_steps:.1f}")
print(f"Min steps: {min_steps}, Max steps: {max_steps}")
