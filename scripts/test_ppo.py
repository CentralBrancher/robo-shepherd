import numpy as np
from stable_baselines3 import PPO
from src.env.gym_shepherd_env import ShepherdGymEnv
import time

env = ShepherdGymEnv(render_mode="human")
model = PPO.load("models/ppo_shepherd_2dogs")

obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, success, done, info = env.step(action)
    time.sleep(1 / 60)  # optional: limit to 60 FPS

print("Episode finished. Success:", info["success"])
env.close()
