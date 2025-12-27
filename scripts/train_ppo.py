import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from src.env.gym_shepherd_env import ShepherdGymEnv

def make_env():
    return ShepherdGymEnv(render_mode=None)

env = DummyVecEnv([make_env])

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
    gamma=0.99,
    ent_coef=0.01,
)

checkpoint_callback = CheckpointCallback(
    save_freq=50_000, save_path="./models/",
    name_prefix="ppo_shepherd"
)

TOTAL_TIMESTEPS = 500_000

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback
)

model.save("./models/ppo_shepherd_2dogs")
print("Training complete! Model saved to './models/ppo_shepherd_2dogs.zip'")
