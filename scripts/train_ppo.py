from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from scripts.callbacks import ProgressCallback, make_checkpoint_callback
from src.env.gym_shepherd_env import ShepherdGymEnv

# -----------------------------
# Parallel environments
# -----------------------------
NUM_ENVS = 4
env = DummyVecEnv([lambda: ShepherdGymEnv(render_mode=None) for _ in range(NUM_ENVS)])

# -----------------------------
# PPO model
# -----------------------------
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=1024,
    batch_size=64,
    learning_rate=3e-4,
    gamma=0.99,
    ent_coef=0.01
)

# -----------------------------
# Callbacks
# -----------------------------
progress_callback = ProgressCallback()
checkpoint_callback = make_checkpoint_callback()

# -----------------------------
# Train
# -----------------------------
TOTAL_TIMESTEPS = 20_000
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[checkpoint_callback, progress_callback]
)

# Save final model
model.save("./models/ppo_shepherd")
print("Training complete! Model saved.")
