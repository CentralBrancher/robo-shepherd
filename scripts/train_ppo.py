from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from scripts.callbacks import ProgressCallback, make_checkpoint_callback
from src.env.gym_shepherd_env import ShepherdGymEnv

# -----------------------------
# Parallel environments
# -----------------------------
NUM_ENVS = 8
env = DummyVecEnv([lambda: ShepherdGymEnv(render_mode=None) for _ in range(NUM_ENVS)])

# -----------------------------
# PPO model
# -----------------------------
MODEL_PATH = "./models/ppo_shepherd_1000000_steps.zip"

model = PPO.load(
    MODEL_PATH,
    env,
    verbose=1
)

# model = PPO(
#     "MlpPolicy",
#     env,
#     verbose=1,
#     n_steps=1024,
#     batch_size=128,
#     learning_rate=3e-4,
#     gamma=0.99,
#     ent_coef=0.01,
#     clip_range=0.2,
#     n_epochs=10
# )

# -----------------------------
# Callbacks
# -----------------------------
progress_callback = ProgressCallback()
checkpoint_callback = make_checkpoint_callback()

# -----------------------------
# Train
# -----------------------------
TOTAL_TIMESTEPS = int(1E12)
try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        reset_num_timesteps=False,
        callback=[checkpoint_callback, progress_callback]
    )
except KeyboardInterrupt:
    model.save("./models/ppo_shepherd_interrupt")

# Save final model
model.save("./models/ppo_shepherd")
print("Training complete! Model saved.")
