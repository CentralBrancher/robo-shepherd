from src.env.gym_shepherd_env import ShepherdGymEnv

env = ShepherdGymEnv(render_mode="human")
obs, _ = env.reset()
done = False

while not done:
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    done = term or trunc

env.close()
