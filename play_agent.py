from stable_baselines3 import DQN
from snake_env import SnakeGameEnv

env = SnakeGameEnv()
model = DQN.load("snake_dqn")  # Load your trained model

obs = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _  = env.step(action)
    env.render()

env.close()