from stable_baselines3 import DQN
from snake_env import SnakeGameEnv
import torch
env = SnakeGameEnv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    exploration_fraction=0.3,
    device=device
)
model.learn(total_timesteps=100_000)
model.save("snake_dqn")

# Evaluation
test_env = SnakeGameEnv()
obs = test_env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = test_env.step(action)
    #test_env.render()

test_env.close()