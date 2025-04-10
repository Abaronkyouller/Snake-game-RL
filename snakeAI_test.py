import pygame
from stable_baselines3 import DQN
from snake import SnakeGameEnv

env = SnakeGameEnv()
model = DQN.load("snake_dqn", env=env)

episodes = 5

for episode in range(episodes):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        score += reward

        env.render()
        pygame.time.delay(100)

    print(f"Episode {episode + 1} score: {score}")

env.close()
