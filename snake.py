import gym
import numpy as np
import random
from gym import spaces
import pygame

class SnakeGameEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, width=400, height=400, cell_size=20):
        super(SnakeGameEnv, self).__init__()

        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.columns = width // cell_size
        self.rows = height // cell_size

        self.action_space = spaces.Discrete(4)  # UP, DOWN, LEFT, RIGHT
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.rows, self.columns, 3), dtype=np.uint8)

        self.reset()

    def reset(self):
        self.snake = [(self.columns // 2, self.rows // 2)]
        self.direction = (1, 0)
        self.done = False
        self.score = 0
        self._place_food()
        return self._get_observation()

    def _place_food(self):
        while True:
            self.food = (random.randint(0, self.columns - 1), random.randint(0, self.rows - 1))
            if self.food not in self.snake:
                break

    def step(self, action):
        if action == 0 and self.direction != (0, 1):  # UP
            self.direction = (0, -1)
        elif action == 1 and self.direction != (0, -1):  # DOWN
            self.direction = (0, 1)
        elif action == 2 and self.direction != (1, 0):  # LEFT
            self.direction = (-1, 0)
        elif action == 3 and self.direction != (-1, 0):  # RIGHT
            self.direction = (1, 0)

        new_head = ((self.snake[0][0] + self.direction[0]) % self.columns,
                    (self.snake[0][1] + self.direction[1]) % self.rows)

        reward = 0

        if new_head in self.snake:
            self.done = True
            reward = -1
        else:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                reward = 1
                self.score += 1
                self._place_food()
            else:
                self.snake.pop()

        return self._get_observation(), reward, self.done, {}

    def _get_observation(self):
        obs = np.zeros((self.rows, self.columns, 3), dtype=np.uint8)
        for x, y in self.snake:
            obs[y, x] = [0, 255, 0]  # Green snake
        fx, fy = self.food
        obs[fy, fx] = [255, 0, 0]  # Red food
        return obs

    def render(self, mode='human'):
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Snake RL")

        self.screen.fill((0, 0, 0))

        for x, y in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0),
                             (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))

        fx, fy = self.food
        pygame.draw.rect(self.screen, (255, 0, 0),
                         (fx * self.cell_size, fy * self.cell_size, self.cell_size, self.cell_size))

        pygame.display.flip()

    def close(self):
        if hasattr(self, 'screen'):
            pygame.quit()


# To use with stable-baselines3:
from stable_baselines3 import DQN
env = SnakeGameEnv()
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("snake_dqn")

obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()