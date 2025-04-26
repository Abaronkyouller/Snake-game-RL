import gym
import numpy as np
import random
import time
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
        self.steps_since_food = 0
        self.action_space = spaces.Discrete(4)  # UP, DOWN, LEFT, RIGHT

        # Use correct RGB range (0-255)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.rows, self.columns, 3), dtype=np.uint8
        )

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
        old_distance = np.linalg.norm(np.array(self.snake[0]) - np.array(self.food))

        if action == 0 and self.direction != (0, 1):  # UP
            self.direction = (0, -1)
        elif action == 1 and self.direction != (0, -1):  # DOWN
            self.direction = (0, 1)
        elif action == 2 and self.direction != (1, 0):  # LEFT
            self.direction = (-1, 0)
        elif action == 3 and self.direction != (-1, 0):  # RIGHT
            self.direction = (1, 0)

        new_head = (
            (self.snake[0][0] + self.direction[0]) % self.columns,
            (self.snake[0][1] + self.direction[1]) % self.rows
        )

        reward = -0.1 

        if new_head in self.snake:
            self.done = True
            reward = -10 
        else:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                reward = 10 
                self.score += 1
                self.steps_since_food = 0
                self._place_food()
            else:
                self.snake.pop()
                self.steps_since_food += 1

        if self.steps_since_food > 100:
            self.done = True
            reward = -10

        new_distance = np.linalg.norm(np.array(new_head) - np.array(self.food))
        reward += (old_distance - new_distance) * 0.5

        return self._get_observation(), reward, self.done, {}

    def _get_observation(self):
        obs = np.zeros((self.rows, self.columns, 3), dtype=np.uint8)
        for x, y in self.snake:
            obs[y, x] = [0, 255, 0]  # Green snake
        fx, fy = self.food
        obs[fy, fx] = [255, 0, 0]  # Red food
        return obs

    def render(self, mode='human'):
        if mode == 'human':
            if not hasattr(self, 'screen'):
                pygame.init()
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("Snake RL")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

            self.screen.fill((0, 0, 0))

            for x, y in self.snake:
                pygame.draw.rect(self.screen, (0, 255, 0),
                                (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))

            fx, fy = self.food
            pygame.draw.rect(self.screen, (255, 0, 0),
                            (fx * self.cell_size, fy * self.cell_size, self.cell_size, self.cell_size))

            pygame.display.flip()
            time.sleep(0.1) 

    def close(self):
        if hasattr(self, 'screen'):
            pygame.quit()
            del self.screen