import gymnasium as gym
import numpy as np
import random
import time
from gymnasium import spaces
import pygame

class SnakeGameEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, width=400, height=400, cell_size=10):
        super(SnakeGameEnv, self).__init__()

        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.columns = width // cell_size
        self.rows = height // cell_size
        self.steps_since_food = 0
        self.action_space = spaces.Discrete(4)  # UP, DOWN, LEFT, RIGHT

        self.observation_space = spaces.Box(
            low=-max(self.columns, self.rows),
            high=max(self.columns, self.rows),
            shape=(8,), # [food_x - head_x, food_y - head_y, dir_x, dir_y, danger_forward, danger_left, danger_right, steps_since_food]
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake = np.array([[self.columns // 2, self.rows // 2]])
        self.direction = (1, 0)
        self.done = False
        self.score = 0
        self._place_food()
        self.steps_since_food = 0
        return self._get_observation(), {}

    def _place_food(self):
        while True:
            self.food = (np.random.randint(0, self.columns), np.random.randint(0, self.rows))
            if not np.any(np.all(self.snake == self.food, axis=1)):
                break

    def step(self, action):
        self.steps_since_food += 1
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

        reward = -0.01

        if np.any(np.all(self.snake == new_head, axis=1)):
            self.done = True
            reward = -1.0
        else:
            self.snake = np.vstack([new_head, self.snake])
            if new_head == self.food:
                reward = 10.0
                self.score += 1
                self.steps_since_food = 0
                self._place_food()
            elif self.steps_since_food > 200:
                self.done = True
                reward = -2.0
            else:
                self.snake = self.snake[:-1]

        new_distance = np.linalg.norm(np.array(new_head) - np.array(self.food))
        reward += (old_distance - new_distance) * 0.5

        return self._get_observation(), reward, self.done, False, {}

    def _get_observation(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        dir_x, dir_y = self.direction

        rel_food_x = food_x - head_x
        rel_food_y = food_y - head_y

        forward_pos = (
            (head_x + dir_x) % self.columns,
            (head_y + dir_y) % self.rows
        )
        left_dir = (-dir_y, dir_x)
        left_pos = (
            (head_x + left_dir[0]) % self.columns,
            (head_y + left_dir[1]) % self.rows
        )
        right_dir = (dir_y, -dir_x)
        right_pos = (
            (head_x + right_dir[0]) % self.columns,
            (head_y + right_dir[1]) % self.rows
        )

        danger_forward = 1.0 if np.any(np.all(self.snake == forward_pos, axis=1)) else 0.0
        danger_left = 1.0 if np.any(np.all(self.snake == left_pos, axis=1)) else 0.0
        danger_right = 1.0 if np.any(np.all(self.snake == right_pos, axis=1)) else 0.0

        return np.array([
            rel_food_x, rel_food_y, dir_x, dir_y,
            danger_forward, danger_left, danger_right,
            self.steps_since_food / 100.0
        ], dtype=np.float32)

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
            time.sleep(0.05)

    def close(self):
        if hasattr(self, 'screen'):
            pygame.quit()
            del self.screen 