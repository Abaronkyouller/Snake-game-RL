import pygame
from snake_env import SnakeGameEnv

# Manually init pygame here
pygame.init()
screen = pygame.display.set_mode((400, 400))
pygame.display.set_caption("Snake RL - Human Play")

env = SnakeGameEnv()
obs = env.reset()

print("Use arrow keys or WASD to control the snake. ESC to quit.")

KEY_ACTION_MAP = {
    pygame.K_UP: 0,
    pygame.K_w: 0,
    pygame.K_DOWN: 1,
    pygame.K_s: 1,
    pygame.K_LEFT: 2,
    pygame.K_a: 2,
    pygame.K_RIGHT: 3,
    pygame.K_d: 3,
}

done = False
action = 3  # Start moving right

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                done = True
            elif event.key in KEY_ACTION_MAP:
                action = KEY_ACTION_MAP[event.key]

    obs, reward, done, _ = env.step(action)
    if done:
        break
    env.render()

env.close()
pygame.quit()