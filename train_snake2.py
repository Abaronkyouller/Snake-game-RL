from stable_baselines3 import DQN
from snake_env import SnakeGameEnv
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import NatureCNN
def make_env():
    return SnakeGameEnv()
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = 10

    env = SubprocVecEnv([make_env for _ in range(num_envs)])

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=5e-4,
        buffer_size=100_000,
        batch_size=64,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        device=device
    )
    model.learn(total_timesteps=1_000_000)
    model.save("snake_dqn")

    # Evaluation (single env, not parallel)
    test_env = SnakeGameEnv()
    obs,_ = test_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = test_env.step(action)
        # test_env.render()
    test_env.close()