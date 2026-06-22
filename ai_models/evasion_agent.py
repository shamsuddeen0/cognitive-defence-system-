import os
import sys
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add the project root and the simulation directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'simulations/virtual_rf_env'))

from simulations.virtual_rf_env.rf_env import RFEnv

class RFEnvWrapper(gym.ObservationWrapper):
    """
    Wrapper for RFEnv to normalize observations.
    Current Obs: [current_channel_idx, p1, p2, ..., pN]
    PSD values (pX) are typically 0.0, 1.0, or 2.0.
    Normalization helps the neural network converge faster.
    """
    def __init__(self, env):
        super(RFEnvWrapper, self).__init__(env)
        # The first element is the index (0 to N-1), others are PSD.
        # We normalize the index by the number of channels and PSD by 2.0.
        self.num_channels = len(env.channels)

    def observation(self, obs):
        # Normalize current_channel_idx
        norm_obs = np.copy(obs)
        norm_obs[0] = norm_obs[0] / self.num_channels
        # Normalize PSD values (usually 0, 1, or 2)
        norm_obs[1:] = norm_obs[1:] / 2.0
        return norm_obs.astype(np.float32)


# Remove the useless java_numpy_array function


def train():
    # 1. Setup Environment
    print("Initializing RF Environment...")
    env = RFEnv()
    env = RFEnvWrapper(env)

    # Wrap with Monitor for logging and DummyVecEnv for SB3 compatibility
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # Normalize observations and rewards
    env = VecNormalize(env, gamma=0.99, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 2. Configure PPO Agent
    print("Configuring PPO Agent...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        tensorboard_log="./tb_logs/"
    )

    # 3. Train
    total_timesteps = 500_000
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    # 4. Save Model
    model_path = "ppo_evasion_model"
    model.save(model_path)

    # Save the VecNormalize statistics so we can use them during evaluation
    env.save("vec_normalize.pkl")

    print(f"Training complete. Model saved to {model_path}")

if __name__ == "__main__":
    train()
