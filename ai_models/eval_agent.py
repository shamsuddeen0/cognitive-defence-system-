import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add the project root and the simulation directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'simulations/virtual_rf_env'))

from simulations.virtual_rf_env.rf_env import RFEnv

def evaluate_agent(model_path, vec_norm_path, jammer_type="reactive"):
    """
    Evaluates the agent against a specific jammer type.
    jammer_type: "random", "frozen", "reactive"
    """
    raw_env = RFEnv()

    # Modify Jammer Logic based on jammer_type
    # Note: Since jammer logic is inside rf_env.py, we'll override the step function
    # or monkey-patch it for evaluation purposes to simulate different jammers.

    def patched_step(action):
        # 1. Apply Action
        prev_idx = raw_env.current_channel_idx
        if action == 1: # Hop Up
            raw_env.current_channel_idx = (raw_env.current_channel_idx + 1) % len(raw_env.channels)
        elif action == 2: # Hop Down
            raw_env.current_channel_idx = (raw_env.current_channel_idx - 1) % len(raw_env.channels)
        elif action == 3: # Random
            raw_env.current_channel_idx = np.random.randint(0, len(raw_env.channels))

        # 2. Jammer Logic (The modified part)
        if jammer_type == "random":
            raw_env.jammer_freqs = [raw_env.channels[np.random.randint(0, len(raw_env.channels))]]
        elif jammer_type == "frozen":
            # Jammer stays on the first channel it picked
            pass
        elif jammer_type == "reactive":
            if np.random.rand() > 0.3:
                raw_env.jammer_freqs = [raw_env.channels[raw_env.current_channel_idx]]

        # 3. Calculate Reward (SINR)
        target_power = 1.0
        interference_power = 1.0 if raw_env.channels[raw_env.current_channel_idx] in raw_env.jammer_freqs else 0.0
        noise_power = raw_env.noise_floor**2

        from simulations.virtual_rf_env.metrics import MetricEngine
        metrics = MetricEngine()
        sinr = metrics.calculate_sinr(target_power, interference_power, noise_power)

        hop_penalty = 0.5 if action != 0 else 0.0
        reward = sinr - hop_penalty

        # Observation
        obs = raw_env._get_obs()

        return obs, reward, False, False, {}

    # Monkey-patch the step method
    raw_env.step = patched_step

    # Wrap env for SB3
    env = DummyVecEnv([lambda: raw_env])
    env = VecNormalize.load(vec_norm_path, env)

    model = PPO.load(model_path)

    obs = env.reset()
    total_rewards = 0
    clear_channels = 0
    ep_length = 100

    for _ in range(ep_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # Calculate if channel is clear (interference == 0)
        # In our environment, obs contains [idx, p1, p2S...].
        # The current channel power is at index 1 + current_channel_idx
        # But we can just check the reward: if reward is high, it's clear.
        # More accurately, check if current_channel is in jammer_freqs.
        if raw_env.channels[raw_env.current_channel_idx] not in raw_env.jammer_freqs:
            clear_channels += 1

        total_rewards += reward[0]

    return total_rewards / ep_length, (clear_channels / ep_length) * 100

def main():
    model_path = os.path.join(os.path.dirname(__file__), "ppo_evasion_model.zip")
    vec_norm_path = os.path.join(os.path.dirname(__file__), "vec_normalize.pkl")

    if not os.path.exists(model_path):
        print("Error: Model not found. Please run evasion_agent.py first.")
        return

    scenarios = ["random", "frozen", "reactive"]
    results_reward = []
    results_clear = []

    print("Evaluating Agent...")
    for s in scenarios:
        avg_rew, clear_pct = evaluate_agent(model_path, vec_norm_path, s)
        results_reward.append(avg_rew)
        results_clear.append(clear_pct)
        print(f"Scenario: {s:10} | Avg Reward: {avg_rew:.2f} | Clear Channel: {clear_pct:.1f}%")

    # Plotting
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(scenarios, results_reward, color='skyblue')
    plt.title("Average Reward per Scenario")
    plt.ylabel("Reward")

    plt.subplot(1, 2, 2)
    plt.bar(scenarios, results_clear, color='salmon')
    plt.title("Clear Channel Percentage")
    plt.ylabel("Percentage (%)")

    plt.tight_layout()
    plt.savefig("evaluation_results.png")
    print("Results saved to evaluation_results.png")

if __name__ == "__main__":
    main()
