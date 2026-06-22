import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add the project root and the simulation directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'simulations/virtual_rf_env'))

from simulations.virtual_rf_env.rf_env import RFEnv

def clear_screen():
    """Clears the terminal screen for a smooth animation effect."""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_status_color(sinr):
    """Returns a colored status string based on SINR."""
    if sinr > 10:
        return "\033[92mCLEAR ✅\033[0m"    # Green
    elif sinr > 0:
        return "\033[93mINTERFERENCE ⚠️\033[0m" # Yellow
    else:
        return "\033[91mJAMMED ❌\033[0m"   # Red

def render_spectrum(env, agent_idx, jammer_freqs, channels):
    """Prints a visual map of the RF spectrum to the console."""
    print("\n" + "="*60)
    print(" 📡 COGNITIVE DEFENSE SYSTEM - LIVE RF MONITOR")
    print("="*60)
    print("\nRF SPECTRUM VIEW")
    print("-" * 60)

    # Header: Channel Frequencies
    header = "  "
    for ch in channels:
        header += f"[{ch:^5}] "
    print(header)

    # Row 1: Agents and Jammers
    row = "  "
    for i, ch in enumerate(channels):
        if i == agent_idx:
            # Agent is here
            symbol = " A "
            if ch in jammer_freqs:
                symbol = "\033[91m[A+J]\033[0m" # Both are here
            else:
                symbol = "\033[94m[ A ]\033[0m" # Only agent
        elif ch in jammer_freqs:
            symbol = "\033[91m[ J ]\033[0m" # Only jammer
        else:
            symbol = "[ . ]" # Clear
        row += f"{symbol} "
    print(row)
    print("-" * 60)

def main():
    model_path = os.path.join(os.path.dirname(__file__), "ppo_evasion_model.zip")
    vec_norm_path = os.path.join(os.path.dirname(__file__), "vec_normalize.pkl")

    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found. Please wait for training to complete.")
        return

    # 1. Setup Environment
    env_raw = RFEnv()
    channels = env_raw.channels

    # Wrap for SB3
    env = DummyVecEnv([lambda: env_raw])
    try:
        env = VecNormalize.load(vec_norm_path, env)
        print("Loaded normalization statistics successfully.")
    except:
        print("Warning: Normalization file not found. Results may be inaccurate.")

    # Load the trained brain
    model = PPO.load(model_path)

    print("\n" + "*"*60)
    print(" SETUP COMPLETE. Starting Real-Time Simulation...")
    print(" Press Ctrl+C to stop the demo.")
    print("*"*60)
    time.sleep(2)

    obs = env.reset()

    try:
        while True:
            # AI decides the next move
            action, _ = model.predict(obs, deterministic=True)

            # Step the environment
            obs, reward, done, info = env.step(action)

            # Correctly getting the raw environment state

            # Correctly getting the raw environment state
            # In DummyVecEnv, we can access the internal envs list
            actual_env = env.envs[0].unwrapped

            agent_idx = actual_env.current_channel_idx
            jammer_freqs = actual_env.jammer_freqs

            # Calculate SINR for the display
            target_power = 1.0
            interference_power = 1.0 if actual_env.channels[agent_idx] in jammer_freqs else 0.0
            noise_power = actual_env.noise_floor**2
            from simulations.virtual_rf_env.metrics import MetricEngine
            sinr = MetricEngine().calculate_sinr(target_power, interference_power, noise_power)

            # Render
            clear_screen()
            render_spectrum(actual_env, agent_idx, jammer_freqs, channels)

            # Metrics row
            status = get_status_color(sinr)
            action_name = {0: "STAY", 1: "HOP UP", 2: "HOP DOWN", 3: "RANDOM"}[action[0]]

            print(f"\nMETRICS:")
            print(f"  Current SINR: {sinr:.2f} dB")
            print(f"  Link Status:  {status}")
            print(f"  AI Action:    {action_name}")
            print(f"\nLog: Agent is currently analyzing the PSD and optimizing for throughput...")

            time.sleep(0.8) # Slow down for the audience to see the "hops"

    except KeyboardInterrupt:
        print("\n\nDemo stopped by operator. Exiting...")

if __name__ == "__main__":
    main()
