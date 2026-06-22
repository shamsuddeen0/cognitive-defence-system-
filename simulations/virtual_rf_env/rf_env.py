import numpy as np
import gymnasium as gym
from gymnasium import spaces
from signals import TargetSignal, JammerEngine
from metrics import MetricEngine

class RFEnv(gym.Env):
    """
    A Gymnasium environment simulating a Cognitive Radio's interaction
    with a jammer in an RF spectrum.
    """
    def __init__(self, fs=1000, channels=[100, 200, 300, 400, 500], noise_floor=0.1):
        super(RFEnv, self).__init__()

        self.fs = fs
        self.channels = channels
        self.noise_floor = noise_floor

        # Signal and Jammer components
        self.target_gen = TargetSignal(fs)
        self.jammer_eng = JammerEngine(fs)
        self.metrics = MetricEngine()

        # State: Current frequency index and a simplified PSD (power per channel)
        # Observation space: [current_freq_idx, power_ch1, power_ch2, ...]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1 + len(channels),), dtype=np.float32
        )

        # Action space: discrete hops
        # 0: Stay, 1: Hop Up, 2: Hop Down, 3: Jump to Random
        self.action_space = spaces.Discrete(4)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomize starting channel
        self.current_channel_idx = np.random.randint(0, len(self.channels))

        # Initialize Jammer (Simple Barrage for the start)
        self.jammer_freqs = [self.channels[np.random.randint(0, len(self.channels))]]

        observation = self._get_obs()
        return observation, {}

    def _get_obs(self):
        # Simplified PSD: Calculate power in each channel
        psd_obs = []
        for ch in self.channels:
            # Simulate signal capture at this frequency
            # In a real simulation, we'd use a filter; here we just check if jammer/target are there
            p = 0.0
            if ch in self.jammer_freqs:
                p += 1.0 # Jammer power
            if ch == self.channels[self.current_channel_idx]:
                p += 1.0 # Target power
            psd_obs.append(p)

        return np.array([self.current_channel_idx] + psd_obs, dtype=np.float32)

    def step(self, action):
        # 1. Apply Action
        prev_idx = self.current_channel_idx
        if action == 1: # Hop Up
            self.current_channel_idx = (self.current_channel_idx + 1) % len(self.channels)
        elif action == 2: # Hop Down
            self.current_channel_idx = (self.current_channel_idx - 1) % len(self.channels)
        elif action == 3: # Random
            self.current_channel_idx = np.random.randint(0, len(self.channels))
        # action 0: Stay

        # 2. Jammer Logic (Simulating a Reactive Jammer)
        # If target moved, jammer follows with some probability
        if np.random.rand() > 0.3:
            self.jammer_freqs = [self.channels[self.current_channel_idx]]

        # 3. Calculate Reward (SINR)
        target_power = 1.0
        # Interference is high if target and jammer are on the same channel
        interference_power = 1.0 if self.channels[self.current_channel_idx] in self.jammer_freqs else 0.0
        noise_power = self.noise_floor**2

        sinr = self.metrics.calculate_sinr(target_power, interference_power, noise_power)

        # Reward = SINR - penalty for hopping
        hop_penalty = 0.5 if action != 0 else 0.0
        reward = sinr - hop_penalty

        # 4. Terminal state (optional)
        terminated = False
        truncated = False

        observation = self._get_obs()

        return observation, reward, terminated, truncated, {}
