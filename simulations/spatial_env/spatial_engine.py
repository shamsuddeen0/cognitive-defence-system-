import numpy as np
import random
import math

class SpatialDefenseEnv:
    """
    Spatially-aware simulation for RF Jamming and Vision Detection.
    Tracks Drone, Jammer, and Base Station positions.
    """
    def __init__(self, screen_width=800, screen_height=600):
        self.width = screen_width
        self.height = screen_height

        # Entity Positions
        self.base_station = np.array([50.0, 50.0])
        self.jammer_pos = np.array([random.uniform(200, 600), random.uniform(200, 400)])
        self.drone_pos = np.array([100.0, 100.0])

        # RF Params
        self.channels = [100, 200, 300, 400, 500]
        self.drone_channel_idx = 0
        self.jammer_channel_idx = 0
        self.noise_floor = 0.01

        # Vision Params
        self.target_present = True
        self.mode = "Day"

    def move_drone(self, target_x, target_y, speed=2.0):
        """Moves drone towards a target point."""
        direction = np.array([target_x, target_y]) - self.drone_pos
        dist = np.linalg.norm(direction)
        if dist > 0:
            self.drone_pos += (direction / dist) * speed

    def move_jammer(self):
        """Randomly wanders the jammer on the ground."""
        self.jammer_pos += np.random.uniform(-1, 1, 2)
        self.jammer_pos = np.clip(self.jammer_pos, [0, 0], [self.width, self.height])

    def calculate_spatial_sinr(self):
        """
        Calculates SINR based on the distance between drone, base station, and jammer.
        SINR = P_signal / (P_jammer + P_noise)
        """
        # Distances
        dist_base = np.linalg.norm(self.drone_pos - self.base_station)
        dist_jammer = np.linalg.norm(self.drone_pos - self.jammer_pos)

        # Path Loss Model (Simplified)
        p_signal = 1.0 / (max(dist_base, 1)**2)

        p_jammer = 0.0
        if self.drone_channel_idx == self.jammer_channel_idx:
            # Jammer power is inversely proportional to distance from the drone
            p_jammer = 10.0 / (max(dist_jammer, 1)**2)

        sinr = p_signal / (p_jammer + self.noise_floor)
        # Convert to dB
        return 10 * np.log10(sinr) if sinr > 0 else -100.0

    def get_vision_confidence(self):
        """Returns confidence scores based on distance and environment."""
        dist_to_jammer = np.linalg.norm(self.drone_pos - self.jammer_pos)

        # Base confidencees
        rgb = 0.8 if self.target_present else 0.1
        ir = 0.7 if self.target_present else 0.05
        rf = 0.6 if self.target_present else 0.2

        # Distance decay: as drone flies away, confidence drops
        decay = 1.0 / (1.0 + 0.001 * dist_to_jammer)

        # Night mode penalty for RGB
        if self.mode == "Night":
            rgb *= 0.2

        return {
            "RGB": rgb * decay,
            "IR": ir * decay,
            "RF": rf * decay
        }

    def step_rf_logic(self, action):
        """Updates RF channel based on DRL action."""
        if action == 1: # Hop Up
            self.drone_channel_idx = (self.drone_channel_idx + 1) % len(self.channels)
        elif action == 2: # Hop Down
            self.drone_channel_idx = (self.drone_channel_idx - 1) % len(self.channels)
        elif action == 3: # Random
            self.drone_channel_idx = np.random.randint(0, len(self.channels))

        # Reactive Jammer logic: Jammer tries to follow the drone's channel
        if random.random() > 0.3:
            self.jammer_channel_idx = self.drone_channel_idx

        return self.calculate_spatial_sinr()

if __name__ == "__main__":
    env = SpatialDefenseEnv()
    print(f"Initial SINR: {env.calculate_spatial_sinr():.2f} dB")
    env.move_drone(400, 300)
    print(f"After moving: {env.calculate_spatial_sinr():.2f} dB")
