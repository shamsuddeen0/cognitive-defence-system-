import pygame
import sys
import os
import numpy as np
import threading
import time
from collections import deque

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'simulations/spatial_env'))
sys.path.append(os.path.join(project_root, 'simulations/virtual_rf_env'))
sys.path.append(os.path.join(project_root, 'simulations/vision_env'))

from simulations.spatial_env.spatial_engine import SpatialDefenseEnv
from simulations.vision_env.consensus_engine import ConsensusEngine
from simulations.virtual_rf_env.rf_env import RFEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# --- COLORS ---
COLOR_BG = (34, 139, 34)      # Forest Green (Ground)
COLOR_DRONE = (0, 200, 255)   # Cyan
COLOR_JAMMER = (255, 50, 50)  # Red
COLOR_BASE = (200, 200, 200)  # Silver
COLOR_TEXT = (255, 255, 255)  # White
COLOR_PANEL = (40, 40, 40, 180) # Semi-transparent gray

class SpatialVisualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1000, 700))
        pygame.display.set_caption("🛡️ CDS - SPATIAL DEFENSE SIMULATION")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Courier New", 18, bold=True)
        self.large_font = pygame.font.SysFont("Courier New", 24, bold=True)

        # Simulation Setup
        self.env = SpatialDefenseEnv(1000, 700)
        self.vision_engine = ConsensusEngine()

        # Load AI Model
        model_path = os.path.join(os.path.dirname(__file__), "ppo_evasion_model.zip")
        vec_norm_path = os.path.join(os.path.dirname(__file__), "vec_normalize.pkl")

        # Use a real RFEnv for VecNormalize loading
        self.rf_dummy_env = DummyVecEnv([lambda: RFEnv()])
        try:
            self.vec_norm = VecNormalize.load(vec_norm_path, self.rf_dummy_env)
        except Exception as e:
            print(f"Normalization load failed: {e}")
            self.vec_norm = None

        self.model = PPO.load(model_path)

        self.running = True
        self.mode = "Day"

    def get_ai_action(self):
        """
        Translates spatial state to the observation format the DRL agent expects.
        Obs: [current_channel_idx, p1, p2, p3, p4, p5]
        """
        # Create a synthetic PSD based on the spatial state
        psd = np.zeros(5)
        # Jammer's presence in the spectrum
        psd[self.env.jammer_channel_idx] = 2.0
        # Drone's presence
        psd[self.env.drone_channel_idx] += 1.0

        obs = np.array([self.env.drone_channel_idx] + psd.tolist(), dtype=np.float32)

        # Normalize if available
        if self.vec_norm:
            # Note: VecNormalize expects a batch
            obs = self.vec_norm.normalize_obs(obs[None, :])[0]

        action, _ = self.model.predict(obs, deterministic=True)
        return action[0] if isinstance(action, np.ndarray) and action.ndim > 0 else action

    def draw_text(self, text, x, y, color=COLOR_TEXT, font=None):
        if font is None: font = self.font
        img = font.render(text, True, color)
        self.screen.blit(img, (x, y))

    def run(self):
        frame = 0
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_n: # Toggle Night/Day
                        self.mode = "Night" if self.mode == "Day" else "Day"
                        self.env.mode = self.mode
                        self.vision_engine.set_mode(self.mode)

            # 1. Update Simulation
            # Drone flies in a slow circle
            angle = time.time() * 0.5
            target_x = 500 + 200 * np.cos(angle)
            target_y = 350 + 150 * np.sin(angle)
            self.env.move_drone(target_x, target_y)
            self.env.move_jammer()

            # AI RF Logic
            action = self.get_ai_action()
            sinr = self.env.step_rf_logic(action)

            # Vision Logic
            detections = self.env.get_vision_confidence()
            frame_dec, frame_conf = self.vision_engine.evaluate_frame(detections)
            final_dec, temp_conf = self.vision_engine.update_consensus(frame_dec)

            # 2. Rendering
            # Background
            self.screen.fill(COLOR_BG if self.mode == "Day" else (20, 40, 80))

            # Base Station
            pygame.draw.circle(self.screen, COLOR_BASE, self.env.base_station.astype(int), 15)
            self.draw_text("C2 BASE", self.env.base_station[0]-30, self.env.base_station[1]+20)

            # Jammer
            pygame.draw.rect(self.screen, COLOR_JAMMER, (*self.env.jammer_pos.astype(int), 20, 20))
            self.draw_text("JAMMER", self.env.jammer_pos[0]-30, self.env.jammer_pos[1]-25, COLOR_JAMMER)

            # Drone
            pygame.draw.circle(self.screen, COLOR_DRONE, self.env.drone_pos.astype(int), 10)
            self.draw_text("SENSING DRONE", self.env.drone_pos[0]-50, self.env.drone_pos[1]-25, COLOR_DRONE)

            # Draw signal line (Base <-> Drone)
            line_color = (0, 255, 0) if sinr > 10 else (255, 0, 0)
            pygame.draw.line(self.screen, line_color, self.env.base_station.astype(int), self.env.drone_pos.astype(int), 2)

            # 3. UI HUD
            # HUD Panel
            hud_rect = pygame.Rect(10, 10, 320, 240)
            surf = pygame.Surface((320, 240), pygame.SRCALPHA)
            surf.fill((40, 40, 40, 180))
            self.screen.blit(surf, (10, 10))

            self.draw_text("COGNITIVE DEFENSE HUD", 20, 20, (255, 215, 0), self.large_font)

            # RF Stats
            self.draw_text(f"Current Channel: {self.env.channels[self.env.drone_channel_idx]} MHz", 20, 60)
            self.draw_text(f"SINR: {sinr:.2f} dB", 20, 80, "green" if sinr > 10 else "red")
            self.draw_text(f"Link: {'STABLE' if sinr > 10 else 'JAMMED'}", 20, 100, "green" if sinr > 10 else "red")

            # Vision Stats
            self.draw_text("Vision Consensus:", 20, 130)
            y_off = 150
            for sensor, conf in detections.items():
                self.draw_text(f"{sensor}: {conf:.2f}", 30, y_off)
                pygame.draw.rect(self.screen, (100, 100, 100), (120, y_off+5, 100, 10))
                pygame.draw.rect(self.screen, (0, 255, 0), (120, y_off+5, int(conf*100), 10))
                y_off += 20

            vis_status = "SPOOF DETECTED" if not final_dec and any(detections.values()) else "TARGET LOCKED" if final_dec else "SEARCHING..."
            vis_color = (0, 255, 0) if final_dec else (255, 0, 0)
            self.draw_text(f"Status: {vis_status}", 20, y_off + 10, vis_color)

            self.draw_text(f"Mode: {self.mode} (Press N to toggle)", 20, 210, (200, 200, 200))

            pygame.display.flip()
            self.clock.tick(15)

        pygame.quit()

if __name__ == "__main__":
    sim = SpatialVisualizer()
    sim.run()
