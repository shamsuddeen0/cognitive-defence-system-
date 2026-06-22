import tkinter as tk
from tkinter import ttk, scrolledtext
import os
import sys
import threading
import time
import numpy as np
from collections import deque

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'simulations/virtual_rf_env'))

# Imports from our modules
from simulations.virtual_rf_env.rf_env import RFEnv
from simulations.virtual_rf_env.metrics import MetricEngine
from simulations.vision_env.vision_sim import VisionSim
from simulations.vision_env.consensus_engine import ConsensusEngine
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

class DefenseDashboard(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("🛡️ COGNITIVE DEFENSE SYSTEM - COMMAND CENTER")
        self.geometry("1000x700")
        self.configure(bg="#1a1a1a")

        # --- STATE INITIALIZATION ---
        self.setup_rf_system()
        self.setup_vision_system()

        # UI Style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TLabel", background="#1a1a1a", foreground="white", font=("Courier New", 12))
        self.style.configure("Header.TLabel", font=("Courier New", 16, "bold"))
        self.style.configure("TProgressbar", thickness=20)

        # --- LAYOUT ---
        self.create_widgets()

        # Start the simulation loop in a background thread
        self.running = True
        self.sim_thread = threading.Thread(target=self.simulation_loop, daemon=True)
        self.sim_thread.start()

    def setup_rf_system(self):
        """Initializes the DRL Radio system."""
        self.rf_env_raw = RFEnv()
        self.channels = self.rf_env_raw.channels

        # Load Model & Normalization
        model_path = os.path.join(os.path.dirname(__file__), "ppo_evasion_model.zip")
        vec_norm_path = os.path.join(os.path.dirname(__file__), "vec_normalize.pkl")

        self.rf_env = DummyVecEnv([lambda: self.rf_env_raw])
        try:
            self.rf_env = VecNormalize.load(vec_norm_path, self.rf_env)
        except Exception as e:
            print(f"Normalization load failed: {e}")

        self.model = PPO.load(model_path)
        self.metric_engine = MetricEngine()

    def setup_vision_system(self):
        """Initializes the Multimodal Vision system."""
        self.vision_sim = VisionSim()
        self.vision_engine = ConsensusEngine()

    def create_widgets(self):
        # Main Container
        main_frame = tk.Frame(self, bg="#1a1a1a", padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_lbl = ttk.Label(main_frame, text="🛡️ COGNITIVE DEFENSE SYSTEM - INTEGRATED MONITOR", style="Header.TLabel")
        title_lbl.pack(pady=(0, 20))

        # Two-column layout
        content_frame = tk.Frame(main_frame, bg="#1a1a1a")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # --- LEFT PANEL: RF DEFENSE ---
        rf_panel = tk.LabelFrame(content_frame, text=" RF ANTI-JAMMING ", fg="white", bg="#1a1a1a", font=("Courier New", 12, "bold"), padx=10, pady=10)
        rf_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        # Spectrum Canvas
        self.canvas = tk.Canvas(rf_panel, width=400, height=150, bg="black", highlightthickness=0)
        self.canvas.pack(pady=10)

        # RF Metrics
        self.sinr_lbl = ttk.Label(rf_panel, text="SINR: 0.00 dB")
        self.sinr_lbl.pack(anchor="w")

        self.rf_status_lbl = ttk.Label(rf_panel, text="Link Status: IDLE", foreground="white")
        self.rf_status_lbl.pack(anchor="w")

        self.rf_action_lbl = ttk.Label(rf_panel, text="AI Action: None")
        self.rf_action_lbl.pack(anchor="w")

        # --- RIGHT PANEL: VISION DEFENSE ---
        vis_panel = tk.LabelFrame(content_frame, text=" MULTIMODAL VISION CONSENSUS ", fg="white", bg="#1a1a1a", font=("Courier New", 12, "bold"), padx=10, pady=10)
        vis_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        self.mode_lbl = ttk.Label(vis_panel, text="Env Mode: Day")
        self.mode_lbl.pack(anchor="w")

        # Confidence Bars
        self.rgb_bar = ttk.Progressbar(vis_panel, length=200, mode='determinate')
        self.rgb_bar.pack(pady=5, anchor="w")
        ttk.Label(vis_panel, text="RGB Confidence").pack(anchor="w")

        self.ir_bar = ttk.Progressbar(vis_panel, length=200, mode='determinate')
        self.ir_bar.pack(pady=5, anchor="w")
        ttk.Label(vis_panel, text="IR Confidence").pack(anchor="w")

        self.rf_vis_bar = ttk.Progressbar(vis_panel, length=200, mode='determinate')
        self.rf_vis_bar.pack(pady=5, anchor="w")
        ttk.Label(vis_panel, text="RF Detection").pack(anchor="w")

        # Consensus Result
        self.vis_status_lbl = ttk.Label(vis_panel, text="SENSING...", font=("Courier New", 14, "bold"))
        self.vis_status_lbl.pack(pady=20)

        # Temporal Progress
        self.temp_bar = ttk.Progressbar(vis_panel, length=200, mode='determinate')
        self.temp_bar.pack(pady=5, anchor="w")
        ttk.Label(vis_panel, text="Confirmation Progress").pack(anchor="w")

        # --- BOTTOM PANEL: LOGS ---
        self.log_area = scrolledtext.ScrolledText(main_frame, height=8, bg="#0a0a0a", fg="#00ff00", font=("Courier New", 10))
        self.log_area.pack(fill=tk.X, pady=20)
        self.log_area.insert(tk.END, "[SYSTEM] Cognitive Defense System Online...\n")
        self.log_area.insert(tk.END, "[INFO] Monitoring RF Spectrum and Vision Modalities...\n")

    def log(self, message):
        self.log_area.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.log_area.see(tk.END)

    def update_rf_visuals(self, agent_idx, jammer_freqs):
        self.canvas.delete("all")
        width = 400
        height = 150
        margin = 40

        # Draw frequency axis
        self.canvas.create_line(margin, 100, width-margin, 100, fill="white")
        for i, freq in enumerate(self.channels):
            x = margin + (i * (width - 2*margin) / (len(self.channels)-1))
            self.canvas.create_text(x, 115, text=str(freq), fill="gray", font=("Courier New", 8))
            self.canvas.create_line(x, 95, x, 105, fill="white")

        # Draw Jammer
        for f in jammer_freqs:
            try:
                j_idx = self.channels.index(f)
                jx = margin + (j_idx * (width - 2*margin) / (len(self.channels)-1))
                self.canvas.create_oval(jx-8, 60-8, jx+8, 60+8, fill="red", outline="white")
                self.canvas.create_text(jx, 50, text="JAMMER", fill="red", font=("Courier New", 8, "bold"))
            except ValueError: pass

        # Draw Agent
        ax = margin + (agent_idx * (width - 2*margin) / (len(self.channels)-1))
        self.canvas.create_oval(ax-8, 80-8, ax+8, 80+8, fill="blue", outline="white")
        self.canvas.create_text(ax, 90, text="AGENT", fill="cyan", font=("Courier New", 8, "bold"))

    def simulation_loop(self):
        frame_count = 0
        while self.running:
            frame_count += 1

            # --- RF Update ---
            obs = self.rf_env.reset() if frame_count == 1 else self.last_obs
            action, _ = self.model.predict(obs, deterministic=True)
            self.last_obs, reward, done, info = self.rf_env.step(action)

            actual_env = self.rf_env.envs[0].unwrapped
            agent_idx = actual_env.current_channel_idx
            jammer_freqs = actual_env.jammer_freqs

            target_power = 1.0
            interference_power = 1.0 if actual_env.channels[agent_idx] in jammer_freqs else 0.0
            noise_power = actual_env.noise_floor**2
            sinr = self.metric_engine.calculate_sinr(target_power, interference_power, noise_power)

            action_map = {0: "STAY", 1: "HOP UP", 2: "HOP DOWN", 3: "RANDOM"}
            action_name = action_map[action[0]]

            # --- Vision Update ---
            if frame_count % 20 == 0:
                self.vision_sim.toggle_target()
            if frame_count % 40 == 0:
                mode = self.vision_sim.toggle_mode()
                self.vision_engine.set_mode(mode)

            detections = self.vision_sim.get_detections()
            frame_decision, frame_conf = self.vision_engine.evaluate_frame(detections)
            final_decision, temporal_conf = self.vision_engine.update_consensus(frame_decision)

            # --- GUI Update (Main Thread) ---
            self.after(0, self.update_gui,
                       agent_idx, jammer_freqs, sinr, action_name,
                       detections, final_decision, temporal_conf, self.vision_sim.mode)

            time.sleep(0.8)

    def update_gui(self, agent_idx, jammer_freqs, sinr, action_name, detections, final_decision, temporal_conf, mode):
        # Update RF
        self.update_rf_visuals(agent_idx, jammer_freqs)
        self.sinr_lbl.config(text=f"SINR: {sinr:.2f} dB")

        status = "CLEAR ✅" if sinr > 10 else "JAMMED ❌"
        color = "green" if sinr > 10 else "red"
        self.rf_status_lbl.config(text=f"Link Status: {status}", foreground=color)
        self.rf_action_lbl.config(text=f"AI Action: {action_name}")

        # Update Vision
        self.mode_lbl.config(text=f"Env Mode: {mode}")
        self.rgb_bar['value'] = detections['RGB'] * 100
        self.ir_bar['value'] = detections['IR'] * 100
        self.rf_vis_bar['value'] = detections['RF'] * 100

        vis_status = "TARGET CONFIRMED ✅" if final_decision else "NO TARGET / SPOOF ❌"
        vis_color = "green" if final_decision else "red"
        self.vis_status_lbl.config(text=vis_status, foreground=vis_color)
        self.temp_bar['value'] = temporal_conf * 100

        # Log RF actions occasionally
        if "JAMMED" in status:
            self.log(f"Electronic Interference Detected. AI performing {action_name}...")
        elif "CONFIRMED" in vis_status:
            self.log("Vision Consensus: Hostile Drone detected and locked.")

if __name__ == "__main__":
    app = DefenseDashboard()
    app.mainloop()
