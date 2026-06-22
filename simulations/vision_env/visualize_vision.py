import os
import sys
import time

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from simulations.vision_env.vision_sim import VisionSim
from simulations.vision_env.consensus_engine import ConsensusEngine

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_conf_bar(conf, width=20):
    """Returns a visual bar representing confidence."""
    filled = int(conf * width)
    bar = "█" * filled + "░" * (width - filled)
    color = "\033[92m" if conf > 0.7 else "\033[93m" if conf > 0.4 else "\033[91m"
    return f"{color}{bar}\033[0m {conf:.2f}"

def main():
    sim = VisionSim()
    engine = ConsensusEngine()

    print("\n" + "*"*60)
    print(" SETUP COMPLETE. Starting Robust Vision Consensus Demo...")
    print(" Press Ctrl+C to stop.")
    print("*"*60)
    time.sleep(2)

    try:
        frame_count = 0
        while True:
            frame_count += 1

            # Periodically toggle target and mode for demo purposes
            if frame_count % 20 == 0:
                sim.toggle_target()
            if frame_count % 40 == 0:
                mode = sim.toggle_mode()
                engine.set_mode(mode)

            # 1. Get Confidence Scores
            detections = sim.get_detections()

            # 2. Evaluate Frame
            frame_decision, frame_conf = engine.evaluate_frame(detections)

            # 3. Update Temporal Consensus
            final_decision, temporal_conf = engine.update_consensus(frame_decision)

            clear_screen()
            print("\n" + "="*60)
            print(f" 🛡️ COGNITIVE DEFENSE SYSTEM - VISION MONITOR | Mode: {sim.mode}")
            print("="*60)

            print("\nSENSOR CONFIDENCE:")
            print("-" * 60)
            for sensor, conf in detections.items():
                print(f"  {sensor:<10} : {get_conf_bar(conf)}")

            print("-" * 60)

            # Final status display
            status = "\033[92mTARGET CONFIRMED ✅\033[0m" if final_decision else "\033[91mNO TARGET / SPOOF ❌\033[0m"

            # Progress bar for temporal confirmation (m-of-n)
            prog_width = 20
            prog_filled = int(temporal_conf * prog_width)
            prog_bar = "■" * prog_filled + "□" * (prog_width - prog_filled)

            print(f"\nCONSENSUS DECISION: {status}")
            print(f"Frame Confidence:    {frame_conf:.2f}")
            print(f"Confirmation Progress: [{prog_bar}] {temporal_conf:.2%}")

            print("\n" + "="*60)
            print(f"Log: { 'Target locked. Maintaining track...' if final_decision else 'Filtering noise. Searching for stable target...'}")

            time.sleep(0.6)

    except KeyboardInterrupt:
        print("\nDemo stopped by operator. Exiting...")

if __name__ == "__main__":
    main()
