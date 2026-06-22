import numpy as np
import random
import time

class VisionSim:
    """
    Simulates a multimodal sensing environment for drone detection.
    Produces confidence scores (0.0 to 1.0) from RGB, IR, and RF.
    """
    def __init__(self):
        # Base confidence for true positives
        self.true_positive_range = {
            "RGB": (0.6, 0.95),
            "IR": (0.5, 0.9),
            "RF": (0.4, 0.8)
        }
        # Base confidence for false positives (spoofing/noise)
        self.false_positive_range = {
            "RGB": (0.0, 0.3),
            "IR": (0.0, 0.2),
            "RF": (0.0, 0.4)
        }
        self.target_present = False
        self.mode = "Day" # "Day" or "Night"

    def toggle_target(self):
        self.target_present = not self.target_present
        return self.target_present

    def toggle_mode(self):
        self.mode = "Night" if self.mode == "Day" else "Day"
        return self.mode

    def get_detections(self):
        """
        Simulates one frame of sensing.
        Returns a dictionary of confidence scores: {sensor: float}
        """
        detections = {}
        for sensor, range_val in self.true_positive_range.items():
            if self.target_present:
                # Generate a confidence score for a real target
                conf = random.uniform(*range_val)

                # Environmental penalty: RGB is nearly blind at night
                if self.mode == "Night" and sensor == "RGB":
                    conf *= 0.2
            else:
                # Generate a noise/spoof score
                conf = random.uniform(*self.false_positive_range[sensor])

            detections[sensor] = conf

        return detections

if __name__ == "__main__":
    sim = VisionSim()
    print("Simulating Vision Environment (Confidence Mode)...")
    for i in range(5):
        sim.toggle_target()
        print(f"Target: {sim.target_present} | Mode: {sim.mode} | Conf: {sim.get_detections()}")
        time.sleep(0.2)
