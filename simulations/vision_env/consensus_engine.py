import numpy as np
from collections import deque

class ConsensusEngine:
    """
    Advanced Multimodal Consensus Engine.
    Uses confidence-weighted sums and temporal filtering to reduce false positives.
    """
    def __init__(self, history_size=5, confirm_threshold=3):
        # Base weights for different modalities
        self.weights = {
            "RGB": 0.3,
            "IR": 0.4,
            "RF": 0.3
        }
        self.decision_threshold = 0.5  # Confidence threshold for a single frame
        self.history = deque(maxlen=history_size)
        self.confirm_threshold = confirm_threshold # Needs m detections in n frames

    def set_mode(self, mode):
        """Adjusts weights based on environment (Day vs Night)."""
        if mode == "Night":
            self.weights = {"RGB": 0.1, "IR": 0.7, "RF": 0.2}
        else:
            self.weights = {"RGB": 0.3, "IR": 0.4, "RF": 0.3}

    def evaluate_frame(self, detections):
        """
        Evaluates a single frame of confidence scores.
        detections: {sensor: float}
        Returns: (frame_decision, weighted_confidence)
        """
        weighted_sum = 0.0
        for sensor, conf in detections.items():
            weighted_sum += conf * self.weights.get(sensor, 0)

        # A frame is considered "Positive" if the weighted sum exceeds the threshold
        frame_decision = weighted_sum >= self.decision_threshold
        return frame_decision, weighted_sum

    def update_consensus(self, frame_decision):
        """
        Updates the temporal buffer and returns the final confirmed decision.
        """
        self.history.append(frame_decision)

        # Final decision: Target is confirmed if at least 'confirm_threshold'
        # frames in the history are positive.
        positive_count = sum(self.history)
        final_decision = positive_count >= self.confirm_threshold

        return final_decision, positive_count / len(self.history)

if __name__ == "__main__":
    engine = ConsensusEngine()
    # Test: Sequence of detections
    test_sequence = [True, False, True, True, True]
    for d in test_sequence:
        dec, conf = engine.update_consensus(d)
        print(f"Frame: {d} | Final Decision: {dec} | History Conf: {conf:.2f}")
