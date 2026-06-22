import numpy as np

class TargetSignal:
    """Generates a modulated communication signal."""
    def __init__(self, fs=1000):
        self.fs = fs

    def generate(self, freq, duration, amplitude=1.0):
        t = np.arange(0, duration, 1/self.fs)
        return amplitude * np.sin(2 * np.pi * freq * t)

class JammerEngine:
    """Generates various types of interference."""
    def __init__(self, fs=1000):
        self.fs = fs

    def barrage(self, freqs, duration, amplitude=1.0):
        """Wide-band interference."""
        t = np.arange(0, duration, 1/self.fs)
        signal = np.zeros_like(t)
        for f in freqs:
            signal += amplitude * np.sin(2 * np.pi * f * t)
        return signal

    def sweep(self, start_f, end_f, duration, amplitude=1.0):
        """Narrow-band interference that sweeps across the spectrum."""
        t = np.arange(0, duration, 1/self.fs)
        # Linear frequency modulation (chirp)
        signal = amplitude * np.sin(2 * np.pi * (start_f + (end_f - start_f) * t / (2 * duration)) * t)
        return signal

    def reactive(self, target_freq, duration, amplitude=1.0, delay=0.01):
        """Sense-and-respond interference."""
        # Simple reactive: just jams the target freq after a small delay
        t = np.arange(0, duration, 1/self.fs)
        signal = np.zeros_like(t)
        # Simulate detection delay
        delay_samples = int(delay * self.fs)
        if delay_samples < len(t):
            signal[delay_samples:] = amplitude * np.sin(2 * np.pi * target_freq * t[delay_samples:])
        return signal
