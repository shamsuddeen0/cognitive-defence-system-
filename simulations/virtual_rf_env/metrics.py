import numpy as np

class MetricEngine:
    """Calculates performance metrics for the RF environment."""

    @staticmethod
    def calculate_power(signal):
        """Returns the average power of a signal."""
        return np.mean(signal**2)

    @staticmethod
    def calculate_sinr(signal_power, interference_power, noise_power):
        """Calculates SINR in decibels."""
        # Avoid division by zero
        denom = interference_power + noise_power
        if denom == 0:
            return float('inf')

        sinr_linear = signal_power / denom
        return 10 * np.log10(sinr_linear)

    @staticmethod
    def get_psd(signal, fs):
        """Computes a simple Power Spectral Density using FFT."""
        n = len(signal)
        fft_res = np.fft.fft(signal)
        psd = (np.abs(fft_res)**2) / (fs * n)
        freqs = np.fft.fftfreq(n, 1/fs)

        # Return positive frequencies only
        pos_mask = freqs >= 0
        return freqs[pos_mask], psd[pos_mask]
