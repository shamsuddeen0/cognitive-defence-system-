import numpy as np
import matplotlib.pyplot as plt

def generate_signal(freq, fs, duration, amplitude=1.0):
    """Generates a simple sine wave as a target signal."""
    t = np.arange(0, duration, 1/fs)
    return amplitude * np.sin(2 * np.pi * freq * t)

def generate_jammer(freqs, fs, duration, amplitude=1.0):
    """Generates a jammer signal (Barrage) covering multiple frequencies."""
    t = np.arange(0, duration, 1/fs)
    jammer_signal = np.zeros_like(t)
    for f in freqs:
        jammer_signal += amplitude * np.sin(2 * np.pi * f * t)
    return jammer_signal

def calculate_sinr(signal_power, noise_power, interference_power):
    """Calculates Signal-to-Interference-plus-Noise Ratio in dB."""
    sinr_linear = signal_power / (interference_power + noise_power)
    return 10 * np.log10(sinr_linear)

def main():
    # Simulation Parameters
    fs = 1000  # Sampling frequency (Hz)
    duration = 1.0  # Seconds
    noise_floor = 0.1  # Noise amplitude

    # Target Signal
    target_freq = 100  # Hz
    target_amp = 1.0
    signal = generate_signal(target_freq, fs, duration, target_amp)
    signal_power = np.mean(signal**2)

    # Jammer (Barrage)
    jammer_freqs = [100, 110, 120] # Jamming the target and surrounding
    jammer_amp = 1.0
    jammer = generate_jammer(jammer_freqs, fs, duration, jammer_amp)
    interference_power = np.mean(jammer**2)

    # Noise
    noise = np.random.normal(0, noise_floor, len(signal))
    noise_power = np.mean(noise**2)

    # Total Received Signal
    received = signal + jammer + noise

    # Calculate SINR
    sinr = calculate_sinr(signal_power, noise_power, interference_power)
    print(f"--- RF Environment Prototype ---")
    print(f"Target Frequency: {target_freq} Hz")
    print(f"Jammer Frequencies: {jammer_freqs}")
    print(f"Signal Power: {signal_power:.4f}")
    print(f"Interference Power: {interference_power:.4f}")
    print(f"Noise Power: {noise_power:.4f}")
    print(f"Resulting SINR: {sinr:.2f} dB")

    # Test Evasion (Move target to 200Hz)
    print("\n--- Testing Evasion (Moving to 200Hz) ---")
    target_freq_evasive = 200
    signal_evasive = generate_signal(target_freq_evasive, fs, duration, target_amp)

    # In a real scenario, the jammer would be at 100-120Hz
    # For this prototype, we assume the jammer doesn't follow the target yet.
    # We calculate SINR by seeing how much of the jammer's energy overlaps with the target's bandwidth.
    # In this simple point-frequency model, if the target freq is not in jammer_freqs,
    # the interference power at that specific freq is 0 (ideally).

    # Simplified interference check for prototype:
    current_interference = 0 if target_freq_evasive not in jammer_freqs else interference_power
    sinr_evasive = calculate_sinr(signal_power, noise_power, current_interference)
    print(f"Target Frequency: {target_freq_evasive} Hz")
    print(f"Resulting SINR: {sinr_evasive:.2f} dB")

if __name__ == "__main__":
    main()
