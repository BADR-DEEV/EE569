import numpy as np
import matplotlib.pyplot as plt

# Define parameters from the equation
Ac = 8  # Carrier amplitude
mu = 0.5  # Modulation index
fm = 5000  # Modulating frequency (Hz)
fc = 1e7  # Carrier frequency (Hz)

# Time vector for the signal (adjusted for pulse-like view)
t = np.linspace(0, 1e-4, 500)  # 100 µs with fewer points for pulse-like effect

# Define the modulating and carrier signals
modulating_signal = np.sin(2 * np.pi * fm * t)  # Sinusoidal modulating signal
envelope = Ac * (1 + mu * modulating_signal)  # Envelope
carrier_signal = np.cos(2 * np.pi * fc * t)  # Carrier signal
modulated_signal = envelope * carrier_signal  # AM modulated signal

# Plot the modulated signal as pulses
plt.figure(figsize=(12, 6))
plt.stem(t, modulated_signal, linefmt='b-', markerfmt='bo', basefmt=" ", label="AM Signal")
plt.plot(t, envelope, label="Upper Envelope", linestyle="--", color="red")
plt.plot(t, -envelope, label="Lower Envelope", linestyle="--", color="green")
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.title("Amplitude Modulated Signal (Pulse Representation)", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
