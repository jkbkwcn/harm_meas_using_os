#!/usr/bin/env python3
"""
Plot ADC Data from Saved .npy File
Loads and visualizes ADC data previously saved by adc_receiver_plot.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy import signal
from pysnr import sinad_signal

N = 12
p = 7
f_s = 128_571.4 * 2  # Hz
first_n_samples = 10000


def os_tri_avg(t, sig, OSR):
    num_chunks = len(sig) // OSR

    avg_t, avg_sig = np.zeros(num_chunks), np.zeros(num_chunks, dtype=int)
    for i in range(num_chunks):
        start_idx = i * OSR
        end_idx = (i + 1) * OSR
        avg_t[i] = np.mean(t[start_idx:end_idx])
        avg_sig[i] = (np.sum(sig[start_idx:end_idx]) + 1) // 2
    return avg_t, avg_sig


def plot_adc_data(data, filename):
    """
    Plot ADC data with multiple visualizations

    Args:
        data: NumPy array of ADC samples
        filename: Name of the file (for title)
    """
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Time axis (sample indices)
    # cut first 8 frames, 8 * 512 elements from data
    data = data[4096:]
    sample_indices = np.arange(len(data))

    # Plot 1: All samples
    axes[0].plot(sample_indices, data, linewidth=0.5)
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("ADC Value")
    axes[0].set_title(f"ADC Data - All {len(data)} Samples\nFile: {filename}")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: First n samples (or all if less)
    num_to_plot = min(first_n_samples, len(data))
    axes[1].plot(sample_indices[:num_to_plot], data[:num_to_plot], marker="o", markersize=2, linewidth=0.8)
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("ADC Value")
    axes[1].set_title(f"ADC Data - First {num_to_plot} Samples (Detailed View)")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Histogram
    axes[2].hist(data, bins=100, edgecolor="black", alpha=0.7)
    axes[2].set_xlabel("ADC Value")
    axes[2].set_ylabel("Count")
    axes[2].set_title("ADC Value Distribution")
    axes[2].grid(True, alpha=0.3, axis="y")

    # New figure for averaged signal histogram
    fig_hist, ax_hist = plt.subplots(1, 1, figsize=(12, 5))

    fig1, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Filtration - averaging
    t = np.arange(len(data)) / f_s
    OSR = 2 * 2**p
    avg_t, avg_sig = os_tri_avg(t, data, OSR)

    # Histogram of averaged signal
    ax_hist.hist(avg_sig, bins=100, edgecolor="black", alpha=0.7, color='red')
    ax_hist.set_xlabel("ADC Value (Averaged)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Averaged Signal Value Distribution")
    ax_hist.grid(True, alpha=0.3, axis="y")

    # Filtration - FIR LP filter
    f_s_avg = (f_s / (OSR))     # ~ 1004.5 Hz
    numtaps = 251
    b_fir = signal.firwin(numtaps, 275.0, fs=f_s_avg)
    a_fir = [1.0]
    fir_sig = signal.lfilter(b_fir, a_fir, avg_sig)

    data_V = data * (3.3 / (2**N))
    avg_sig_V = avg_sig * (3.3 / (2 ** (N + p)))
    fir_sig_V = fir_sig * (3.3 / (2 ** (N + p)))
    
    
    sinad_raw, noise_harmonic_power_raw = sinad_signal(data_V, f_s)
    enob = (sinad_raw - 1.76) / 6.02
    sinad_avg, noise_harmonic_power_avg = sinad_signal(avg_sig_V, f_s_avg)
    enob_avg = (sinad_avg - 1.76) / 6.02
    sinad_fir, noise_harmonic_power_fir = sinad_signal(fir_sig_V[numtaps-1:], f_s_avg)
    enob_fir = (sinad_fir - 1.76) / 6.02
    
    print(f"SINAD (Raw): {sinad_raw:.2f} dB")
    print(f"SINAD (Averaged): {sinad_avg:.2f} dB")
    print(f"SINAD (Averaged + FIR): {sinad_fir:.2f} dB")
    print(f"ENOB (Raw): {enob:.2f} bits")
    print(f"ENOB (Averaged): {enob_avg:.2f} bits")
    print(f"ENOB (Averaged + FIR): {enob_fir:.2f} bits")    

    # Plot 1: All samples
    axes[0].plot(t, data_V, linewidth=0.5)
    axes[0].plot(avg_t, avg_sig_V, color="red", linewidth=1.5)
    axes[0].plot(avg_t, fir_sig_V, color="black", linewidth=1.5)
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Voltage [V]")
    axes[0].set_title(f"ADC Data - All {len(data)} Samples\nFile: {filename}")
    axes[0].legend([
        "RAW DATA",
        "AVG",
        "AVG + FIR LP"
    ])
    axes[0].grid(True, alpha=0.3)

    # Plot 2: First n samples (or all if less)
    num_to_plot = min(first_n_samples, len(data))
    avg_t_short, avg_sig_short = os_tri_avg(t[:num_to_plot], data[:num_to_plot], OSR)
    avg_sig_short_V = avg_sig_short * (3.3 / (2 ** (N + p)))
    axes[1].plot(t[:num_to_plot], data_V[:num_to_plot], marker="o", markersize=2, linewidth=0.8)
    axes[1].plot(avg_t_short, avg_sig_short_V, color="red", linewidth=1.5)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Voltage [V]")
    axes[1].set_title(f"ADC Data - First {num_to_plot} Samples (Detailed View)")
    axes[1].grid(True, alpha=0.3)

    # FFT Plot
    fig2, axes = plt.subplots(2, 1, figsize=(12, 10))
    n_full = len(data)
    fft_vals_full = np.fft.rfft(data - np.mean(data))
    fft_freqs_full = np.fft.rfftfreq(n_full, 1 / f_s)
    amplitude_db_full = 20 * np.log10(np.abs(fft_vals_full) / n_full + 1e-9)
    amplitude_db_full -= np.max(amplitude_db_full)
    axes[0].plot(fft_freqs_full / 1000, amplitude_db_full, color="blue", linewidth=0.7)
    axes[0].axvline(x=0.05, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[0].axvline(x=0.1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[0].axvline(x=0.15, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[0].axvline(x=0.25, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[0].set_title("FFT Spectrum - All Samples")
    axes[0].set_xlabel("Frequency [kHz]")
    axes[0].set_ylabel("Amplitude [dB]")
    axes[0].grid(True, alpha=0.3)

    n_avg = len(avg_sig)
    fft_vals_avg = np.fft.rfft(avg_sig - np.mean(avg_sig))
    fft_freqs_avg = np.fft.rfftfreq(n_avg, 1 / (f_s / OSR))
    amplitude_db_avg = 20 * np.log10(np.abs(fft_vals_avg) / n_avg + 1e-9)
    # amplitude_db_avg -= np.max(amplitude_db_avg)
    axes[1].plot(fft_freqs_avg / 1000, amplitude_db_avg, color="red", linewidth=0.7)
    axes[1].axvline(x=0.05, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[1].axvline(x=0.1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[1].axvline(x=0.15, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[1].axvline(x=0.25, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[1].set_title("FFT Spectrum - Averaged Signal")
    axes[1].set_xlabel("Frequency [kHz]")
    axes[1].set_ylabel("Amplitude [dB]")
    axes[1].grid(True, alpha=0.3)

    # FFT Plot after averaging and FIR filter
    fig3, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].plot(fft_freqs_avg / 1000, amplitude_db_avg, color="red", linewidth=0.7)
    axes[0].axvline(x=0.05, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[0].axvline(x=0.1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[0].axvline(x=0.15, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[0].axvline(x=0.25, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[0].set_title("FFT Spectrum - Averaged Signal")
    axes[0].set_xlabel("Frequency [kHz]")
    axes[0].set_ylabel("Amplitude [dB]")
    axes[0].grid(True, alpha=0.3)

    n_fir = len(fir_sig[1024:])
    fft_vals_fir = np.fft.rfft(fir_sig[1024:] - np.mean(fir_sig[1024:]))
    fft_freqs_fir = np.fft.rfftfreq(n_fir, 1 / f_s_avg)
    amplitude_db_fir = 20 * np.log10(np.abs(fft_vals_fir) / n_fir + 1e-9)
    # amplitude_db_fir -= np.max(amplitude_db_fir)

    axes[1].plot(fft_freqs_fir / 1000, amplitude_db_fir, color="black", linewidth=0.7)
    axes[1].axvline(x=0.05, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[1].axvline(x=0.1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[1].axvline(x=0.15, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[1].axvline(x=0.25, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[1].set_title("FFT Spectrum - averaging + FIR")
    axes[1].set_xlabel("Frequency [kHz]")
    axes[1].set_ylabel("Amplitude [dB]")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """Main function"""
    # ========== CONFIGURATION ==========
    DATA_FILE = "measured_data/adc_data_sin_200uVpkpk_50Hz.npy"  # Change to your .npy file
    # ===================================

    # Allow command line argument to override
    if len(sys.argv) > 1:
        DATA_FILE = sys.argv[1]

    # Check if file exists
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: File '{DATA_FILE}' not found!")
        print(f"\nUsage: python {os.path.basename(__file__)} [filename.npy]")
        sys.exit(1)

    print(f"Loading data from: {DATA_FILE}")

    try:
        # Load the data
        data = np.load(DATA_FILE)

        print(f"✓ Loaded {len(data)} samples")
        print(f"  Data type: {data.dtype}")
        print(f"  Shape: {data.shape}")
        print(f"  Min/Max: {data.min()}/{data.max()}")
        print(f"  Mean: {data.mean():.2f}")
        print(f"  Std Dev: {data.std():.2f}")

        # Plot the data
        print("\nGenerating plots...")
        plot_adc_data(data, os.path.basename(DATA_FILE))

    except Exception as e:
        print(f"❌ Error loading or plotting data: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
