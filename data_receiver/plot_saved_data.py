import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from si_formatter import si_formatter
from pysnr import sinad_signal

DATA_FILE = "measured_data\\adc_data_sin_3Vpkpk_75Hz.npy"

N = 12  # STM32 ADC resolution
OSR = 256  # Oversampling ratio
f_s = 128_571.4 * 2  # Sampling frequency in Hz
f_sig = 75  # Signal frequency in Hz | Only for calculating periods to display

print(f"Calculating parameters for ADC with {N}-bit resolution and OSR={OSR}...")
print(f"  Sampling Frequency: {si_formatter(f_s, 'Hz')}\n\n")


p = int(np.log2(OSR/2))  # Bits added by oversampling

T_sig = 1 / f_sig  # Signal period in seconds

T_s = 1 / f_s  # Sampling period in seconds

n_periods_wide = 15  # Number of signal periods to display in wide view
n_samples_wide = int(n_periods_wide * T_sig / T_s)

n_periods_narrow = 3  # Number of signal periods to display in narrow view
n_samples_narrow = int(n_periods_narrow * T_sig / T_s)


def os_tri_avg(t, sig, OSR: int):
    num_chunks = len(sig) // OSR

    avg_t, avg_sig = np.zeros(num_chunks), np.zeros(num_chunks, dtype=int)
    for i in range(num_chunks):
        start_idx = i * OSR
        end_idx = (i + 1) * OSR
        avg_t[i] = np.mean(t[start_idx:end_idx])
        avg_sig[i] = (np.sum(sig[start_idx:end_idx]) + 1) // 2
    return avg_t, avg_sig


def plot_adc_data(data, filename):
    t = np.linspace(0, len(data) * T_s, len(data), endpoint=False)

    print("Calculating averaged signal...")
    avg_t, avg_sig = os_tri_avg(t, data, OSR)
    print(f"Averaged signal has {len(avg_sig)} samples.")
    print(f"  Min/Max: {avg_sig.min()}/{avg_sig.max()}")
    print(f"  Mean: {avg_sig.mean():.2f}")
    print(f"  Std Dev: {avg_sig.std():.2f}\n\n")
    
    # save avg_sig to csv
    # np.savetxt("adc_data_sin_3Vpkpk_75Hz_avg.csv", avg_sig, delimiter=",", fmt="%d")
    
    data_V = data * (3.3 / (2**N))
    avg_sig_V = avg_sig * (3.3 / (2 ** (N + p)))

    sig_Vpkpk = np.max(avg_sig_V) - np.min(avg_sig_V)
    sig_Vp = sig_Vpkpk / 2
    sig_Vrms = sig_Vp / np.sqrt(2)
    print(f"Signal Peak-to-Peak Voltage: {si_formatter(sig_Vpkpk, 'V')}")
    print(f"Signal Peak Voltage: {si_formatter(sig_Vp, 'V')}")
    print(f"Signal RMS Voltage: {si_formatter(sig_Vrms, 'V')}\n\n")
    
    # histogram plot
    fig, axes = plt.subplots(1, 1, figsize=(8, 5))
    axes.hist(data, bins=2**N, color="gray", edgecolor="black", alpha=0.7)
    axes.set_title("Histogram of ADC Values")
    axes.set_xlabel("ADC Value")
    axes.set_ylabel("Count")
    axes.grid(True, alpha=0.3)
    fig.tight_layout()

    fig1, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].plot(t, data_V, label="Raw Signal", color="blue", linewidth=0.5)
    axes[0].plot(avg_t, avg_sig_V, label="Averaged Signal", color="red", linewidth=1.5)
    axes[0].set_xlim(0, n_periods_wide * T_sig)
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Voltage [V]")
    axes[0].set_title(f"ADC Data - All {len(data)} Samples\nFile: {filename}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    num_to_plot = min(n_samples_narrow, len(data))
    axes[1].plot(t[:num_to_plot], data_V[:num_to_plot], marker="o", label="Raw Signal", color="blue", linewidth=0.5, markersize=2)
    axes[1].plot(avg_t[: (num_to_plot // OSR)], avg_sig_V[: (num_to_plot // OSR)], marker="o", color="red", label="Averaged Signal", linewidth=1.5, markersize=4)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Voltage [V]")
    axes[1].set_title(f"ADC Data - First {num_to_plot} Samples (Detailed View)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig1.tight_layout()

    # FFT Plot
    fig2, axes = plt.subplots(2, 1, figsize=(12, 10))
    n_full = len(data_V)
    fft_vals_full = np.fft.rfft(data_V - np.mean(data_V))
    fft_freqs_full = np.fft.rfftfreq(n_full, 1 / f_s)
    
    print("Calculating SINAD...")
    sinad, noise_harmonic_power = sinad_signal(data_V, f_s)
    enob = (sinad - 1.76) / 6.02
    print(f"SINAD: {sinad:.2f} dB")
    print(f"Noise Power: {noise_harmonic_power:.2f} V^2")
    print(f"ENOB: {enob:.2f} bits\n")
    
    sinad_avg, noise_harmonic_power_avg = sinad_signal(avg_sig_V, f_s / OSR)
    enob_avg = (sinad_avg - 1.76) / 6.02
    print(f"SINAD (Averaged): {sinad_avg:.2f} dB")
    print(f"Noise Power (Averaged): {noise_harmonic_power_avg:.2f} V^2")
    print(f"ENOB (Averaged): {enob_avg:.2f} bits\n")
    
    print(f"ENOB gain due to averaging: {enob_avg - enob:.2f} bits\n\n")

    # Convert to RMS voltage
    amplitude_rms_full = np.abs(fft_vals_full) / n_full
    amplitude_rms_full[1:-1] *= 2  # Non-DC and non-Nyquist components
    
    # Convert to dBV (dB relative to 1V RMS)
    amplitude_db_full = 20 * np.log10(amplitude_rms_full + 1e-12)

    axes[0].plot(fft_freqs_full / 1000, amplitude_db_full, color="blue", linewidth=0.7)
    axes[0].set_title("FFT Spectrum - All Samples")
    axes[0].set_xlabel("Frequency [kHz]")
    axes[0].set_ylabel("Amplitude [dBV RMS]")
    axes[0].grid(True, alpha=0.3)

    n_avg = len(avg_sig_V)
    fft_vals_avg = np.fft.rfft(avg_sig_V - np.mean(avg_sig_V))
    fft_freqs_avg = np.fft.rfftfreq(n_avg, 1 / (f_s / OSR))

    # Convert to RMS voltage
    amplitude_rms_avg = np.abs(fft_vals_avg) / n_avg
    amplitude_rms_avg[1:-1] *= 2  # Non-DC and non-Nyquist components
    # Convert to dBV (dB relative to 1V RMS)
    amplitude_db_avg = 20 * np.log10(amplitude_rms_avg + 1e-12)

    axes[1].plot(fft_freqs_avg, amplitude_db_avg, color="red", linewidth=0.7)
    axes[1].set_title("FFT Spectrum - Averaged Signal")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Amplitude [dBV RMS]")
    axes[1].grid(True, alpha=0.3)
    fig2.tight_layout()

    plt.show()


def main():
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: File '{DATA_FILE}' not found!")
        sys.exit(1)

    print(f"Loading data from: {DATA_FILE}")

    try:
        # Load the data
        data = np.load(DATA_FILE)

        print(f"Loaded {len(data)} samples")
        print(f"  Data type: {data.dtype}")
        print(f"  Shape: {data.shape}")
        print(f"  Min/Max: {data.min()}/{data.max()}")
        print(f"  Mean: {data.mean():.2f}")
        print(f"  Std Dev: {data.std():.2f}\n\n")

        # Plot the data
        plot_adc_data(data, os.path.basename(DATA_FILE))

    except Exception as e:
        print(f"❌ Error loading or plotting data: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
