import numpy as np
import matplotlib.pyplot as plt
from util import si_formatter, get_prefix_multiplier
from adc_oversampling import adc
from adc_tri_dither import tri_wave, os_tri_avg


if __name__ == "__main__":
    vref = 3.3
    bits = 6
    LSB = vref / (2**bits)
    print(f"ADC resolution: {bits} bits | LSB = {si_formatter(LSB, 'V')}")

    f_s = int(1e3)
    print(f"Sampling frequency: {si_formatter(f_s, 'Hz')}")
    Ts = 1 / f_s
    t_stop = 10_000 * Ts
    nop = int(t_stop * f_s + 1)
    t_cont = np.linspace(0, t_stop, nop * 1000)
    t_s = np.linspace(0, t_stop, nop)

    # Input signal
    # signal = LSB * (4.6)
    A1 = 0.1 * LSB
    f1 = 50
    offset = 20 * LSB
    A2 = 0.1 * LSB
    f2 = 250

    noise_rms = 0.3 * LSB

    def signal(t):
        noise = np.random.normal(0, noise_rms, size=t.shape)
        return A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2 * t) + offset + noise
        # return A1 * np.sin(2 * np.pi * f1 * t) + offset

    signal_cont = signal(t_cont)
    signal_s = signal(t_s)
    # Sampling
    adc_signal = adc(signal_s, bits, vref)

    # Oversampling with p extra bits
    p = 3
    LSB_OS = vref / (2 ** (bits + p))
    OSR = 2 * 2**p
    print(f"\nOversampling ratio: {OSR} for {p} extra bits")

    f_os = f_s * OSR
    print(f"Oversampling frequency: {si_formatter(f_os, 'Hz')}")
    nop_os = int(t_stop * f_os + 1)
    t_os = np.linspace(0, t_stop, nop_os)

    signal_os = signal(t_os)
    adc_signal_os = adc(signal_os, bits, vref)

    # Dithering
    n = 0
    dither_amplitude = (n + 0.5) * LSB
    print(f"\nDither amplitude: {si_formatter(dither_amplitude, 'V', sig_digits=6)}")

    T_tri = OSR / f_os
    f_tri = 1 / T_tri
    print("\nDithering triangle wave parameters:")
    print(f"Triangle wave period: {si_formatter(T_tri, 's')}")
    print(f"Triangle wave frequency: {si_formatter(f_tri, 'Hz')}")
    t_shift = 0
    # Use TRI data from data3.txt as dithering signal
    tri_wave_cont = tri_wave(t_cont, f_tri, dither_amplitude)
    signal_dithered_cont = signal_cont + tri_wave_cont

    tri_wave_os = tri_wave(t_os, f_tri, dither_amplitude)
    signal_dithered = signal_os + tri_wave_os
    adc_signal_dithered = adc(signal_dithered, bits, vref)

    t_os_avg, adc_signal_dithered_avg = os_tri_avg(t_os, adc_signal_dithered, OSR)

    # FFT Calculations
    print("\n" + "=" * 60)
    print("FFT ANALYSIS")
    print("=" * 60)

    # FFT of adc_signal (no oversampling)
    N_s = len(adc_signal)
    N_fft_s = N_s  # Zero-padded FFT length
    adc_signal_dc_removed = adc_signal - np.mean(adc_signal)
    fft_adc = np.fft.fft(adc_signal_dc_removed, n=N_fft_s)
    fft_adc_mag = np.abs(fft_adc) / N_s * 2  # Normalize and scale for single-sided
    fft_adc_mag[0] /= 2  # DC component should not be doubled
    freq_s = np.fft.fftfreq(N_fft_s, Ts)

    # FFT of adc_signal_os (oversampled, no averaging)
    N_os = len(adc_signal_dithered)
    N_fft_os = N_os  # Zero-padded FFT length
    adc_signal_os_dc_removed = adc_signal_dithered - np.mean(adc_signal_dithered)
    fft_adc_os = np.fft.fft(adc_signal_os_dc_removed, n=N_fft_os)
    fft_adc_os_mag = np.abs(fft_adc_os) / N_os * 2
    fft_adc_os_mag[0] /= 2
    freq_os = np.fft.fftfreq(N_fft_os, 1 / f_os)

    # FFT of adc_signal_dithered_avg (oversampled + dithered + averaged)
    N_avg = len(adc_signal_dithered_avg)
    N_fft_avg = N_avg  # Zero-padded FFT length
    adc_signal_dithered_avg_dc_removed = adc_signal_dithered_avg - np.mean(adc_signal_dithered_avg)
    fft_adc_avg = np.fft.fft(adc_signal_dithered_avg_dc_removed, n=N_fft_avg)
    fft_adc_avg_mag = np.abs(fft_adc_avg) / N_avg * 2
    fft_adc_avg_mag[0] /= 2
    freq_avg = np.fft.fftfreq(N_fft_avg, Ts)

    # Find peak frequency and magnitude for each signal
    # Only consider positive frequencies
    pos_freq_s = freq_s[: N_fft_s // 2]
    pos_freq_os = freq_os[: N_fft_os // 2]
    pos_freq_avg = freq_avg[: N_fft_avg // 2]

    peak_idx_s = np.argmax(fft_adc_mag[1 : N_fft_s // 2]) + 1  # Skip DC
    peak_idx_os = np.argmax(fft_adc_os_mag[1 : N_fft_os // 2]) + 1
    peak_idx_avg = np.argmax(fft_adc_avg_mag[1 : N_fft_avg // 2]) + 1

    # Find magnitude at exact signal frequency
    sig_freq_idx_s = np.argmin(np.abs(freq_s[: N_fft_s // 2] - f1))
    sig_freq_idx_os = np.argmin(np.abs(freq_os[: N_fft_os // 2] - f1))
    sig_freq_idx_avg = np.argmin(np.abs(freq_avg[: N_fft_avg // 2] - f1))

    # Calculate dB values
    peak_mag_db_s = 20 * np.log10(fft_adc_mag[peak_idx_s] + 1e-12)
    peak_mag_db_os = 20 * np.log10(fft_adc_os_mag[peak_idx_os] + 1e-12)
    peak_mag_db_avg = 20 * np.log10(fft_adc_avg_mag[peak_idx_avg] + 1e-12)

    sig_mag_db_s = 20 * np.log10(fft_adc_mag[sig_freq_idx_s] + 1e-12)
    sig_mag_db_os = 20 * np.log10(fft_adc_os_mag[sig_freq_idx_os] + 1e-12)
    sig_mag_db_avg = 20 * np.log10(fft_adc_avg_mag[sig_freq_idx_avg] + 1e-12)

    print(f"Signal frequency: {si_formatter(f1, 'Hz')}")
    print(f"\nNo Oversampling:")
    print(f"  Peak frequency: {si_formatter(freq_s[peak_idx_s], 'Hz')}")
    print(f"  Peak magnitude: {fft_adc_mag[peak_idx_s]:.4f} LSB ({peak_mag_db_s:.2f} dB)")
    print(f"  Magnitude at {si_formatter(f1, 'Hz')}: {fft_adc_mag[sig_freq_idx_s]:.4f} LSB ({sig_mag_db_s:.2f} dB)")
    print(f"\nOversampled (no averaging):")
    print(f"  Peak frequency: {si_formatter(freq_os[peak_idx_os], 'Hz')}")
    print(f"  Peak magnitude: {fft_adc_os_mag[peak_idx_os]:.4f} LSB ({peak_mag_db_os:.2f} dB)")
    print(f"  Magnitude at {si_formatter(f1, 'Hz')}: {fft_adc_os_mag[sig_freq_idx_os]:.4f} LSB ({sig_mag_db_os:.2f} dB)")
    print(f"\nOversampled + Dithered + Averaged:")
    print(f"  Peak frequency: {si_formatter(freq_avg[peak_idx_avg], 'Hz')}")
    print(f"  Peak magnitude: {fft_adc_avg_mag[peak_idx_avg]:.4f} LSB_OS ({peak_mag_db_avg:.2f} dB)")
    print(f"  Magnitude at {si_formatter(f1, 'Hz')}: {fft_adc_avg_mag[sig_freq_idx_avg]:.4f} LSB_OS ({sig_mag_db_avg:.2f} dB)")
    print("=" * 60 + "\n")

    # print(f"\nSignal  level: {si_formatter(signal_cont, 'V', sig_digits=6)}")
    # print(f"ADC output: {si_formatter(adc_signal*LSB, 'V', sig_digits=6)}")
    # print(f"ADC output (OS, dither, avg): {si_formatter(adc_signal_dithered_avg[0]*LSB_OS, 'V', sig_digits=6)}")

    # print(f"\nDifference (no OS): {si_formatter(abs(signal_cont - adc_signal*LSB), 'V', sig_digits=6)}")
    # print(f"Difference (OS, dither, avg): {si_formatter(abs(signal_cont - adc_signal_dithered_avg[0]*LSB_OS), 'V', sig_digits=6)}")
    # print(f"Oversampling improvement: {round(abs(signal_cont - adc_signal*LSB) / abs(signal_cont - adc_signal_dithered_avg[0]*LSB_OS), 2)}x")

    ### Plotting
    fig1, ax1_v = plt.subplots()
    t_prefix, t_mult = get_prefix_multiplier(t_stop)
    ax1_v.set_xlabel(f"Time ({t_prefix}s)")
    ax1_v.set_ylabel("Amplitude (V)")
    margin = 3 * LSB
    ax1_v.set_ylim(0, 3.3)

    ax1_lsb = ax1_v.twinx()
    ax1_lsb.set_ylabel("ADC Output")
    ax1_lsb.set_ylim(0, 2**bits)
    all_positions = np.arange(0, 2**bits)
    tick_labels = [format(int(val), f"0{bits}b") for val in all_positions]
    ax1_lsb.set_yticks(all_positions)
    ax1_lsb.set_yticklabels(tick_labels)
    for quant in np.arange(0, (2**bits)):
        ax1_lsb.axhline(quant, linestyle="dashed", color="k")

    ax1_lsb_os = ax1_v.twinx()
    ax1_lsb_os.set_ylabel("ADC Output with Oversampling")
    # ax1_lsb_os.spines.right.set_position(("axes", 1.05))
    ax1_lsb_os.set_ylim(0, 2 ** (bits + p))
    # all_positions_os = np.arange(0, 2 ** (bits + p))
    # tick_labels_os = [format(int(val), f"0{bits+p}b") for val in all_positions_os]
    # ax1_lsb_os.set_yticks(all_positions_os)
    # ax1_lsb_os.set_yticklabels(tick_labels_os)
    # for quant in np.arange(0, (2 ** (bits + p))):
    #     ax1_lsb_os.axhline(quant, linestyle="dotted", color="gray")

    ax1_v.plot(t_cont * t_mult, signal_cont, "r-", label="Input Signal")

    ax1_lsb.scatter(t_s * t_mult, adc_signal, c="r", s=100, label="No Oversampling", zorder=20)

    ax1_v.plot(t_cont * t_mult, signal_dithered_cont, "g-", label="Dithering Signal")
    s = np.full_like(t_os, 50)
    for i in range(0, len(t_os), OSR):
        s[i] = 100
    ax1_lsb.scatter(t_os * t_mult, adc_signal_dithered, c="g", s=s, label="Oversampled", zorder=10)

    ax1_lsb_os.scatter(t_os_avg * t_mult, adc_signal_dithered_avg, c="b", label="Averaged Oversampled", zorder=10)

    ax1_v.legend(loc="upper left")
    ax1_lsb.legend(loc="upper center")
    ax1_lsb_os.legend(loc="upper right")
    plt.tight_layout()

    # FFT Plot
    fig2, (ax2_1, ax2_2, ax2_3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig2.suptitle("FFT Analysis - Frequency Domain", fontsize=14, fontweight="bold")

    # Plot 1: No Oversampling
    fft_adc_mag_db = 20 * np.log10(fft_adc_mag[: N_fft_s // 2] + 1e-12)  # in dB, avoid log(0)
    ax2_1.semilogx(freq_s[: N_fft_s // 2], fft_adc_mag_db, "r-", linewidth=1.5)
    ax2_1.plot(freq_s[peak_idx_s], fft_adc_mag_db[peak_idx_s], "rs", markersize=10, label=f"Peak: {peak_mag_db_s:.2f} dB")
    ax2_1.plot(freq_s[sig_freq_idx_s], fft_adc_mag_db[sig_freq_idx_s], "ro", markersize=8, label=f'At {si_formatter(f1, "Hz")}: {sig_mag_db_s:.2f} dB')
    ax2_1.set_ylabel("Magnitude (dB LSB)")
    ax2_1.set_title("No Oversampling")
    ax2_1.grid(True, alpha=0.3)
    ax2_1.set_xlim([0, f_s / 2])
    ax2_1.axvline(f1, color="k", linestyle="--", alpha=0.5)
    ax2_1.legend()

    # Plot 2: Oversampled (no averaging)
    fft_adc_os_mag_db = 20 * np.log10(fft_adc_os_mag[: N_fft_os // 2] + 1e-12)  # in dB, avoid log(0)
    ax2_2.semilogx(freq_os[: N_fft_os // 2], fft_adc_os_mag_db, "g-", linewidth=1.5)
    ax2_2.plot(freq_os[peak_idx_os], fft_adc_os_mag_db[peak_idx_os], "gs", markersize=10, label=f"Peak: {peak_mag_db_os:.2f} dB")
    ax2_2.plot(freq_os[sig_freq_idx_os], fft_adc_os_mag_db[sig_freq_idx_os], "go", markersize=8, label=f'At {si_formatter(f1, "Hz")}: {sig_mag_db_os:.2f} dB')
    ax2_2.set_ylabel("Magnitude (dB LSB)")
    ax2_2.set_title("Oversampled (no averaging)")
    ax2_2.grid(True, alpha=0.3)
    ax2_2.set_xlim([0, f_os / 2])
    ax2_2.axvline(f1, color="k", linestyle="--", alpha=0.5)
    ax2_2.legend()

    # Plot 3: Oversampled + Dithered + Averaged
    fft_adc_avg_mag_db = 20 * np.log10(fft_adc_avg_mag[: N_fft_avg // 2] + 1e-12)  # in dB, avoid log(0)
    ax2_3.semilogx(freq_avg[: N_fft_avg // 2], fft_adc_avg_mag_db, "b-", linewidth=1.5)
    ax2_3.plot(freq_avg[peak_idx_avg], fft_adc_avg_mag_db[peak_idx_avg], "bs", markersize=10, label=f"Peak: {peak_mag_db_avg:.2f} dB")
    ax2_3.plot(freq_avg[sig_freq_idx_avg], fft_adc_avg_mag_db[sig_freq_idx_avg], "bo", markersize=8, label=f'At {si_formatter(f1, "Hz")}: {sig_mag_db_avg:.2f} dB')
    ax2_3.set_xlabel("Frequency (Hz)")
    ax2_3.set_ylabel("Magnitude (dB LSB_OS)")
    ax2_3.set_title("Oversampled + Dithered + Averaged")
    ax2_3.grid(True, alpha=0.3)
    ax2_3.set_xlim([0, f_s / 2])
    ax2_3.axvline(f1, color="k", linestyle="--", alpha=0.5)
    ax2_3.legend()

    plt.tight_layout()
    plt.show()
