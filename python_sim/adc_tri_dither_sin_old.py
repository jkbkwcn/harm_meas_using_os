import numpy as np
import matplotlib.pyplot as plt
from util import si_formatter, get_prefix_multiplier
from adc_oversampling import adc
from adc_tri_dither import tri_wave, os_tri_avg

if __name__ == "__main__":
    vref = 3.3
    bits = 3
    LSB = vref / (2**bits)
    print(f"ADC resolution: {bits} bits | LSB = {si_formatter(LSB, 'V')}")

    f_s = int(1e3)
    print(f"Sampling frequency: {si_formatter(f_s, 'Hz')}")
    Ts = 1 / f_s
    t_stop = 10 * Ts
    nop = int(t_stop * f_s + 1)
    t_cont = np.linspace(0, t_stop, nop * 100)
    t_s = np.linspace(0, t_stop, nop)

    # Input signal
    # signal = LSB * (4.6)
    A = 0.25 * LSB
    offset = LSB * 3.5
    f = 0.25*f_s
    signal_cont = A * np.sin(2 * np.pi * f * t_cont) + offset
    signal = A * np.sin(2 * np.pi * f * t_s) + offset

    # Sampling
    adc_signal = adc(signal, bits, vref)

    # Oversampling with p extra bits
    p = 6
    LSB_OS = vref / (2 ** (bits + p))
    OSR = 2 * 2**p
    print(f"\nOversampling ratio: {OSR} for {p} extra bits")

    f_os = f_s * OSR
    print(f"Oversampling frequency: {si_formatter(f_os, 'Hz')}")
    nop_os = int(t_stop * f_os + 1)
    t_os = np.linspace(0, t_stop, nop_os)

    signal_os = A * np.sin(2 * np.pi * f * t_os) + offset
    adc_signal_os = adc(signal_os, bits, vref)

    # Dithering
    n = 3
    dither_amplitude = (n + 0.5) * LSB
    print(f"\nDither amplitude: {si_formatter(dither_amplitude, 'V', sig_digits=6)}")

    T_tri = OSR / f_os
    f_tri = 1 / T_tri
    print("\nDithering triangle wave parameters:")
    print(f"Triangle wave period: {si_formatter(T_tri, 's')}")
    print(f"Triangle wave frequency: {si_formatter(f_tri, 'Hz')}")
    t_shift = 0
    tri_wave_cont = tri_wave(t_cont + t_shift, f_tri, dither_amplitude)
    signal_dithered_cont = signal_cont + tri_wave_cont

    tri_wave_os = tri_wave(t_os + t_shift, f_tri, dither_amplitude)
    signal_dithered = signal_os + tri_wave_os
    adc_signal_dithered = adc(signal_dithered, bits, vref)

    t_os_avg, adc_signal_dithered_avg = os_tri_avg(t_os, adc_signal_dithered, OSR)

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
    ax1_lsb_os.spines.right.set_position(("axes", 1.05))
    ax1_lsb_os.set_ylim(0, 2 ** (bits + p))
    all_positions_os = np.arange(0, 2 ** (bits + p))
    tick_labels_os = [format(int(val), f"0{bits+p}b") for val in all_positions_os]
    ax1_lsb_os.set_yticks(all_positions_os)
    ax1_lsb_os.set_yticklabels(tick_labels_os)
    for quant in np.arange(0, (2 ** (bits + p))):
        ax1_lsb_os.axhline(quant, linestyle="dotted", color="gray")

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
    plt.show()
