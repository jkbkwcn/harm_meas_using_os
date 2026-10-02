import numpy as np
import matplotlib.pyplot as plt
from util import si_formatter, get_prefix_multiplier
from adc_oversampling import adc, avg
from scipy.signal import sawtooth
import pandas as pd
from scipy.interpolate import interp1d


def tri_wave(t: np.ndarray, f: float, A: float) -> np.ndarray:
    """Generate a triangle wave.

    Args:
        t (np.ndarray): Time array.
        f (float): Frequency of the triangle wave.
        A (float): Amplitude of the triangle wave.

    Returns:
        np.ndarray: Triangle wave values at time t.
    """
    return A * sawtooth(2 * np.pi * f * t, width=0.5)


def os_tri_avg(t: np.ndarray[float], sig: np.ndarray[int], OSR: int) -> tuple[np.ndarray[float], np.ndarray[int]]:
    num_chunks = len(sig) // OSR

    avg_t, avg_sig = np.zeros(num_chunks), np.zeros(num_chunks, dtype=int)
    for i in range(num_chunks):
        start_idx = i * OSR
        end_idx = (i + 1) * OSR
        avg_t[i] = np.mean(t[start_idx:end_idx])
        avg_sig[i] = (np.sum(sig[start_idx:end_idx]) + 1) // 2
    return avg_t, avg_sig


if __name__ == "__main__":
    # ADC parameters
    Vref = 3.3
    bits = 3
    LSB = Vref / (2**bits)
    print(f"ADC resolution: {bits} bits | LSB = {si_formatter(LSB, 'V')}")

    f_s = int(1e3)
    print(f"Sampling frequency: {si_formatter(f_s, 'Hz')}")
    T_s = 1 / f_s

    # Time arrays
    t_stop = 3 * T_s
    nop = int(t_stop * f_s + 1)
    t_cont = np.linspace(0, t_stop, nop * 1000)
    t_s = np.linspace(0, t_stop, nop)

    # Input signal
    A = LSB * (5.1)
    noise_rms = 0
    signal_cont = np.full_like(t_cont, A) + np.random.normal(scale=noise_rms, size=len(t_cont))
    signal_s = np.full_like(t_s, A) + np.random.normal(scale=noise_rms, size=len(t_s))

    # Load triangle wave data from data3.txt
    data3 = pd.read_csv("python_sim\\data3.txt", sep="\t", names=["t", "ADC", "TRI"])
    tri_interp = interp1d(data3.t, data3.TRI, kind="linear", bounds_error=False, fill_value=(0, 0))

    # Sampling
    adc_signal = adc(signal_s, bits, Vref)

    # Oversampling with p extra bits
    p = 3
    bits_os = bits + p
    LSB_OS = Vref / (2 ** (bits_os))
    OSR = 2 * 2**p
    print(f"\nOversampling ratio: {OSR} for {p} extra bits")

    f_os = f_s * OSR
    print(f"Oversampling frequency: {si_formatter(f_os, 'Hz')}")
    nop_os = int(t_stop * f_os + 1)
    t_os = np.linspace(0, t_stop, nop_os)

    signal_os = np.full_like(t_os, A) + np.random.normal(scale=noise_rms, size=len(t_os))
    adc_signal_os = adc(signal_os, bits, Vref)

    # Dithering
    n = 0
    dither_amplitude = (n + 0.5) * LSB
    print(f"\nDither amplitude: {si_formatter(dither_amplitude, 'V', sig_digits=6)}")

    k = 1
    T_tri = OSR / (k * f_os)
    f_tri = 1 / T_tri
    print(f"Triangle wave period: {si_formatter(T_tri, 's')}")
    print(f"Triangle wave frequency: {si_formatter(f_tri, 'Hz')}")
    t_shift = 0

    # Use TRI data from data3.txt as dithering signal
    tri_wave_cont = tri_interp(t_cont + t_shift) - np.mean(tri_interp(data3.t))  # Center around zero
    tri_wave_cont = tri_wave_cont * (dither_amplitude / np.max(np.abs(tri_wave_cont)))  # Scale to dither amplitude
    signal_dithered_cont = signal_cont + tri_wave_cont

    tri_wave_os = tri_interp(t_os + t_shift) - np.mean(tri_interp(data3.t))  # Center around zero
    tri_wave_os = tri_wave_os * (dither_amplitude / np.max(np.abs(tri_wave_os)))  # Scale to dither amplitude
    signal_dithered = signal_os + tri_wave_os
    adc_signal_dithered = adc(signal_dithered, bits, Vref)

    adc_code_count = {}
    for i in range(OSR):
        code = adc_signal_dithered[i]
        adc_code_count[code] = adc_code_count.get(code, 0) + 1

    t_os_avg, adc_signal_dithered_avg = os_tri_avg(t_os, adc_signal_dithered, OSR)

    ### Plotting
    # setup
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
    s = [100 if i % OSR == 0 else 50 for i in range(len(t_os))]

    # plotting
    ax1_v.plot(t_cont * t_mult, signal_cont, "r-", label="Input Signal")

    ax1_lsb.scatter(t_s * t_mult, adc_signal, c="r", s=100, label="ADC", zorder=10)

    ax1_v.plot(t_cont * t_mult, signal_dithered_cont, "g-", label="Dithering Signal")

    ax1_lsb.scatter(t_os * t_mult, adc_signal_dithered, c="g", s=s, label="ADC OS DITH", zorder=10)

    ax1_lsb_os.scatter(t_os_avg * t_mult, adc_signal_dithered_avg, c="b", label="ADC OS DITH AVG", zorder=10)

    desc = "Dithered ADC Code Counts:\n"
    avg_desc = "Result = ("
    for code in sorted(adc_code_count.keys()):
        desc += f"Code {format(code, f'0{bits}b')}: {adc_code_count[code]}\n"
        avg_desc += f"{adc_code_count[code]}*{format(code, f'0{bits}b')} + "
    avg_desc = avg_desc[:-3] + f" + 1) >> 1 = {format(adc_signal_dithered_avg[0], f'0{bits_os}b')}"
    desc = desc + avg_desc

    ax1_v.set_title(desc)

    fig1.legend()
    fig1.tight_layout()
    plt.show()
