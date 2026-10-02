import numpy as np
import matplotlib.pyplot as plt
from util import si_formatter


def adc(sig: np.ndarray[float], bits: int, vref: float) -> np.ndarray[int]:
    """
    N-bit ADC model with mid-tread quantization (rounding)
    
    Input range: [0, Vref]
    Output codes: 0 to (2^N - 1)
    LSB = Vref / 2^N
    
    Quantization levels are at: code * LSB + LSB/2
    - Values >= code*LSB + LSB/2 round to (code+1)
    - Values < code*LSB + LSB/2 round to code
    """
    # Clip input to valid range [0, Vref]
    sig_clipped = np.clip(sig, 0, vref)
    # Quantize: divide by LSB and round to nearest integer
    codes = np.round(sig_clipped * (2**bits) / vref).astype(int)
    # Clip to maximum code value (2^bits - 1)
    codes = np.clip(codes, 0, 2**bits - 1)
    return codes


def avg(t: np.ndarray[float], sig: np.ndarray[int], OSR: int) -> tuple[np.ndarray[float], np.ndarray[int]]:
    p = np.emath.logn(4, OSR)
    num_blocks = len(sig) // OSR
    avg_sig = np.zeros(num_blocks)
    new_t = np.zeros(num_blocks)
    for i in range(num_blocks):
        start = i * OSR
        stop = (i + 1) * OSR
        avg_sig[i] = np.sum(sig[start:stop]).astype(int)
        new_t[i] = np.mean(t[start:stop])
    avg_sig = avg_sig // 2**p
    return new_t, avg_sig


if __name__ == "__main__":
    # Params
    vref = 3.3
    bits = 10
    LSB = vref / 2**bits
    vin_st = np.linspace(0, vref, int(1e6))

    # Input signals with and without dither
    A = vref / (4 * 2**bits)
    print(f"Signal amplitude: {si_formatter(A, 'V')}")

    fs = int(1e6)
    f0 = int(100)

    t_stop = 0.01
    t = np.linspace(0, t_stop, int(t_stop * fs + 1))

    offset = vref / 2
    print(f"Signal offset: {si_formatter(offset, 'V')}")
    sig_clean = offset + A * np.sin(2 * np.pi * f0 * t)
    sig_white = sig_clean + np.random.normal(scale=1 / 2 ** (bits - 1), size=len(t))

    # Averaging
    OSR = 16
    t_osr_cl, avg_osr_cl = avg(t, adc(sig_clean, bits, vref), OSR)
    t_osr, avg_osr = avg(t, adc(sig_white, bits, vref), OSR)

    # #########################################################################
    # ################################# PLOTS #################################
    # #########################################################################

    # ############################# FIGURE 1 #############################
    fig1, ax11 = plt.subplots()

    ax11.plot(vin_st, vin_st, color="black")
    ax11.set_xlabel(r"$V_{IN}$", color="black")
    ax11.set_ylabel(r"$V_{IN}$", color="black")
    ax11.tick_params(axis="y", colors="black")
    ax11.set_title("ADC characteristics")
    ax11.grid()

    ax12 = ax11.twinx()
    ax12.plot(vin_st, adc(vin_st, bits, vref), color="red")
    ax12.set_ylabel(r"$ADC_{OUT}$", color="red")
    ax12.tick_params(axis="y", colors="red")

    ax11.set_ylim(0, vref)
    ax12.set_ylim(0, 2**bits)

    # ############################# FIGURE 2 #############################
    fig2, (ax21, ax23) = plt.subplots(2, 1)

    ax21.plot(t * 1e3, sig_clean, color="black")
    ax21.set_xlabel(r"$t [ms]$", color="black")
    ax21.set_ylabel(r"$U [V]$", color="black")
    ax21.tick_params(axis="y", colors="black")
    ax21.set_title("Clean signal sampling")

    ax22 = ax21.twinx()
    ax22.plot(t * 1e3, adc(sig_clean, bits, vref), color="red")
    ax22.set_ylabel(r"$ADC_{OUT}$", color="red")
    ax22.tick_params(axis="y", colors="red")

    for line in np.linspace(0, vref, 2**bits + 1):
        ax21.axhline(line, color="blue", linestyle="--", linewidth=1, alpha=0.5)

    ax21.set_ylim(np.min(sig_clean - LSB), np.max(sig_clean + LSB))
    ax22.set_ylim(np.min(sig_clean - LSB) * (2**bits) / vref, np.max(sig_clean + LSB) * (2**bits) / vref)

    ax23.plot(t * 1e3, sig_white, color=(0.2, 0.2, 0.2), alpha=0.8)
    ax23.plot(t * 1e3, sig_clean, color="black")
    ax23.set_xlabel(r"$t [ms]$", color="black")
    ax23.set_ylabel(r"$U [V]$", color="black")
    ax23.tick_params(axis="y", colors="black")
    ax23.set_title("Noisy signal sampling")

    ax24 = ax23.twinx()
    ax24.step(t * 1e3, adc(sig_white, bits, vref), color="red", alpha=0.8)
    ax24.set_ylabel(r"$ADC_{OUT}$", color="red")
    ax24.tick_params(axis="y", colors="red")

    ax23.set_ylim(np.min(sig_white - LSB / 2), np.max(sig_white + LSB / 2))
    ax24.set_ylim(np.min(sig_white - LSB / 2) * (2**bits) / vref, np.max(sig_white + LSB / 2) * (2**bits) / vref)

    for line in np.linspace(0, vref, 2**bits + 1):
        ax23.axhline(line, color="blue", linestyle="--", linewidth=1, alpha=0.5)

    plt.tight_layout()

    # ############################# FIGURE 3 #############################
    fig2, ax31 = plt.subplots()

    ax31.plot(t * 1e3, adc(sig_white, bits, vref) * 4, color="black", alpha=0.5, label=r"$ADC_{OUT}$", marker="o", markersize=3, linestyle="None")
    ax31.plot(t_osr_cl * 1e3, avg_osr_cl, color="blue", label="Summed noiseless samples", marker="o", markersize=5, linestyle="None")
    ax31.plot(t_osr * 1e3, avg_osr, color="red", alpha=0.5, label="Summed noisy samples", marker="o", markersize=5, linestyle="None")
    ax31.set_xlabel(r"$t [ms]$", color="black")
    ax31.set_ylabel(r"$LSB$", color="black")
    ax31.tick_params(axis="y", colors="black")
    ax31.legend(loc="lower right")
    ax31.set_title("Clean signal sampling")
    ax31.grid()

    plt.tight_layout()
    plt.show()
