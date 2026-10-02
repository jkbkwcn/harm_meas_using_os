import numpy as np
from matplotlib.pylab import plt


def calculate_adc_metrics(freq_fft, sig_fft, num_spurs_to_reject=10):
    """
    Oblicza SINAD, SNR i ENOB analizując widmo mocy.
    Automatycznie znajduje sygnał podstawowy i spursy, wycinając je dynamicznie.
    """
    # 1. Obliczenie mocy (skalowanie do max dla stabilności numerycznej)
    sig_pwr = np.abs(sig_fft) ** 2
    n_samples = len(sig_pwr)

    # Kopia do pracy, z której będziemy "wycinać" sygnały
    working_spec = np.copy(sig_pwr)
    # Usuwamy DC (pierwszy prążek), aby nie został uznany za sygnał/spur
    working_spec[0] = 0

    plt.figure(9)
    plt.plot(freq_fft, np.abs(sig_fft))
    plt.grid()
    plt.legend()

    plt.figure(10)
    plt.plot(freq_fft, working_spec)
    plt.grid()
    plt.legend()

    def extract_peak_and_clear(freq, spec):
        """Znajduje max, sumuje energię prążka i usuwa go ze spektrum."""
        idx_max = np.argmax(spec)
        max_val = spec[idx_max]
        print(f"Found peak at index {idx_max} with value {max_val}, at frequency {freq[idx_max]} Hz")

        # Sumujemy energię (całka pod pikiem)
        peak_energy = max_val
        spec[idx_max] = 0

        print("Expanding peak energy calculation...")
        BW = 1000   # liczba binów na stronę (do dobrania!)
        low = max(0, idx_max - BW)
        high = min(len(working_spec), idx_max + BW + 1)

        peak_energy = np.sum(working_spec[low:high])
        working_spec[low:high] = 0

        return peak_energy, idx_max

    # 2. Znalezienie częstotliwości podstawowej (Fundamental)
    # Szukamy po prostu największego piku w całym spektrum (poza DC)
    print("Finding Fundamental Frequency...")
    p_fundamental, idx_fundamental = extract_peak_and_clear(freq_fft, working_spec)

    plt.figure(11)
    plt.plot(freq_fft, working_spec)
    plt.grid()
    plt.legend()

    # 3. Znalezienie Spursów (10 najwyższych pozostałych sygnałów)
    spurs_energies = []
    for _ in range(num_spurs_to_reject):
        if np.max(working_spec) <= 0:
            break
        spur_pwr, _ = extract_peak_and_clear(freq_fft, working_spec)
        spurs_energies.append(spur_pwr)

    p_spurs_total = sum(spurs_energies)

    plt.figure(12)
    plt.plot(freq_fft, working_spec)
    print("Working spec shape: ", working_spec.shape)
    plt.grid()

    # 4. Pozostała moc to szum (Noise Floor)
    # Po wycięciu piku głównego i spursów, w working_spec zostały tylko szumy
    p_noise_floor = np.sum(working_spec)

    # 5. Obliczenie parametrów (Moc sygnału / (Moc szumu + Moc zniekształceń))
    # SINAD: Sygnał / (Szum + Spursy)
    sinad = 10 * np.log10(p_fundamental / (p_noise_floor + p_spurs_total))

    # SNR: Sygnał / Szum (bez spursów i harmonicznych)
    snr = 10 * np.log10(p_fundamental / p_noise_floor)

    # ENOB: (SINAD - 1.76) / 6.02
    enob = (sinad - 1.76) / 6.02

    return {"SINAD": sinad, "SNR": snr, "ENOB": enob, "Fundamental_Pwr": p_fundamental, "Spurs_Sum_Pwr": p_spurs_total, "Noise_Floor_Pwr": p_noise_floor}
