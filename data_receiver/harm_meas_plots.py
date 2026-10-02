import serial
import time
import numpy as np
import matplotlib.pyplot as plt

# --- KONFIGURACJA ---
PORT = 'COM3'
BAUD = 115200
BYTES_TO_READ = 2048 * 100 # Musi być liczbą parzystą, aby poprawnie stworzyć uint16
OUTPUT_FILE = "odebrane_dane.txt"
measure_and_write = False


def format_as_hex(data):
    hex_string = ""
    for i in range(0, len(data), 16):
        chunk = data[i:i + 16]
        line = " ".join(f"{b:02X}" for b in chunk)
        hex_string += f"{i:04X}: {line}\n"
    return hex_string


def process_and_plot(raw_bytes, r=4, fs_hz=288000):
    if len(raw_bytes) % 2 != 0:
        raw_bytes = raw_bytes[:-1]

    # 1. Konwersja i scalanie do uint16
    data_u8 = np.frombuffer(raw_bytes, dtype=np.uint8)
    data_u16 = (data_u8[1::2].astype(np.uint16) << 8) | data_u8[0::2].astype(np.uint16)

    # 2. Usuwanie r najstarszych bitów i składowej stałej
    mask = (1 << (16 - r)) - 1
    signal = (data_u16 & mask).astype(np.float64)
    signal -= np.mean(signal)

    # 3. Okno czasowe (np. Hamming) - poprawia dokładność FFT, zapobiega "wyciekowi"
    window = np.hamming(len(signal))
    signal_windowed = signal * window

    # 4. Obliczenie FFT
    num_samples = len(signal)
    fft_values = np.fft.rfft(signal_windowed)
    fft_freqs = np.fft.rfftfreq(num_samples, 1 / fs_hz)

    # Amplituda w dB (znormalizowana do 0 dB jako max)
    amplitude_db = 20 * np.log10(np.abs(fft_values) + 1e-9)
    amplitude_db -= np.max(amplitude_db)

    # --- OKNO 1: Pełny zakres (Czas + Pełne FFT) ---
    fig1, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(12, 7))
    fig1.canvas.manager.set_window_title('Analiza Pełnozakresowa')

    ax_t.plot(np.arange(num_samples) / fs_hz * 1000, signal, linewidth=0.7)
    ax_t.set_title("Przebieg czasowy (bez DC)")
    ax_t.set_xlabel("Czas [ms]")
    ax_t.grid(True)

    ax_f.plot(fft_freqs / 1000, amplitude_db, color='red', linewidth=0.7)
    ax_f.set_title("Pełne widmo FFT (0 - 144 kHz)")
    ax_f.set_xlabel("Częstotliwość [kHz]")
    ax_f.set_ylabel("Amplituda [dB]")
    ax_f.grid(True)

    plt.tight_layout()

    # --- OKNO 2: Zoom na niskie częstotliwości ---
    fig2, (ax_z1, ax_z2) = plt.subplots(2, 1, figsize=(12, 7))
    fig2.canvas.manager.set_window_title('Zoom: Niskie Częstotliwości')
    plt.subplots_adjust(hspace=0.4)

    # Zakres 0 - 1 kHz
    ax_z1.plot(fft_freqs, amplitude_db, color='blue')
    ax_z1.set_xlim(0, 500)  # Zakres w Hz
    ax_z1.set_ylim(np.max(amplitude_db) - 60, np.max(amplitude_db) + 5)
    ax_z1.set_title("Widmo: Zakres 0 - 1000 Hz")
    ax_z1.set_xlabel("Częstotliwość [Hz]")
    ax_z1.set_ylabel("Amplituda [dB]")
    ax_z1.grid(True, which='both')

    # Zakres 0 - 10 kHz
    ax_z2.plot(fft_freqs / 1000, amplitude_db, color='green')
    ax_z2.set_xlim(0, 10)  # Zakres w kHz
    ax_z2.set_ylim(np.max(amplitude_db) - 60, np.max(amplitude_db) + 5)
    ax_z2.set_title("Widmo: Zakres 0 - 10 kHz")
    ax_z2.set_xlabel("Częstotliwość [kHz]")
    ax_z2.set_ylabel("Amplituda [dB]")
    ax_z2.grid(True, which='both')

    plt.tight_layout()
    plt.show()

    return signal, fft_freqs, amplitude_db


def measure_speed():
    try:
        with serial.Serial(PORT, BAUD, timeout=5) as ser:
            print(f"Otwarto port {PORT}. Czekam na dane...")
            ser.reset_input_buffer()

            received_data = bytearray()

            while ser.in_waiting == 0:
                pass

            start_time = time.perf_counter()

            while len(received_data) < BYTES_TO_READ:
                if ser.in_waiting > 0:
                    chunk = ser.read(ser.in_waiting)
                    received_data.extend(chunk)
                    progress = (len(received_data) / BYTES_TO_READ) * 100
                    print(f"Postęp: {progress:.1f}% ({len(received_data)} B)", end='\r')

            duration = time.perf_counter() - start_time

            print(f"\n\n--- WYNIKI TESTU ---")
            print(f"Odebrano: {len(received_data) / 1024:.2f} KB w {duration:.4f} s")
            print(f"--------------------\n")

            # Zapis do pliku (HEX)
            if (measure_and_write):
                with open(OUTPUT_FILE, "w") as f:
                    f.write(format_as_hex(received_data))

            # Przetwarzanie i wykres
            return received_data

    except serial.SerialException as e:
        print(f"Błąd portu: {e}")
        return None


if __name__ == "__main__":
    raw_data = measure_speed()

    if raw_data:
        # Wywołanie konwersji i rysowania
        final_values = process_and_plot(raw_data)

        # Opcjonalne sprawdzenie pierwszych 5 wartości w konsoli
        print("Pierwsze 5 wartości uint16:", final_values[:5])