import serial
import struct
import numpy as np
import matplotlib.pyplot as plt


def os_tri_avg(t: np.ndarray[float], sig: np.ndarray[int], OSR: int) -> tuple[np.ndarray[float], np.ndarray[int]]:
    num_chunks = len(sig) // OSR

    avg_t, avg_sig = np.zeros(num_chunks), np.zeros(num_chunks, dtype=int)
    for i in range(num_chunks):
        start_idx = i * OSR
        end_idx = (i + 1) * OSR
        avg_t[i] = np.mean(t[start_idx:end_idx])
        avg_sig[i] = (np.sum(sig[start_idx:end_idx]) + 1) // 2
    return avg_t, avg_sig


# ================= KONFIGURACJA =================
PORT = "COM7"  # zmień
BAUD = 115200  # bez znaczenia dla USB CDC
N_FRAMES = 500  # ILOŚĆ PEŁNYCH RAMEK DO ZEBRANIA
f_s = 128_571.4  # częstotliwość próbkowania ADC
# ================================================

SYNC = b"\xa5\x5a"
HDR_FMT = "<BBHH"
HDR_SIZE = struct.calcsize(HDR_FMT)


class USBFrameParser:
    def __init__(self):
        self.buf = bytearray()

    def feed(self, data: bytes):
        self.buf.extend(data)
        frames = []

        while True:
            pos = self.buf.find(SYNC)
            if pos < 0:
                self.buf = self.buf[-1:]
                break

            if len(self.buf) < pos + HDR_SIZE:
                break

            sync0, sync1, frame_id, length = struct.unpack(HDR_FMT, self.buf[pos : pos + HDR_SIZE])

            end = pos + HDR_SIZE + length
            if len(self.buf) < end:
                break

            payload = self.buf[pos + HDR_SIZE : end]
            frames.append((frame_id, payload))
            self.buf = self.buf[end:]

        return frames


def main():
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    parser = USBFrameParser()

    collected = []
    frames_cnt = 0

    print(f"Zbieram {N_FRAMES} pełnych ramek...")

    while frames_cnt < N_FRAMES:
        data = ser.read(4096)
        if not data:
            continue

        frames = parser.feed(data)

        for frame_id, payload in frames:
            samples = np.frombuffer(payload, dtype=np.uint16)
            collected.append(samples)
            frames_cnt += 1

            print(f"Ramka {frame_id} | {frames_cnt}/{N_FRAMES}")

            if frames_cnt >= N_FRAMES:
                break

    ser.close()

    signal = np.concatenate(collected, dtype=int)
    t = np.arange(len(signal)) / f_s
    n = 12
    p = 6
    OSR = 2 * 2**p
    f_os = f_s / OSR

    avg_t, avg_sig = os_tri_avg(t, signal, OSR)

    # ====== WYKRES ======
    plt.figure(figsize=(12, 5))
    plt.plot(t, signal * (3.3 / 2**n), color="blue", linewidth=0.7, marker="o")
    plt.plot(avg_t, avg_sig * (3.3 / 2 ** (n + p)), color="red", linewidth=1.5, marker="o")
    plt.title(f"ADC – {N_FRAMES} ramek ({len(signal)} próbek)")
    plt.xlabel("Czas [s]")
    plt.ylabel("Napięcie [V]")
    plt.grid(True)
    plt.tight_layout()

    # fft
    n = len(avg_sig)
    fft_vals = np.fft.rfft(avg_sig - np.mean(avg_sig))
    fft_freqs = np.fft.rfftfreq(n, 1 / f_os)
    amplitude_db = 20 * np.log10(np.abs(fft_vals) / n + 1e-9)
    amplitude_db -= np.max(amplitude_db)

    plt.figure(figsize=(12, 5))
    plt.plot(fft_freqs / 1000, amplitude_db, color="red", linewidth=0.7)
    plt.title("Widmo FFT")
    plt.xlabel("Częstotliwość [kHz]")
    plt.ylabel("Amplituda [dB]")
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
