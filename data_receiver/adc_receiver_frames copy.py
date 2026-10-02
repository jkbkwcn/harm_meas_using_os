import serial
import struct
import numpy as np
import matplotlib.pyplot as plt

PORT = 'COM3'

BAUD = 115200
READ_CHUNK = 8*1030

SYNC = b'\xA5\x5A'
HDR_SIZE = 6

N_FRAMES = 2

samples = np.empty(N_FRAMES*512, dtype=np.uint16)
samples_test = []
frame_idx = 0

rx = bytearray()
lost_frames = 0
last_frame_id = None
with serial.Serial(PORT, BAUD, timeout=1) as ser:

    while frame_idx < N_FRAMES:
        print(f"Pobieram {READ_CHUNK} bajtów z portu szeregowego...")
        data = ser.read(READ_CHUNK)
        # # print raw hex data for debugging
        # print("Odebrano dane (hex):", data.hex())
          
        if not data:
            continue

        rx = data

        same_data = False
        while True:
            if same_data:
                print("Szukam innych ramek w buforze...")

            i = rx.find(SYNC)
            if i < 0:
                break
            
            print("Znaleziono SYNC na pozycji", i)

            if len(rx) < i + HDR_SIZE:
                print("\t\tW danych brak pełnego nagłówka, czytam nowe...")
                break

            hdr = rx[i:i+HDR_SIZE]
            _, _, frame_id, length = struct.unpack('<BBHH', hdr)
            
            if last_frame_id is not None and frame_id != (last_frame_id + 1):
                print(f"\t\tWykryto utratę ramek! Oczekiwano ID={((last_frame_id + 1))}, otrzymano ID={frame_id}")
            
            last_frame_id = frame_id
            print("\t\tOdebrano nagłówek (hex):", hdr.hex(" ", 1))
            print(f"\t\tOdebrano nagłówek ID={frame_id}, długość={length} B")

            total_len = HDR_SIZE + length
            if len(rx) < i + total_len:
                print("\t\tW danych brak pełnego frame, czytam nowe...")
                break

            print("Ramka kompletna, przetwarzam dane...")
            payload = rx[i+HDR_SIZE:i+total_len]
            frame_samples = struct.unpack('<{}H'.format(length//2), payload)
            samples_test.extend(frame_samples)
            

            # take = min(len(frame_samples), N_FRAMES - frame_idx)
            # samples[frame_idx:frame_idx+take] = frame_samples[:take]
            # frame_idx += take

            rx = rx[i+total_len:]
            frame_idx += 1
            same_data = True
            if len(rx) < 1030:
                print("Brak dalszych danych w buforze.")
                break

    print("Zbieranie zakończone")

# PORT USB ZAMKNIĘTY TUTAJ


t = np.arange(len(samples)) / 128_114  # czas [s]

plt.figure()
t_test = np.arange(len(samples_test)) / 128_114  # czas [s]
plt.plot(t_test, samples_test)
plt.xlabel("Czas [s]")
plt.ylabel("ADC (V)")
plt.title("Przebieg czasowy ADC")
plt.grid(True)
plt.show()

