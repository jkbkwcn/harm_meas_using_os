import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

filename1 = "python_sim\data1.txt"
filename2 = "python_sim\data2.txt"
filename3 = "python_sim\data3.txt"

data1 = pd.read_csv(filename1, sep='\t', names=["t", "ADC", "TRI"])
data2 = pd.read_csv(filename2, sep='\t', names=["t", "ADC", "TRI"])
data3 = pd.read_csv(filename3, sep='\t', names=["t", "ADC", "TRI"])

# Line interpolation to use in the model with potentially diffrent time step
data1_int = interp1d(data1.t, data1.ADC,  kind='linear',
                    bounds_error=False, fill_value=(0, 0))
data2_int = interp1d(data2.t, data2.ADC,  kind='linear',
                    bounds_error=False, fill_value=(0, 0))
data3_int = interp1d(data3.t, data3.ADC,  kind='linear',
                    bounds_error=False, fill_value=(0, 0))

t_int = np.linspace(data1.t[0], data1.t[len(data1.t) - 1], int(1e6))

plt.figure(1)
plt.subplot(321)
plt.title(r"Output voltage histogram vs anti-aliasing filter $f_{3dB}$")
plt.hist(data1_int(t_int), bins=50, color='blue')
plt.grid()

plt.subplot(322)
plt.title(r"Output waveform vs anti-aliasing filter $f_{3dB}$")
plt.plot(t_int[:20000] * 1e3, data1_int(t_int)[:20000], 'b', label=r"$f_{3dB} \approx 72 kHz$")
plt.grid()
plt.ylabel(r"Voltage [V]")
plt.legend(loc="lower right")

plt.subplot(323)
plt.hist(data2_int(t_int), bins=50, color='red')
plt.grid()

plt.subplot(324)
plt.plot(t_int[:20000] * 1e3, data2_int(t_int)[:20000], 'r', label=r"$f_{3dB} \approx 16 kHz$")
plt.grid()
plt.ylabel(r"Voltage [V]")
plt.legend(loc="lower right")

plt.subplot(325)
plt.hist(data3_int(t_int), bins=50, color='black')
plt.grid()
plt.xlabel(r"Voltage [V]")

plt.subplot(326)
plt.plot(t_int[:20000] * 1e3, data3_int(t_int)[:20000], 'k', label=r"$f_{3dB} \approx 4.8 kHz$")
plt.grid()
plt.xlabel(r"t [ms]")
plt.ylabel(r"Voltage [V]")
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()


