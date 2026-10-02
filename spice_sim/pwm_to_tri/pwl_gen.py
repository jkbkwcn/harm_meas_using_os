import numpy as np

# ==========================
# Configuration Parameters
# ==========================

t_start = 0.0
f_pwm = 1e6             # 10 kHz PWM frequency
T_pwm = 1 / f_pwm

cycles_per_triangle = 1000  # one full up–down sweep
num_triangles = 4          # number of triangle repetitions
num_cycles = cycles_per_triangle * num_triangles

duty_start = 0.0001
duty_end = 0.9999

v_low = 0.0
v_high = 3.3
trise = 1e-12              # small rise/fall time for LTspice stability

output_file = "pwm_ramp_triangle.txt"

# ==========================
# Generate duty triangle
# ==========================

half_cycles = cycles_per_triangle // 2

# One triangle shape (up + down)
duty_triangle = np.concatenate([
    np.linspace(duty_start, duty_end, half_cycles, endpoint=False),
    np.linspace(duty_end, duty_start, half_cycles, endpoint=False)
])

# Repeat this pattern to fill total number of cycles
duties = np.tile(duty_triangle, num_triangles)

# Time axis for cycle start points
cycle_times = np.arange(t_start, t_start + num_cycles * T_pwm, T_pwm)
t_end = cycle_times[-1] + T_pwm

# ==========================
# Generate PWM PWL data
# ==========================

pwl_data = []

for t_cycle, duty in zip(cycle_times, duties):
    t_on = duty * T_pwm
    t_off = t_cycle + t_on

    # Rising edge
    pwl_data.append((t_cycle, v_low))
    pwl_data.append((t_cycle + trise, v_high))

    # Falling edge
    pwl_data.append((t_off, v_high))
    pwl_data.append((t_off + trise, v_low))

# Ensure end of waveform
if pwl_data[-1][0] < t_end:
    pwl_data.append((t_end, v_low))

# ==========================
# Write to file
# ==========================

with open(output_file, "w") as f:
    for t, v in pwl_data:
        f.write(f"{t:.9e} {v:.9e}\n")

# ==========================
# Info
# ==========================

print(f"PWL file '{output_file}' generated successfully!")
print(f"  PWM Frequency: {f_pwm/1e3:.1f} kHz")
print(f"  Duty ramp: {duty_start*100:.1f}% → {duty_end*100:.1f}% → {duty_start*100:.1f}%")
print(f"  Triangular pattern repeats: {num_triangles} times")
print(f"  Total cycles: {num_cycles}")
print(f"  Total time: {t_end*1e6:.3f} µs")
print(f"  Total PWL points: {len(pwl_data)}")
