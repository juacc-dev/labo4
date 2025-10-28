lockin = SR830("GPIB0::8::INSTR")

print(f"Conectado al Lock-in: {lockin.query('*IDN?')}")

Q = 100  # Factor de calidad
FILTER_SLOPE = 12
AMPLITUDE = 1  # volts

FREQ_MIN = 3e1
FREQ_MAX = 2e3
N_SAMPLES = 250

freq = np.linspace(FREQ_MIN, FREQ_MAX, N_SAMPLES)
freq = freq[::-1]

lockin.set_internal_ref_cfg(voltage=AMPLITUDE)
lockin.set_use_internal_ref(True)
lockin.set_filter_cfg(slope=FILTER_SLOPE)

lockin.set_input("A-B")

df_v = measure(lockin, freq, Q=Q)

filepath = PATH / f"({count}) data.csv"
count += 1

df.to_csv(filepath, index=False)
print(f"Se guardó {filepath}")

del lockin
