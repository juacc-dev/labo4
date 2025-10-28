lockin = SR830("GPIB0::8::INSTR")

print(f"Conectado al Lock-in: {lockin.query('*IDN?')}")

# %%

Q = 100  # Factor de calidad
FILTER_SLOPE = 12
AMPLITUDE = 1  # volts

FREQ_MIN = 3e1
FREQ_MAX = 2e3
N_SAMPLES = 500

freq = np.linspace(FREQ_MIN, FREQ_MAX, N_SAMPLES)
lockin.set_internal_ref_cfg(voltage=AMPLITUDE)
lockin.set_use_internal_ref(True)
lockin.set_filter_cfg(slope=FILTER_SLOPE)

# %%

MATERIAL = "cobre"
GEOMETRIA = "grande"

# Voltaje (A-B)

lockin.set_input("A-B")
df_v = measure(lockin, freq, Q=Q)

filepath = PATH / f"con {MATERIAL} {GEOMETRIA} A-B.csv"
df_i.to_csv(filapath)
print(f"Se guardó {filepath}")

plot_measure(df_i)

# %%

# Corriente (A)

lockin.set_input("A")
df_i = measure(lockin, freq, Q=Q)

filepath = PATH / f"con {MATERIAL} {GEOMETRIA} I.csv"
df_i.to_csv(filepath)
print(f"Se guardó {filepath}")

plot_measure(df_i)
