channel_name = "Dev4/ai1"
duration = 5
freq = 10000

df_raw = measure(
    channel_name,
    duration,
    freq
)

start = where_start(df_raw, START_ZERO, ZERO_TOLERANCE)
df_skipped = df_raw[:start]
df = df_raw[start:]

xf, yf = fourier_transform(df, skip_first=SKIP_SPECTRUM)
xf, yf = low_pass_filter(xf, yf, FREQ_MAX)

peaks = find_highest_peaks(xf, yf, N_PEAKS, separation=PEAK_SEP)


fig, (ax_data, ax_fft) = plt.subplots(
    2, 1,
    figsize=(9, 7),
    height_ratios=[1, 2]
)

plot_data(
    df,
    ax=ax_data,
    label="Vibración",
)

if start is not None:
    plot_data(
        df_skipped_first,
        ax=ax_data,
        label="Ignorado",
        color="grey",
        alpha=0.6
    )

ax_data.set(
    xlabel="Tiempo [s]",
    ylabel="Amplitud [V]"
)

plot_fft(xf, yf, peaks, ax=ax_fft)

fig.tight_layout()
plt.show()

# %%

name = ""

if name == "":
    name = str(count)
    count += 1

df_raw.to_csv(PATH / f"{name}.csv", index=False)

print(name)

# %%

delta_t = 1 / xf[peaks[0]]
delta_i = int(delta_t * sec_to_pt * 2)

pos_peaks, _ = scipy.signal.find_peaks(
    df["Voltaje [V]"],
    distance=delta_i
)
neg_peaks, _ = scipy.signal.find_peaks(
    -df["Voltaje [V]"],
    distance=delta_i
)

df_pos, fit_pos = peak_fit(df, pos_peaks)
df_neg, fit_neg = peak_fit(df, neg_peaks, p0=[-1, 1, 1])

print(fit_pos.report())
print(fit_neg.report())

plot_decay(
    df,
    df_pos,
    df_neg,
    fit_pos,
    fit_neg
)

plt.show()
