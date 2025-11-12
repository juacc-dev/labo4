import pandas as pd
import numpy as np
from pathlib import Path
import pylabo.plot as plot
import pylabo.fit as fit
import matplotlib.pyplot as plt
import re
from scipy.optimize import curve_fit

from util import value_and_error  # , make_df

color = "#fb4934"

dir = Path("./data/3 - 29-10/barrido en voltaje")


def find_voltages(dir):
    voltages = []

    pattern = re.compile(r"\(\d{1,2}\) in \((?P<voltage>.*)V\)")

    for file in dir.glob("*.csv"):
        x = pattern.search(file.stem)

        if not x:
            continue

        voltages.append(float(x.group("voltage")))

    return np.array(sorted(voltages))


def find_files(dir, voltage):
    file_in: str
    file_out: str

    pattern_in = re.compile(f".* in \\({voltage}V\\)")
    pattern_out = re.compile(f".* out \\({voltage}V\\)")

    for file in dir.glob("*.csv"):
        if pattern_in.match(file.stem):
            file_in = file

        if pattern_out.match(file.stem):
            file_out = file

    return file_in, file_out


def transferencia(df_in, df_out):
    freq = df_in["Frecuencia [Hz]"]

    v_in, err_in = value_and_error(df_in)
    v_out, err_out = value_and_error(df_out)

    t = np.abs(v_out) / np.abs(v_in)

    delta_va = 2 * np.sqrt(
        (np.real(v_out) * np.real(err_out)) ** 2 *
        (np.imag(v_out) * np.imag(err_out)) ** 2
    )

    delta_vb = 2 * np.sqrt(
        (np.real(v_in) * np.real(err_in)) ** 2 *
        (np.imag(v_in) * np.imag(err_in)) ** 2
    )

    terr = t * np.sqrt(
        (delta_va / np.abs(v_out)) ** 2 + (delta_vb / np.abs(v_in)) ** 2
    )

    return pd.DataFrame({
        "Frecuencia [Hz]": freq,
        "Error Frecuencia [Hz]": pd.Series(),
        "Transferencia": t,
        "Error": terr
    })


voltage = find_voltages(dir)

transf = []
# transf_err = []
out = []
out_err = []
freq = None


def get_abs_err(v, v_err):
    x = np.real(v)
    y = np.imag(v)
    xerr = np.real(v_err)
    yerr = np.imag(v_err)

    return np.sqrt(((x * xerr) ** 2 + (y * yerr) ** 2) / (x ** 2 + y ** 2))


for volt in voltage:
    file_in, file_out = find_files(dir, volt)
    df_in = pd.read_csv(file_in)
    df_out = pd.read_csv(file_out)

    v_out, v_out_err = value_and_error(df_out)

    freq = df_out["Frecuencia [Hz]"]

    df_t = transferencia(df_in, df_out)
    transf.append(df_t)
    # transf.append(df_t["Transferencia"])
    # transf_err.append(df_t["Error"])

    out.append(np.abs(v_out))
    out_err.append(get_abs_err(v_out, v_out_err))


freq = np.array(freq)

out = np.array(out)
out_err = np.array(out_err)

freq_fit = np.zeros_like(voltage)
out_fit = np.zeros_like(voltage)
out_fit_err = np.zeros_like(voltage)

for i in range(voltage.size):
    idx_max = np.argmax(out[i])

    freq_fit[i] = freq[idx_max]
    out_fit[i] = out[i][idx_max]
    out_fit_err[i] = out_err[i][idx_max]

df_peaks = pd.DataFrame({
    "Tensión de la fuente [V]": voltage,
    "Error voltage [V]": None,
    "Salida [V]": out_fit,
    "Error [V]": out_fit_err
})

fit_func = fit.fit(
    fit.funs.linear,
    df_peaks
)

plt.rcParams["figure.figsize"] = (7, 5)
plt.rcParams["font.size"] = 11
plt.rcParams['legend.fontsize'] = 11

print(fit_func.report())
plot.datafit(
    df_peaks,
    fit_func,
    fmt="o",
    data_color=color,
    fit_color="black",
    datalabel="Picos",
    fitlabel="Ajuste lineal",
    height_ratios=[2, 1]
)
plt.tight_layout()
plt.savefig("2d.png")
plot.show()

x, y = np.meshgrid(freq, voltage)

# plt.rcParams["figure.figsize"] = (8, 6)

# plt.rcParams["font.size"] = 13
# plt.rcParams['legend.fontsize'] = 13

fig = plt.figure()
ax = plt.axes(projection="3d", computed_zorder=False)

ax.plot3D(
    freq_fit,
    voltage,
    out_fit,
    color="black",
    zorder=2,
    label="Ajuste lineal",
    alpha=0.9
)
ax.plot(
    freq_fit,
    voltage,
    out_fit,
    'o',
    zorder=3,
    color=color,
    label="Picos"
)
ax.plot_surface(
    x,
    y,
    np.abs(out),
    cmap="cool",
    alpha=0.9,
    zorder=1,
    label="Espectro en función de la tensión"
)

ax.set(
    xlabel="Frecuencia [Hz]",
    ylabel="Tensión de la fuente [V]",
    zlabel="Salida [V]"
)
plt.legend()
plt.tight_layout()
plt.savefig("3d.png")
plt.show()
