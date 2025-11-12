import pandas as pd
import numpy as np
from pathlib import Path
import pylabo.plot as plot
import pylabo.fit as fit
import matplotlib.pyplot as plt
import re

from util import value_and_error, transferencia, even_taylor_cut

dir = Path("./data/3 - 29-10/barrido en voltaje")

plt.rcParams['legend.fontsize'] = 12
plt.rcParams["figure.figsize"] = (7, 5)


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


voltage = find_voltages(dir)

transf = []
# transf_err = []
out = []
out_err = []
freq = None

# fig, ax = plt.subplots(1, 1)

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

    out.append(v_out)
    out_err.append(v_out_err)


freq = np.array(freq)


def find_resonance(transf, voltage, n, win):
    param_str = ["x0", "a0"]
    p0 = [50096.0, 0.485043]
    # p0.extend([0 for _ in range(1, n+1)])

    for i in range(1, n+1):
        p0.append(0)
        param_str.append(f"a{i*2}")

    model = fit.Function(
        even_taylor_cut,
        param_str
    )

    x0 = []
    x0_err = []

    for df_t, volt in zip(transf, voltage):
        idx_max = df_t["Transferencia"].idxmax()
        # f_max = freq[idx_max]
        # t_max = df_t["Transferencia"].max()

        # win = n_points // 2
        df_t = df_t[idx_max - win:idx_max + win]

        fit_func = fit.fit(
            model,
            df_t,
            p0=p0
        )
        # p0 = fit_func.param_val

        x0.append(fit_func.param_val[0])
        x0_err.append(fit_func.param_err[0])

    print(fit_func.report())
    fig, ax = plot.datafit(df_t, fit_func, datalabel=f"{volt}V")
    # ax[0].set_title(f"Orden {2*n}, {2*win + 1} puntos")
    plt.tight_layout()
    plt.savefig(f"plots/{volt}V.png")
    plot.show()

    x0 = np.array(x0) * 1000
    x0_err = np.array(x0_err) * 1000
    return x0, x0_err


N = 2
M = 5

n = N
win = M

x0, x0_err = find_resonance(transf, voltage, n, win)


fig, ax = plt.subplots(1, 1)

ax.errorbar(voltage, x0, yerr=x0_err, fmt='o', capsize=10)

for volt, x, err in zip(voltage, x0, x0_err):
    print(f"{volt:.2f}: {x/1000:e} +- {err/1000:e}")

ax.set(
    xlabel="Tensión de la fuente [V]",
    ylabel="Resonancia [mHz]"
)

# ax.legend()
plt.tight_layout()
plt.savefig("barrido.png")
plt.show()
