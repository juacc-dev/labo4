import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams['axes.grid'] = True
plt.rcParams["figure.figsize"] = (9, 7)
plt.rcParams['legend.loc'] = "best"
plt.rcParams['legend.fontsize'] = 11

path = Path("./data/2 - 22-10")

# files = [
#     "(0) out_fine.csv",
#     "(1) out_fine (extendido).csv",
#     "(2) out_grueso (pre).csv",
#     "(3) out_grueso (post).csv",
#     "(5) in_grueso (post).csv",
#     "(6) in_fine.csv",
#     "(7) in_grueso (pre).csv",
#     "(8) in_grueso (pre).csv",
# ]

colors = [
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'tab:cyan',
]


files_out = [
    "(0) out_fine.csv",
    "(1) out_fine (extendido).csv",
    "(2) out_grueso (pre).csv",
    "(3) out_grueso (post).csv",
]

files_in = [
    "(5) in_grueso (post).csv",
    "(6) in_fine.csv",
    # "(7) in_grueso (pre).csv",
    "(8) in_grueso (pre).csv",
]

fig, ax = plt.subplots(2, 1, sharex=True)

for i in range(len(files_in)):
    df = pd.read_csv(path / files_in[i])
    freq = df["Frecuencia [Hz]"]

    rg = (float(freq.iat[0]), float(freq.iat[-1]))

    n = len(files_in)
    ax[0].axvspan(
        rg[0],
        rg[1],
        ymin=i/n,
        ymax=(i+1)/n,
        label=files_in[i],
        color=colors[i]
    )
    ax[0].legend()

for i in range(len(files_out)):
    df = pd.read_csv(path / files_out[i])
    freq = df["Frecuencia [Hz]"]

    rg = (float(freq.iat[0]), float(freq.iat[-1]))

    n = len(files_out)
    ax[1].axvspan(
        rg[0],
        rg[1],
        ymin=i/n,
        ymax=(i+1)/n,
        label=files_out[i],
        color=colors[i]
    )
    ax[1].legend()

plt.show()
