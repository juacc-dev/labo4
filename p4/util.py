import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import pylabo.plot as plot

ERROR_RELATIVE = 0.75 / 100.0
ERROR_FIXED = 2.2

channels = [
    "canal 101",
    "canal 102",
    "canal 103",
    "canal 104",
    "canal 105",
    # "canal 107",
]

dist = np.array([
    81.4,
    123.1,
    164.0,
    211.9,
    249.6,
    # 410.5,
])


def setup_matplotlib():
    plt.rcParams["font.family"] = "Noto Sans"
    plt.rcParams["font.size"] = 16
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['axes.grid'] = True
    plt.rcParams["figure.figsize"] = (9, 6)
    plt.rcParams['legend.loc'] = "best"


def to_pylabo(df, ch=None):
    tiempo = df["Time"]
    temp = df["Value"]

    temp_err = pd.Series(
        np.maximum(ERROR_RELATIVE * temp, ERROR_FIXED),
        index=temp.index
    )

    if not (df["Alert"] == 0.0).all():
        print(f"Alert for {ch}")

    pylabo_df = pd.DataFrame({
        "Time": tiempo,
        "Error Time": pd.Series(),
        "Temperature": temp,
        "Error Temperature": temp_err
    })

    return pylabo_df


def make_dfs(path):
    path = Path(path)
    dfs = {}

    for ch in channels:
        stem = ch + ".csv"
        df_raw = pd.read_csv(path / stem)

        dfs[ch] = to_pylabo(df_raw)

    return dfs


def plot_dfs(dfs, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    for ch, df in dfs.items():
        plot.data(df, ax=ax, label=ch)

    ax.set(
        xlabel="Tiempo [s]",
        ylabel="Temperatura [C]",
        title=title
    )

    return ax
