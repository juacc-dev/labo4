def peak_fit(
    df,
    peaks,
    p0=[1, 1, 1]
):
    x, _, y, y_err = split_dataframe(df)

    df_peaks = pd.DataFrame({
        "x": x.iloc[peaks],
        "x_err": pd.Series(),
        "y": y.iloc[peaks],
        "y_err": y_err.iloc[peaks],
    })

    model = pylabo.fit.Function(
        lambda x, a, k, c: a * np.exp(-k * x) + c,
        ["A", "k", "C"]
    )

    fit_func = pylabo.fit.fit(
        model,
        df_peaks,
        p0=p0
    )

    return df_peaks, fit_func


def plot_decay(
    df,
    df_pos,
    df_neg,
    fit_pos,
    fit_neg
):
    fig, ax = plt.subplots(
        3, 1,
        sharex=True,
        height_ratios=[2, 1, 1]
    )

    plot_data(
        df,
        ax=ax[0],
        label="Datos"
    )

    plot_fitted(
        df_pos,
        fit_pos,
        ax=ax[0],
        label="Ajuste superior"
    )

    plot_fitted(
        df_neg,
        fit_neg,
        ax=ax[0],
        label="Ajuste inferior"
    )

    plot_residue(
        df_pos,
        fit_pos,
        ax=ax[1],
        label="Residuo de ajuste superior"
    )

    plot_residue(
        df_neg,
        fit_pos,
        ax=ax[2],
        label="Residuo de ajuste inferior"
    )

    for axis in ax:
        axis.set(
            ylabel="Voltaje [V]"
        )
        axis.legend()

    ax[2].set(xlabel="Tiempo [s]")
