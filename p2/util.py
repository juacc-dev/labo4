import numpy as np
from enum import Enum
import pandas as pd
import pylabo.fit as fit

from errores import error_chi

MU0 = 4 * np.pi * 1e-7
ERR_NOISE = 0.2 / 100


class Material(Enum):
    NADA_1 = [
        './data/1 - 24-9/sin A-B.csv',
        './data/1 - 24-9/sin I.csv',
    ]

    COBRE_1 = [
        './data/1 - 24-9/con A-B cobre.csv',
        './data/1 - 24-9/con I cobre.csv',
    ]

    COBRE_GRANDE = [
        './data/2 - 1-10/con cobre grande A-B.csv',
        './data/2 - 1-10/con cobre grande I.csv',
    ]

    ALUMINIO_CHICO = [
        './data/2 - 1-10/con aluminio chico A-B.csv',
        './data/2 - 1-10/con aluminio chico I.csv',
    ]

    ALUMINIO_19 = [
        './data/3 - 8-10/(0) con aluminio 19mm A-B.csv',
        './data/3 - 8-10/con aluminio 19mm I.csv'
    ]
    ALUMINIO_12 = [
        './data/3 - 8-10/(1) con aluminio 12.65mm A-B.csv',
        './data/3 - 8-10/(2) con aluminio 12.65mm I.csv'
    ]
    LATON = [
        './data/3 - 8-10/con laton chico A-B.csv',
        './data/3 - 8-10/(3) con laton chico I.csv'
    ]
    ALUMINIO_13_1 = [
        './data/3 - 8-10/(5) con aluminio 13.65 A-B.csv',
        './data/3 - 8-10/(4) con aluminio 13.65 I.csv'
    ]
    ALUMINIO_13_2 = [
        './data/3 - 8-10/(6) con aluminio 13.65 A-B.csv',
        './data/3 - 8-10/(4) con aluminio 13.65 I.csv'
    ]
    NADA = [
        './data/3 - 8-10/(7) sin nada A-B.csv',
        './data/3 - 8-10/(8) sin nada I.csv'
    ]


radius = {
    Material.COBRE_1: 12e-3 / 2,
    Material.COBRE_GRANDE: 12e-3 / 2,
    Material.ALUMINIO_19: 19e-3 / 2,
    Material.ALUMINIO_CHICO: 19e-3 / 2,
    Material.LATON: 13.10e-3 / 2,
}

rho = {
    Material.COBRE_1: 1.68e-8,
    Material.COBRE_GRANDE: 1.68e-8,
    Material.ALUMINIO_19: 2.65e-8,
    Material.ALUMINIO_CHICO: 2.65e-8,
    Material.LATON: 1.68e-8,
}


def value_and_error(df):
    value = np.array(df["X [V]"] + 1j * df["Y [V]"])
    sens = 2 * df["Escala [V]"] / 2 ** 16

    err = np.sqrt(
        df["Ruido X [V]"] ** 2 + sens ** 2 + (value.real * ERR_NOISE) ** 2
    )\
        + 1j * np.sqrt(
        df["Ruido Y [V]"] ** 2 + sens ** 2 + (value.imag * ERR_NOISE) ** 2
    )

    return value, err


def calculate_chi(df_v, df_i):
    freq = df_v["Frecuencia [Hz]"]

    # Voltaje y corriente como números complejos
    volt = np.array(df_v["X [V]"] + 1j * df_v["Y [V]"])
    curr = np.array(df_i["X [V]"] + 1j * df_i["Y [V]"])

    volt, err_volt = value_and_error(df_v)
    curr, err_curr = value_and_error(df_i)

    phase = np.arctan2(
        np.imag(volt), np.real(volt)
    ) - np.arctan2(
        np.imag(curr), np.real(curr)
    )

    chi = volt / (freq * curr) * np.exp(1j * phase)

    chi_err = error_chi(freq, curr, volt, err_curr, err_volt, phase)

    return np.asarray(chi, dtype=complex), np.asarray(chi_err, dtype=complex)


def complex_fit(
    df: pd.DataFrame,
    model_func,
    model_params: list,
    p0: list = None
):
    df_real = pd.DataFrame({
        "Frecuencia [Hz]": df["Frecuencia [Hz]"],
        "Error Frecuencia [Hz]": pd.Series(),

        "X [V]": df["X [V]"],
        "Error X [V]": df["Error X [V]"],
    })

    df_imag = pd.DataFrame({
        "Frecuencia [Hz]": df["Frecuencia [Hz]"],
        "Error Frecuencia [Hz]": pd.Series(),

        "Y [V]": df["Y [V]"],
        "Error Y [V]": df["Error Y [V]"]
    })

    model_real = fit.Function(
        lambda f, k, alpha, scale, offset: np.real(
            model_func(f, k, alpha, scale, offset)),
        model_params
    )

    model_imag = fit.Function(
        lambda f, k, alpha, scale, offset: np.imag(
            model_func(f, k, alpha, scale, offset)),
        model_params
    )

    fit_func_real = fit.fit(
        model_real,
        df_real,
        p0=p0
    )

    fit_func_imag = fit.fit(
        model_imag,
        df_imag,
        p0=p0
    )
    # if fit_func_real is None or fit_func_imag is None:
    #     failed_part = "real" if fit_func_real is None else "imag"
    #     print(f"Failed to fit {name} ({failed_part}).")
    #     fit_func_real = None
    #     fit_func_imag = None

    return df_real, df_imag, fit_func_real, fit_func_imag


def try_fit(
    df: pd.DataFrame,
    model_func,
    model_params,
    p0=None,
    n_max=20
):
    for n in range(n_max):
        try:
            df_real, df_imag, fit_func_real, fit_func_imag = complex_fit(
                df[n:],
                model_func,
                model_params,
                p0=p0
            )

            if fit_func_real is not None and fit_func_imag is not None:
                return df_real[n:], df_imag[n:], fit_func_real, fit_func_imag, n

        except Exception as e:
            print(f"An error occurred {e}")
            break

    return None, None, None, None, np.nan
