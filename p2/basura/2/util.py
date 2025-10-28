import numpy as np
from enum import Enum

from errores import error_chi

ERR_NOISE = 0.2 / 100


class Material(Enum):
    COBRE = [
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
    # NADA = [
    #     './data/3 - 8-10/(7) sin nada A-B.csv',
    #     './data/3 - 8-10/(8) sin nada I.csv'
    # ]


radius = {
    Material.COBRE: 12e-3 / 2,
    Material.ALUMINIO_19: 19e-3 / 2,
    Material.ALUMINIO_CHICO: 19e-3 / 2,
}

rho = {
    Material.COBRE: 1.68e-8,
    Material.ALUMINIO_19: 2.65e-8,
    Material.ALUMINIO_CHICO: 19e-3 / 2,
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

    chi = - np.abs(volt) / (freq * np.abs(curr)) * np.exp(1j * phase)

    chi_err = error_chi(freq, curr, volt, err_curr, err_volt, phase)

    return np.asarray(chi, dtype=complex), np.asarray(chi_err, dtype=complex)
