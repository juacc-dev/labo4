import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvisa
import bisect
import time
from scipy.optimize import curve_fit
from scipy.special import jv
from pathlib import Path

plt.rcParams["font.family"] = "serif"
plt.rcParams['axes.grid'] = True
plt.rcParams["figure.figsize"] = (9, 7)
plt.rcParams['legend.loc'] = "best"
plt.rcParams['legend.fontsize'] = 11

PATH = Path(
    r"C:\Users\publico\Documents\L4 2025-2 grupo 6\datos\3 - 8-10"
).mkdir(parents=True, exist_ok=True)


MU0 = 4 * np.pi * 1e-7

RHO = {
    "cobre": 1.68e-8,
    "aluminio": 2.65e-8
}

RADIO = {
    "aluminio chico": 0,
    "cobre chico": 12e-3,
    "cobre grande": 0,
}
