import pandas as pd
import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration, \
    READ_ALL_AVAILABLE

import scipy.fft
import scipy.signal
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.family"] = "serif"
plt.rcParams['axes.grid'] = True
plt.rcParams["figure.figsize"] = (9, 7)
plt.rcParams['legend.loc'] = "best"
plt.rcParams['legend.fontsize'] = 11

#%%

PATH = Path(
    r"C:\Users\publico\Documents\Laboratorio 4, 2025.2 - Grupo 6\datos 10-9"
)

FREQ_MAX = 500       # Donde se corta el espectro (en Hz)
SKIP_SPECTRUM = 0    # Para ignorar los primeros puntos del espectro

N_PEAKS = 5          # Número de picos a encontrar
PEAK_SEP = 100       # Separación mínima entre picos (en Hz)

# Para detectar dónde empieza la vibración
START_ZERO = 50      # Primeros puntos para tomar como cero
ZERO_TOLERANCE = 10  # Qué tan distinto de cero tiene que ser el primer punto
STOP_AFTER = 0       # Hasta qué segundo de la medición usar
