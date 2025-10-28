import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvisa
import bisect
import time
# from scipy.optimize import curve_fit
# import scipy.stats
from pathlib import Path

plt.rcParams["font.family"] = "serif"
plt.rcParams['axes.grid'] = True
plt.rcParams["figure.figsize"] = (9, 7)
plt.rcParams['legend.loc'] = "best"
plt.rcParams['legend.fontsize'] = 11

PATH = Path(
    r"C:\Users\publico\Documents\L4 2025-2 grupo 6\datos\3 - 8-10"
)

# %%

count = 0
