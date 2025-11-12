import pandas as pd
import numpy as np
from pathlib import Path
import pylabo.plot as plot
from pylabo.lib.utils import unpack_df
import pylabo.fit as fit
import matplotlib.pyplot as plt

from util import transferencia, even_taylor_cut


df_in = pd.read_csv("./data/3 - 29-10/(2) in (muchos puntos).csv")
df_out = pd.read_csv("./data/3 - 29-10/(17) out (muchos puntos).csv")

df_t = transferencia(df_in, df_out)

f, _, t, terr = unpack_df(df_t)

idx = t.idxmin()
f_min = f[idx]
t_min = t[idx]

N = 2
POINTS = 3

win = POINTS // 2
df_t = df_t[idx - win:idx + win + 1]

param_str = ["x0"]
for i in range(N//2):
    param_str.append(f"a{i*2}")

p0 = [f_min, t_min]
p0.extend([0 for _ in range(N//2)])

model = fit.Function(
    even_taylor_cut,
    param_str
)

fit_func = fit.fit(
    model,
    df_t,
    p0=p0
)
print(fit_func.report())

plot.datafit(
    df_t,
    fit_func
)

# plt.errorbar(f, t, yerr=terr, fmt='.')
plt.show()
