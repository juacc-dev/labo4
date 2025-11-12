import pandas as pd
import numpy as np
from pathlib import Path
import pylabo.plot as plot
import pylabo.fit as fit
import matplotlib.pyplot as plt
from pylabo.lib.utils import unpack_df

from util import transferencia, even_taylor_cut


path = Path("./data/2 - 22-10")

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

df_in = [pd.read_csv(path / file) for file in files_in]
df_out = [pd.read_csv(path / file) for file in files_out]

df_in = pd.concat(df_in).sort_values(
    "Frecuencia [Hz]",
    ignore_index=True
)

df_out = pd.concat(df_out).sort_values(
    "Frecuencia [Hz]",
    ignore_index=True
)

# df_in = make_df(df_raw_in)
# df_out = make_df(df_raw_out)

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
print(param_str)
print(p0)

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
