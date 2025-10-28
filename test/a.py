import numpy as np
import pylabo.fit as fit
import pylabo.plot as plot
import pandas as pd
import pylabo.lib.utils as utils

df = pd.read_csv("data.csv")
utils.insert_empty_xerr(df)

model = fit.funs.sin
model = fit.Function(
    lambda x, a, w, d: a * np.sin(w * x + d),
    ["a", "w", "d"]
)

func_fit = fit.fit(
    model,
    df,
    p0=[1, 1, 0]
)

print(func_fit.report())

plot.datafit(df, func_fit)
plot.show()
