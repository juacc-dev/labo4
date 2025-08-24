import pylabo.plot as plot
import pylabo.fit as fit
import numpy as np
import pandas as pd


df = pd.read_csv("data.csv")

model = fit.funs.sin

fig, ax = plot.subplots(
    2,
    1,
    sharex=True
)

fit_func = fit.fit(
    model,
    df,
    no_xerr=True
)

plot.fit(
    fit_func,
    fig=fig, ax=ax[0]
)
plot.residue(
    df,
    fit_func,
    no_xerr=True,
    fig=fig, ax=ax[1]
)
plot.show()
