import pylabo.fit as fit
import pylabo.plot as plot
import pandas as pd
import pylabo.lib.utils as utils

df = pd.read_csv("data.csv")
utils.insert_empty_xerr(df)

model = fit.funs.sin

func_fit = fit.fit(
    model,
    df
)

plot.fulfit(df, func_fit)
plot.show()
