import numpy as np
import lmfit

# --------------------------------------------------------------
# 1️⃣  Define the model – it receives a Parameter object
# --------------------------------------------------------------
def model(params, t):
    a = params['a']
    b = params['b']
    c = params['c']
    d = params['d']
    return np.column_stack((a * np.exp(-b * t),
                            c * np.exp(-d * t)))   # (N,2)

# --------------------------------------------------------------
# 2️⃣  Create a Parameters container with bounds / hints
# --------------------------------------------------------------
pars = lmfit.Parameters()
pars.add('a', value=1.0, min=0)   # a must be positive
pars.add('b', value=1.0, min=-np.inf, max=np.inf)
pars.add('c', value=1.0, min=0)
pars.add('d', value=1.0)

# --------------------------------------------------------------
# 3️⃣  Residual function that lmfit will minimise
# --------------------------------------------------------------
def resid(params, t, data):
    pred = model(params, t)               # (N,2)
    return (pred - data).ravel()          # flatten → 1‑D residual vector

# --------------------------------------------------------------
# 4️⃣  Run the fit
# --------------------------------------------------------------
t = np.linspace(0, 4, 30)
y_obs = ...                               # (N,2) your measurements

out = lmfit.minimize(resid, pars, args=(t, y_obs), method='leastsq')

# --------------------------------------------------------------
# 5️⃣  Results
# --------------------------------------------------------------
lmfit.report_fit(out.params)
