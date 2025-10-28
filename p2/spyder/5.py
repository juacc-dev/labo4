chi, chi_err = calculate_chi(df_v, df_i)

fig, ax = plt.subplots(1, 1)

ax.errorbar(
    freq,
    chi.real,
    yerr=chi_err.real,
    label="Parte real"
)
ax.errorbar(
    freq,
    chi.imag,
    yerr=chi_err.imag,
    label="Parte imaginaria"
)
ax.set(
    xlabel="Frecuencia [Hz]",
    ylabel="Susceptibilidad",
)
ax.legend()

plt.show()

# %%


def chi2_r(
    residue,
    yerr,
    n_data,
    n_params
):
    chi2 = np.sum((residue / yerr) ** 2)
    degrees_of_freedom = n_data - n_params

    return chi2 / degrees_of_freedom


def p_value(
    residue,
    yerr,
    n_data,
    n_params
):
    chi2 = np.sum((residue / yerr) ** 2)
    degrees_of_freedom = n_data - n_params

    return scipy.stats.chi2.sf(chi2, df=degrees_of_freedom)


def fit(model, freq, chi, chi_err, p0):
    try:
        p_opt_real, p_cov_real = curve_fit(
            lambda f, k, alpha, scale, offset: np.real(
                model_func(f, k, alpha, scale, offset)
            ),
            freq,
            chi.real,
            p0=p0,
            sigma=chi_err,
            absolute_sigma=True
        )

    except Exception:
        print("No se pudo ajustar la parte real.")
        p_opt_real, p_cov_real = None, None

    try:
        p_opt_imag, p_cov_imag = curve_fit(
            lambda f, k, alpha, scale, offset: np.imag(
                model_func(f, k, alpha, scale, offset)
            ),
            freq,
            chi.imag,
            p0=p0,
            sigma=chi_err,
            absolute_sigma=True
        )

    except Exception:
        print("No se pudo ajustar la parte imaginaria.")
        p_opt_imag, p_cov_imag = None, None

    if p_opt_real is None and p_opt_imag is None:
        raise Exception

    p_err_real = np.sqrt(np.diag(p_cov_real))
    p_err_imag = np.sqrt(np.diag(p_cov_imag))

    return p_opt_real, p_err_real, p_opt_imag, p_err_imag


def model(f, k, alpha, scale, offset):
    f = np.array(f)
    x = (-1 + 1j) * np.sqrt(np.pi * MU0 * np.abs(k) * f)

    return - scale * jv(2, x) / jv(0, x) * np.exp(1j * np.pi / 180 * alpha) + offset

# %%


fig, ax = plt.subplots(3, 1, sharex=True, height_ratios=[3, 1, 1])

K0 = RADIO ** 2 / RHO
CHI_MAX = (chi.real ** 2 + chi.imag ** 2).max()
p0 = [K0, 0, CHI_MAX, 0]

p_opt_real, p_err_real, p_opt_imag, p_err_imag = fit(
    model,
    freq,
    chi,
    chi_err,
    p0
)

chi_fit = np.real(model(freq, *p_opt_real)) + 1j * np.imag(model(freq), *p_err_imag)

k0 = [p_opt_real[0], p_opt_imag[0]]
k0_err = [p_err_real[0], p_err_imag[0]]

residuos = [
    chi.real - chi_fit.real,
    chi.imag - chi_fit.imag,
]

chi2 = [
    chi2_r(residuos[0], chi_err.real, len(residuos[0]), len(p0)),
    chi2_r(residuos[1], chi_err.imag, len(residuos[1]), len(p0)),
]

print(f"Se esperaba obtener k0 = {K0:.0e}.")
print(f"Se obtuvo (real) {k0[0]:.0e} +- {k0_err[0]:.0e}")
print(f"Se obtuvo (imag) {k0[1]:.0e} +- {k0_err[1]:.0e}")
print("")
print(f"chi^2_red = {0:.2e} (real), {0:.2e} (imag)")
print(f"P valor = {0:.2e} (real), {0:.2e} (imag)")


fig, ax = plt.subplots(3, 1, sharex=True, height_ratios=[3, 1, 1])

ax[0].set(
    ylabel="Susceptibilidad",
)
ax[1].set(
    ylabel="Residuos",
)
ax[2].set(
    xlabel="Frecuencia [Hz]",
    ylabel="Residuos",
)

ax[0].errorbar(
    freq,
    chi.real,
    yerr=chi_err.real,
    label="Parte real"
)
ax[0].errorbar(
    freq,
    chi.imag,
    yerr=chi_err.imag,
    label="Parte imaginaria"
)

ax[0].plot(
    freq,
    chi_fit.real,
    label="Ajuste real"
)
ax[0].plot(
    freq,
    chi_fit.imag,
    label="Ajuste imaginario"
)

ax[1].axhline(
    y=0,
    color="black",
    alpha=0.9
)
ax[1].errorbar(
    freq,
    residuos[0],
    label="Residuos parte real"
)

ax[2].axhline(
    y=0,
    color="black",
    alpha=0.9
)
ax[2].errorbar(
    freq,
    residuos[1],
    label="Residuos parte imaginaria"
)

for axis in ax:
    axis.legend()

plt.show()
