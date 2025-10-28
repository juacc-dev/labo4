def promedio(lockin, n=30, tau=0):
    z = np.zeros(n, dtype=complex)

    for i in range(n):
        z[i] = lockin.right_value()
        time.sleep(tau / n)

    z_avg = np.average(z)
    stdev = np.std(np.real(z)) + 1j * np.std(np.imag(z))

    return z_avg, stdev


def measure(lockin: SR830, freqs, N_TAU=5, Q=20) -> pd.DataFrame:
    data = np.zeros_like(freqs, dtype=complex)
    noise = np.zeros_like(freqs, dtype=complex)

    # Error en frecuencia y en amplitud
    times = np.zeros_like(freqs)
    scales = np.zeros_like(freqs)

    for i in range(freqs.size):

        freq = lockin.set_internal_ref_cfg(freq=freqs[i])[0]

        print(f"Frecuencia: {freq:.0f} Hz")
        # Mantiene el factor de calidad ~constante
        integration_time = Q / freq
        integration_time = lockin.set_filter_cfg(time=integration_time)[0]

        time.sleep((N_TAU - 1) * integration_time)

        data[i], noise[i] = promedio(lockin, n=30, tau=integration_time)

        # Creo??
        times[i] = integration_time
        scales[i] = lockin.get_scale()

    # return data
    df = pd.DataFrame({
        "Frecuencia [Hz]": freqs,
        "X [V]": np.real(data),
        "Ruido X [V]": np.real(noise),
        "Y [V]": np.imag(data),
        "Ruido Y [V]": np.imag(noise),
        "Tiempo de integración [s]": times,
        "Escala [V]": scales
    })

    return df


def plot_measure(df):
    fig, ax = plt.subplots(
        1, 1
    )

    freq = df["Frecuencia [Hz]"]
    x = df["X [V]"]
    xerr = np.abs(df["Ruido X [V]"])
    y = df["Y [V]"]
    yerr = np.abs(df["Ruido Y [V]"])

    ax.errorbar(freq, x, yerr=xerr, fmt=".", capsize=2, label="Parte real")
    ax.errorbar(freq, y, yerr=yerr, fmt=".",
                capsize=2, label="Parte imaginaria")

    ax.legend()
    ax.set(
        xlabel="Frecuencia [Hz]",
        ylabel="Amplitud"
    )

    plt.show()


def errores(curr, volt, err_curr, err_volt, alpha):
    Xx, Yx = np.real(curr), np.imag(curr)
    Xy, Yy = np.real(volt), np.imag(volt)
    eXx, eYx = np.real(err_curr), np.imag(err_curr)
    eXy, eYy = np.real(err_volt), np.imag(err_volt)

    ux = np.sqrt(Xx**2 + Yx**2)
    err_ux = np.sqrt((Xx * eXx)**2 + (Yx * eYx)**2)/ux

    uy = np.sqrt(Xy**2 + Yy**2)
    err_uy = np.sqrt((Xy * eXy)**2 + (Yy * eYy)**2)/uy

    err_alfa = np.sqrt((np.sqrt((eYy / Xy)**2+(eXy * Yy / Xy ** 2) ** 2)/(1+(Yy/Xy)**2))
                       ** 2 + (np.sqrt((eYx / Xx)**2 + (eXx * Yx / Xx ** 2) ** 2)/(1 + (Yx / Xx)**2))**2)

    cov_ux_alfa = ((Xx*Yx)*(Xx**2+Yx**2)**(-2/3)) * \
        (eYx**2-eXx**2)  # ( Xx*Yx / ux**3) * (eYx**2- eXx**2)
    cov_uy_alfa = ((Xy*Yy)*(Xy**2+Yy**2)**(-2/3)) * \
        (-eYy**2+eXy**2)  # ( Xy*Yy / uy**3) * (eXy**2- eYy**2)

    return err_alfa, err_ux, err_uy, cov_ux_alfa, cov_uy_alfa


def chi_err_real(uy, err_uy, ux, err_ux, alfa, err_alfa, cov_uy_alfa, cov_ux_alfa, f):
    valor = -(uy/(f*ux))*np.sin(alfa)
    err = valor*np.sqrt((err_uy / uy)**2 + (err_ux / ux)**2 + (err_alfa / np.tan(alfa))
                        ** 2 + (2 / np.tan(alfa))*(cov_uy_alfa / uy - cov_ux_alfa / ux))
    return err


def chi_err_imag(uy, err_uy, ux, err_ux, alfa, err_alfa, cov_uy_alfa, cov_ux_alfa, f):
    valor = -(uy/(f*ux))*np.cos(alfa)
    err = valor*np.sqrt((err_uy / uy)**2 + (err_ux / ux)**2 + (err_alfa * np.tan(alfa))
                        ** 2 + (2 * (-np.tan(alfa))) * (cov_uy_alfa / uy - cov_ux_alfa / ux))
    return err


def error_chi(freq, curr, volt, err_curr, err_volt, phase):
    err_phase, err_curr_abs, err_volt_abs, cov_curr, cov_volt = errores(
        curr, volt, err_curr, err_volt, phase
    )

    err_real = chi_err_real(
        np.abs(volt), err_volt_abs, np.abs(
            curr), err_curr_abs, phase, err_phase,
        cov_volt, cov_curr, freq
    )

    err_imag = chi_err_imag(
        np.abs(volt), err_volt_abs, np.abs(
            curr), err_curr_abs, phase, err_phase,
        cov_volt, cov_curr, freq
    )

    return err_real + 1j * err_imag


def calculate_chi(df_v, df_i):
    freq = df_v["Frecuencia [Hz]"]

    # Voltaje y corriente como números complejos
    volt = np.array(df_v["X [V]"] + 1j * df_v["Y [V]"])
    curr = np.array(df_i["X [V]"] + 1j * df_i["Y [V]"])

    sens_v = 2 * df_v["Escala [V]"] / 2 ** 16
    sens_i = 2 * df_i["Escala [V]"] / 2 ** 16

    err_volt = np.sqrt(df_v["Ruido X [V]"] ** 2 + sens_v ** 2) + \
        1j * np.sqrt(df_v["Ruido Y [V]"] ** 2 + sens_v ** 2)

    err_curr = np.sqrt(df_i["Ruido X [V]"] ** 2 + sens_i ** 2) + \
        1j * np.sqrt(df_i["Ruido Y [V]"] ** 2 + sens_i ** 2)

    phase = np.arctan2(
        np.imag(volt), np.real(volt)
    ) - np.arctan2(
        np.imag(curr), np.real(curr)
    )

    chi = - np.abs(volt) / (freq * np.abs(curr)) * np.exp(1j * phase)

    chi_err = error_chi(freq, curr, volt, err_curr, err_volt, phase)

    return np.asarray(chi, dtype=complex), np.asarray(chi_err, dtype=complex)
