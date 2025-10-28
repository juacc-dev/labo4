import numpy as np


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
