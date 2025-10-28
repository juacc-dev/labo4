import scipy.stats
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger()


def set_if_none(value, default):
    return default if value is None else value


def chi2_r(
    residue,
    yerr,
    n_data,
    n_params
):
    """Reduced chi squared"""

    chi2 = np.sum((residue / yerr) ** 2)
    degrees_of_freedom = n_data - n_params

    return chi2 / degrees_of_freedom


def r2(
    y_data,
    residue
):
    """R squared"""
    return 1 - np.var(residue) / np.var(y_data)


def p_value(
    residue,
    yerr,
    n_data,
    n_params
):
    chi2 = np.sum((residue / yerr) ** 2)
    degrees_of_freedom = n_data - n_params

    return scipy.stats.chi2.sf(chi2, df=degrees_of_freedom)


def split_params(n, x, *args):
    """Function for internal use. It's common functionality for operator
    overloading."""

    return args[:n], args[n:]


class Function:
    """
    Mathematical function with information about the parameters.
    """

    def __init__(
        self,
        f,               # Callable
        param_str: list[str],  # Parameter names
        eq: str = None      # LaTeX formula
    ):
        self.f = f
        self.param_str = param_str
        self.eq = eq

    def __add__(self, other):
        def f(x, *args):
            n = len(self.param_str)
            args1, args2 = split_params(n, x, *args)

            return self.f(x, *args1) + other.f(x, *args2)

        return Function(
            f,
            self.param_str + other.param_str
        )

    def __sub__(self, other):
        def f(x, *args):
            n = len(self.param_str)
            args1, args2 = split_params(n, x, *args)

            return self.f(x, *args1) - other.f(x, *args2)

        return Function(
            f,
            self.param_str + other.param_str
        )

    def __mul__(self, other):
        def f(x, *args):
            n = len(self.param_str)
            args1, args2 = split_params(n, x, *args)

            return self.f(x, *args1) * other.f(x, *args2)

        return Function(
            f,
            self.param_str + other.param_str
        )

    def __truediv__(self, other):
        def f(x, *args):
            n = len(self.param_str)
            args1, args2 = split_params(n, x, *args)

            return self.f(x, *args1) / other.f(x, *args2)

        return Function(
            f,
            self.param_str + other.param_str
        )

    # This is function composition: f & g -> f(g(x))
    def __and__(self, other):
        def f(x, *args):
            n = len(self.param_str)
            args1, args2 = split_params(n, x, *args)

            return self.f(other.f(x, *args2), *args2)

        return Function(
            f,
            self.param_str + other.param_str
        )


class FittedFunction(Function):
    """
    Function class together with numeric parameters and information about the
    fit.
    """

    def __init__(
        self,
        func: Function,
        param_val: list[float],  # Optimal parameters
        param_cov: list[float],  # Covariance matrix
        param_err: list[float],  # Parameter uncertainty
        residue,                 # y_data - y_fit
        tests=None               # table with chi squared and stuff
    ):
        super().__init__(
            func.f,
            func.param_str,
            func.eq
        )

        self.param_val = param_val
        self.param_cov = param_cov
        self.param_err = param_err
        self.residue = residue
        self.tests = tests

    def report(self) -> pd.DataFrame:
        """Create a dataframe with the results of the fit: tests (like reduced
        chi squared) and optimal parameters, the latter with their uncertainty.
        """

        # 1st column: parameter names
        names = list(self.tests.keys()) + self.param_str

        # 2nd column: values / optimal values
        values = list(self.tests.values()) + list(self.param_val)

        # 3rd column: uncertainty. Tests don't have any
        errors = [None for _ in range(
            len(self.tests))] + list(self.param_err)

        df = pd.DataFrame({
            "Parámetro": pd.Series(names),
            "Valor": pd.Series(values),
            "Error": pd.Series(errors)
        })

        return df


def fit_real(
    func: Function,
    data_x,
    data_y,
    p0=None,
    yerr=None,
    absolute_sigma=True
):
    """
    Fit a real function (wraped in the Function class) to data using curve_fit.
    """

    if p0 is None:
        logger.warning(
            "Passing no initial parameters to nonlinear function."
        )

    try:
        param_opt, param_cov = curve_fit(
            func.f,
            data_x,
            data_y,
            p0=p0,
            sigma=yerr,
            absolute_sigma=absolute_sigma
        )

    except RuntimeError as e:
        logger.error(f"Failed to fit function. Error: {e}")
        return None, None

    return param_opt, param_cov


def fit(
    model: Function,
    df: pd.DataFrame,
    p0=None,
    absolute_sigma=True
) -> FittedFunction:
    """
    Fit a function to data.
    Returns an object containing all information about the result.
    """

    x_data, _, y_data, yerr = split_dataframe(df)

    yerr = yerr if not yerr.isna().all() else None

    p_opt, p_cov = fit_real(
        model,
        x_data,
        y_data,
        p0=p0,
        yerr=yerr,
        absolute_sigma=absolute_sigma
    )

    if p_opt is None and p_cov is None:
        return None

    # Error in parameters
    p_err = np.sqrt(np.diag(p_cov))

    y_fit = model.f(x_data, *p_opt)

    residue = y_fit - y_data

    # Tests
    r_sq = r2(y_data, residue)
    chi = chi2_r(
        residue,
        yerr,
        len(residue),
        len(p_opt)
    ) if yerr is not None else None

    p_val = p_value(residue, yerr, len(residue), len(p_opt))

    # New Function object
    fit_func = FittedFunction(
        model,
        p_opt,
        p_cov,
        p_err,
        residue,
        tests={
            "R2": r_sq,
            "chi2 red": chi,
            "P value": p_val
        }
    )

    return fit_func


def fmt_choice(n_points: int):
    if n_points < 200:
        return "."

    else:
        return "-"


def plot_fitted(
    df: pd.DataFrame,
    fit_func: FittedFunction,
    ax=None,
    label=None,
    fmt=None,
    **kwargs
):
    """
    Plot the fitted function. More points are used as to draw a smooth curve.
    """
    x_axis, _, _, _ = split_dataframe(df)

    n_points = int(8 * plt.rcParams["figure.dpi"])

    x_fit = np.linspace(x_axis.min(), x_axis.max(), n_points)
    y_fit = fit_func.f(x_fit, *fit_func.param_val)

    fig = None

    if ax is None:
        fig, ax = plt.subplots(
            1,
            1,
        )

    ax.plot(
        x_fit,
        y_fit,
        label=label
    )

    return fig, ax


def plot_residue(
    df: pd.DataFrame,
    fit_func: FittedFunction,
    ax=None,
    fmt=None,
    ylabel=None,
    **kwargs
):
    """Plot the residue from a fit."""

    x_axis, x_err, _, y_err = split_dataframe(df)

    fig = None

    if ax is None:
        fig, ax = plt.subplots(
            1,
            1,
        )

    # fmt may be 'o' or '.' depending on the number of points
    fmt = set_if_none(fmt, fmt_choice(x_axis.size))
    ylabel = set_if_none(ylabel, "Residuos")

    # If there is no uncertainty in X, don't plot it
    x_err = x_err if not x_err.isna().all() else None

    ax.axhline(
        y=0,
        color="black",
        alpha=0.9
    )

    ax.errorbar(
        x_axis,
        fit_func.residue,
        xerr=x_err,
        yerr=y_err,
        **kwargs
    )

    return fig, ax
