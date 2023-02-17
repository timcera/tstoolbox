"""Collection of functions for the manipulation of time series."""

import warnings
from typing import Literal

import pandas as pd
from pydantic import confloat, validate_arguments
from scipy.stats import t
from toolbox_utils import tsutils

warnings.filterwarnings("ignore")


@validate_arguments
@tsutils.doc(tsutils.docstrings)
def calculate_fdc(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    clean=False,
    skiprows=None,
    index_type="datetime",
    names=None,
    percent_point_function=None,
    plotting_position="weibull",
    source_units=None,
    target_units=None,
    sort_values: Literal["ascending", "descending"] = "ascending",
    sort_index: Literal["ascending", "descending"] = "ascending",
    add_index: bool = False,
    include_ri: bool = False,
    include_sd: bool = False,
    include_cl: bool = False,
    ci: confloat(gt=0, lt=1) = 0.9,
):
    """Return the frequency distribution curve.

    DOES NOT return a time-series.

    Parameters
    ----------
    percent_point_function : str
        [optional, default is None, transformation]

        The distribution used to shift the plotting position values.
        Choose from 'norm', 'lognorm', 'weibull', and None.

    plotting_position : str
        [optional, default is 'weibull', transformation]

        ${plotting_position_table}

    sort_values : str
        [optional, default is 'ascending', input filter]

        Sort order is either 'ascending' or 'descending'.

    sort_index : str
        [optional, default is 'ascending', input filter]

        Sort order is either 'ascending' or 'descending'.

    ${input_ts}

    ${columns}

    ${start_date}

    ${end_date}

    ${skiprows}

    ${index_type}

    ${names}

    ${source_units}

    ${target_units}

    ${clean}

    ${tablefmt}

    add_index : bool
        [optional, default is False]

        Add a monotonically increasing index.

    include_ri : bool
        [optional, default is False]

        Include the recurrence interval (sometimes called the return interval).
        This is the inverse of the calculated plotting position defined by the
        equations available with the `plotting_position` keyword.

    include_sd : bool
        [optional, default is False]

        Include a standard deviation column for each column in the
        input.  The equation used is::

            Sd = (Pc(1 - Pc)/N)**0.5

        where::

            Pc is the cumulative probability
            N is the number of values

    include_cl : bool
        [optional, default is False]

        Include two columns showing the upper and lower confidence limit
        for each column in the input.  The equations used are::

            U = Pc + 2(1 - Pc) t Sd
            L = Pc - 2Pc t Sd

        where::

            Pc is the cumulative probability
            t is the Student's "t" value for number of samples and
                confidence interval as defined with `ci` keyword
            Sd is the standard deviation with the equation above

    ci : float
        [optional, default is 0.9]

        This is the confidence interval used when the `include_cl`
        keyword is active.  The confidence interval of 0.9 implies an
        upper limit of 0.95 and a lower limit of 0.05 since 0.9 = 0.95
        - 0.05.
    """
    sort_values = sort_values == "ascending"

    tsd = tsutils.common_kwds(
        input_ts,
        skiprows=skiprows,
        names=names,
        index_type=index_type,
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        source_units=source_units,
        target_units=target_units,
        clean=clean,
    )

    ppf = tsutils.set_ppf(percent_point_function)
    newts = pd.DataFrame()
    for col in tsd:
        tmptsd = tsd[col].dropna()
        if len(tmptsd) > 1:
            return_interval = tsutils.set_plotting_position(
                tmptsd.count(), plotting_position
            )
            xdat = ppf(return_interval)
            tmptsd.sort_values(ascending=sort_values, inplace=True)
            tmptsd.index = xdat * 100
            tmptsd = pd.DataFrame(tmptsd)
            if include_ri:
                tmptsd[f"{col}_ri"] = 1.0 / xdat
            if include_sd or include_cl:
                sd = (xdat * (1 - xdat) / len(xdat)) ** 0.5
            if include_sd:
                tmptsd[f"{col}_sd"] = sd
            if include_cl:
                tval = t.ppf(ci, df=len(xdat) - 1)
                ul = 2 * (1 - xdat) * tval * sd
                ll = 2 * xdat * tval * sd
                tmptsd[f"{col}_ul"] = (xdat + ul) * 100
                tmptsd[f"{col}_ll"] = (xdat - ll) * 100
                tmptsd[f"{col}_vul"] = tmptsd[col] + ul * tmptsd[col]
                tmptsd[f"{col}_vll"] = tmptsd[col] - ll * tmptsd[col]
        else:
            tmptsd = pd.DataFrame()
        newts = newts.join(tmptsd, how="outer")
    newts.index.name = "Plotting_position"
    newts = newts.groupby(newts.index).first()
    if sort_index == "descending":
        return newts.iloc[::-1]
    if add_index:
        newts.reset_index(inplace=True)
    return newts
