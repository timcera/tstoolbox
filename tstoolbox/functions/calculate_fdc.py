#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import warnings

import mando
from mando.rst_text_formatter import RSTHelpFormatter

from scipy.stats import t
import pandas as pd

from .. import tsutils

warnings.filterwarnings("ignore")


@mando.command("calculate_fdc", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def calculate_fdc_cli(
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
    sort_values="ascending",
    sort_index="ascending",
    tablefmt="csv",
    add_index=False,
    include_sd=False,
    include_cl=False,
    ci=0.9,
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

        {plotting_position_table}

    sort_values : str
        [optional, default is 'ascending', input filter]

        Sort order is either 'ascending' or 'descending'.

    sort_index : str
        [optional, default is 'ascending', input filter]

        Sort order is either 'ascending' or 'descending'.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {skiprows}
    {index_type}
    {names}
    {source_units}
    {target_units}
    {clean}
    {tablefmt}
    add_index : bool
        [optional, default is False]

        Add a monotonically increasing index.
    include_cd : bool
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
    tsutils.printiso(
        calculate_fdc(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            percent_point_function=percent_point_function,
            plotting_position=plotting_position,
            source_units=source_units,
            target_units=target_units,
            sort_values=sort_values,
            sort_index=sort_index,
            add_index=add_index,
            include_sd=include_sd,
            include_cl=include_cl,
            ci=ci,
        ),
        showindex="always",
        tablefmt=tablefmt,
    )


@tsutils.validator(
    percent_point_function=[str, ["domain", ["norm", "lognorm", "weibull"]], 1],
    plotting_position=[
        [
            str,
            [
                "domain",
                [
                    "weibull",
                    "benard",
                    "bos-levenbach",
                    "filliben",
                    "yu",
                    "tukey",
                    "blom",
                    "cunnane",
                    "gringorton",
                    "hazen",
                    "larsen",
                    "gumbel",
                    "california",
                ],
            ],
            1,
        ],
        [float, ["range", [0, 1]], 1],
    ],
    sort_values=[str, ["domain", ["ascending", "descending"]], 1],
    sort_index=[str, ["domain", ["ascending", "descending"]], 1],
    include_sd=[bool, ["domain", [True, False]], 1],
    include_cl=[bool, ["domain", [True, False]], 1],
    ci=[float, ["range", [0, 1]], 1],
)
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
    sort_values="ascending",
    sort_index="ascending",
    add_index=False,
    include_sd=False,
    include_cl=False,
    ci=0.9,
):
    """Return the frequency distribution curve."""
    sort_values = bool(sort_values == "ascending")

    tsd = tsutils.common_kwds(
        tsutils.read_iso_ts(
            input_ts, skiprows=skiprows, names=names, index_type=index_type
        ),
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
            xdat = ppf(tsutils.set_plotting_position(tmptsd.count(), plotting_position))
            tmptsd.sort_values(ascending=sort_values, inplace=True)
            tmptsd.index = xdat * 100
            tmptsd = pd.DataFrame(tmptsd)
            if include_sd is True or include_cl is True:
                sd = (xdat * (1 - xdat) / len(xdat)) ** 0.5
            if include_sd is True:
                tmptsd[col + "_sd"] = sd
            if include_cl is True:
                tval = t.ppf(ci, df=len(xdat) - 1)
                ul = 2 * (1 - xdat) * tval * sd
                ll = 2 * xdat * tval * sd
                tmptsd[col + "_ul"] = (xdat + ul) * 100
                tmptsd[col + "_ll"] = (xdat - ll) * 100
                tmptsd[col + "_vul"] = tmptsd[col] + ul * tmptsd[col]
                tmptsd[col + "_vll"] = tmptsd[col] - ll * tmptsd[col]
        else:
            tmptsd = pd.DataFrame()
        newts = newts.join(tmptsd, how="outer")
    newts.index.name = "Plotting_position"
    newts = newts.groupby(newts.index).first()
    if sort_index == "descending":
        return newts.iloc[::-1]
    if add_index is True:
        newts.reset_index(inplace=True)
    return newts


calculate_fdc.__doc__ = calculate_fdc_cli.__doc__
