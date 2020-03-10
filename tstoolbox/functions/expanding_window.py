#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import mando
from mando.rst_text_formatter import RSTHelpFormatter
import pandas as pd

from .. import tsutils


@mando.command("expanding_window", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def expanding_window_cli(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    clean=False,
    statistic="",
    min_periods=1,
    center=False,
    source_units=None,
    target_units=None,
    print_input=False,
    tablefmt="csv",
):
    """Calculate an expanding window statistic.

    Parameters
    ----------
    statistic : str
        [optional, default is '']

        +-----------+----------------------+
        | statistic | Meaning              |
        +===========+======================+
        | corr      | correlation          |
        +-----------+----------------------+
        | count     | count of real values |
        +-----------+----------------------+
        | cov       | covariance           |
        +-----------+----------------------+
        | kurt      | kurtosis             |
        +-----------+----------------------+
        | max       | maximum              |
        +-----------+----------------------+
        | mean      | mean                 |
        +-----------+----------------------+
        | median    | median               |
        +-----------+----------------------+
        | min       | minimum              |
        +-----------+----------------------+
        | skew      | skew                 |
        +-----------+----------------------+
        | std       | standard deviation   |
        +-----------+----------------------+
        | sum       | sum                  |
        +-----------+----------------------+
        | var       | variance             |
        +-----------+----------------------+

    min_periods : int
        [optional, default is 1]

        Minimum number of observations in window required to have a value

    center : boolean
        [optional, default is False]

        Set the labels at the center of the window.

    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {skiprows}
    {index_type}
    {names}
    {clean}
    {source_units}
    {target_units}
    {print_input}
    {tablefmt}

    """
    tsutils.printiso(
        expanding_window(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            clean=clean,
            statistic=statistic,
            min_periods=min_periods,
            center=center,
            source_units=source_units,
            target_units=target_units,
            print_input=print_input,
        ),
        tablefmt=tablefmt,
    )


@tsutils.validator(
    statistic=[
        str,
        [
            "domain",
            [
                "corr",
                "count",
                "cov",
                "kurt",
                "max",
                "mean",
                "median",
                "min",
                "skew",
                "std",
                "sum",
                "var",
            ],
        ],
        None,
    ],
    min_periods=[int, ["range", [0,]], 1],
    center=[bool, ["domain", [True, False]], 1],
)
def expanding_window(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    clean=False,
    statistic="",
    min_periods=1,
    center=False,
    source_units=None,
    target_units=None,
    print_input=False,
):
    """Calculate an expanding window statistic."""
    tsd = tsutils.common_kwds(
        tsutils.read_iso_ts(
            input_ts, skiprows=skiprows, names=names, index_type=index_type
        ),
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        dropna=dropna,
        source_units=source_units,
        target_units=target_units,
        clean=clean,
    )

    ntsd = tsd.expanding(min_periods=min_periods, center=center)

    if statistic:
        nntsd = pd.DataFrame()
        for stat in tsutils.make_list(statistic):
            ntsd = eval("ntsd.{0}()".format(stat))
            ntsd.columns = [
                tsutils.renamer(i, "expanding.{0}".format(stat)) for i in ntsd.columns
            ]
            nntsd = nntsd.join(ntsd, how="outer")
    else:
        nntsd = ntsd

    return tsutils.return_input(print_input, tsd, nntsd)


expanding_window.__doc__ = expanding_window_cli.__doc__
