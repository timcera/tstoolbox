#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

from typing import List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import warnings

import mando
from mando.rst_text_formatter import RSTHelpFormatter

import pandas as pd
import typic

from .. import tsutils

warnings.filterwarnings("ignore")


@mando.command("aggregate", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def aggregate_cli(
    input_ts="-",
    groupby=None,
    statistic="mean",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    agg_interval=None,
    ninterval=None,
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    source_units=None,
    target_units=None,
    print_input=False,
    tablefmt="csv",
    skipna=True,
):
    """Take a time series and aggregate to specified frequency.

    Parameters
    ----------
    statistic : str
        [optional, defaults to 'mean', transformation]

        Any of 'mean', 'sem', 'sum', 'std', 'max', 'min', 'median',
        'first', 'last', 'ohlc', or list of same to calculate on each
        `groupby` group.

        Python example::
            statistic=['mean', 'max', 'first']

        Command line example::
            --statistic=mean,max,first
    {groupby}

        The `groupby` keyword has a special option 'all' which will aggregate
        all records.

        {pandas_offset_codes}

    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {clean}
    {round_index}
    {skiprows}
    {index_type}
    {names}
    {source_units}
    {target_units}
    {print_input}
    {tablefmt}
    skipna :
        [optional, defaults to True, transformation]

        If False will return a NaN for any aggregation group that has a NaN.
        If True will fill all NaNs with 0 before aggregation.  Ignored if
        `groupby` is "all" or `groupby` is "months_across_years".
    agg_interval :
        DEPRECATED:
        Use the 'groupby' option instead.
    ninterval :
        DEPRECATED:
        Just prefix the number in front of the 'groupby' pandas offset code.
    """
    tsutils.printiso(
        aggregate(
            input_ts=input_ts,
            groupby=groupby,
            statistic=statistic,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            clean=clean,
            agg_interval=agg_interval,
            ninterval=ninterval,
            round_index=round_index,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            source_units=source_units,
            target_units=target_units,
            print_input=print_input,
            skipna=skipna,
        ),
        tablefmt=tablefmt,
    )


@tsutils.transform_args(statistic=tsutils.make_list)
@typic.al
def aggregate(
    input_ts="-",
    groupby=None,
    statistic: List[
        Literal[
            "mean", "sum", "std", "sem", "max", "min", "median", "first", "last", "ohlc"
        ]
    ] = "mean",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    agg_interval=None,
    ninterval=None,
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    source_units=None,
    target_units=None,
    print_input=False,
    skipna: bool = True,
):
    """Take a time series and aggregate to specified frequency."""
    aggd = {"hourly": "H", "daily": "D", "monthly": "M", "yearly": "A", "all": "all"}

    if agg_interval is not None:
        if groupby is not None:
            raise ValueError(
                tsutils.error_wrapper(
                    """
You cannot specify both 'groupby' and 'agg_interval'.  The 'agg_interval'
option is deprecated in favor of 'groupby'.
"""
                )
            )
        warnings.warn(
            tsutils.error_wrapper(
                """
The 'agg_interval' option has been deprecated in favor of 'groupby' to be
consistent with other tstoolbox commands.
"""
            )
        )
        groupby = aggd.get(agg_interval, agg_interval)

    if groupby is None:
        groupby = "D"
    else:
        groupby = aggd.get(groupby, groupby)

    if ninterval is not None:
        ninterval = int(ninterval)

        inter = None
        try:
            inter = int(groupby[0])
        except (ValueError, TypeError):
            pass

        if inter is not None:
            raise ValueError(
                """
*
*   You cannot specify the 'ninterval' option and prefix a number in the
*   'groupby' option.  The 'ninterval' option is deprecated in favor of
*   prefixing the number in the pandas offset code used in the 'groupby'
*   option.
*
"""
            )

        warnings.warn(
            """
*
*   The 'ninterval' option has been deprecated in favor of prefixing the
*   desired interval in front of the 'groupby' pandas offset code.
*
*   For example: instead of 'grouby="D"' and 'ninterval=7', you can just
*   have 'groupby="7D"'.
*
"""
        )
    else:
        ninterval = ""

    tsd = tsutils.common_kwds(
        tsutils.read_iso_ts(
            input_ts, skiprows=skiprows, names=names, index_type=index_type
        ),
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        round_index=round_index,
        dropna=dropna,
        source_units=source_units,
        target_units=target_units,
        clean=clean,
    )
    newts = pd.DataFrame()
    for method in statistic:
        if groupby == "all":
            try:
                tmptsd = pd.DataFrame(eval("""tsd.aggregate({0})""".format(method))).T
            except NameError:
                tmptsd = pd.DataFrame(eval("""tsd.aggregate('{0}')""".format(method))).T
            tmptsd.index = [tsd.index[-1]]
        elif groupby == "months_across_years":
            tmptsd = tsd.groupby(lambda x: x.month).agg(method)
            tmptsd.index = list(range(1, 13))
        else:
            tmptsd = eval(
                """tsd.resample('{0}{1}').agg(pd.Series.{2}, skipna={3})""".format(
                    ninterval, groupby, method, skipna
                )
            )
        tmptsd.columns = [tsutils.renamer(i, method) for i in tmptsd.columns]
        newts = newts.join(tmptsd, how="outer")
    if groupby == "all":
        newts.index.name = "POR"
    if groupby == "months_across_years":
        newts.index.name = "Months"
    return tsutils.return_input(print_input, tsd, newts)


aggregate.__doc__ = aggregate_cli.__doc__
