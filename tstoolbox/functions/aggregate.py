#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import warnings

import mando
from mando.rst_text_formatter import RSTHelpFormatter

import pandas as pd

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
        ),
        tablefmt=tablefmt,
    )


@tsutils.validator(
    statistic=[
        str,
        [
            "domain",
            [
                "mean",
                "sum",
                "std",
                "sem",
                "max",
                "min",
                "median",
                "first",
                "last",
                "ohlc",
            ],
        ],
        None,
    ]
)
def aggregate(
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
    methods = tsutils.make_list(statistic)
    newts = pd.DataFrame()
    for method in methods:
        if groupby == "all":
            try:
                tmptsd = pd.DataFrame(eval("""tsd.aggregate({0})""".format(method))).T
            except NameError:
                tmptsd = pd.DataFrame(eval("""tsd.aggregate('{0}')""".format(method))).T
            tmptsd.index = [tsd.index[-1]]
        else:
            tmptsd = eval(
                """tsd.resample('{0}{1}').{2}()""".format(ninterval, groupby, method)
            )
        tmptsd.columns = [tsutils.renamer(i, method) for i in tmptsd.columns]
        newts = newts.join(tmptsd, how="outer")
    return tsutils.return_input(print_input, tsd, newts)


aggregate.__doc__ = aggregate_cli.__doc__
