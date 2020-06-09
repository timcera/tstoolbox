#!/sjr/beodata/local/python_linux/bin/python
"""A correlation routine."""

from __future__ import absolute_import, print_function

import mando
from mando.rst_text_formatter import RSTHelpFormatter
import numpy as np
import pandas as pd

from .. import tsutils
from . import lag


def autocorrelation(series):
    """Perform autocorrelation if lags==0"""
    n = len(series)
    data = np.asarray(series)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        return ((data[: n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0

    x = np.arange(n) + 1
    y = [r(loc) for loc in x]

    return x, y


@mando.command("correlation", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def correlation_cli(
    lags,
    method="pearson",
    input_ts="-",
    print_input=False,
    start_date=None,
    end_date=None,
    columns=None,
    clean=False,
    index_type="datetime",
    names=None,
    source_units=None,
    target_units=None,
    skiprows=None,
    tablefmt="csv",
    round_index=None,
    dropna=None,
):
    """Develop a correlation between time-series and potentially lags.

    Parameters
    ----------
    lags : str, int, or list
        If an integer will calculate all lags up to and including the
        lag number.  If a list will calculate each lag in the list.  If
        a string must be a comma separated list of integers.  If lags ==
        0 then will only cross correlate on the input time-series.

        Python example::

            lags=[2, 5, 3]

        Command line example::

            --lags='2,5,3'

    method : str
        [optional, default to "pearson"]

        Method of correlation::

            pearson : standard correlation coefficient

            kendall : Kendall Tau correlation coefficient

            spearman : Spearman rank correlation
    {print_input}
    {input_ts}
    {start_date}
    {end_date}
    {clean}
    {skiprows}
    {index_type}
    {names}
    {source_units}
    {target_units}
    {columns}
    {tablefmt}
    {round_index}
    {dropna}

    """
    tsutils.printiso(
        correlation(
            lags,
            method=method,
            input_ts=input_ts,
            print_input=print_input,
            start_date=start_date,
            end_date=end_date,
            columns=columns,
            clean=clean,
            index_type=index_type,
            names=names,
            source_units=source_units,
            target_units=target_units,
            skiprows=skiprows,
            round_index=round_index,
            dropna=dropna,
        ),
        showindex="always",
        tablefmt=tablefmt,
    )


@tsutils.validator(
    lags=[int, ["range", [0, None]], None],
    method=[str, ["domain", ["pearson", "kendall", "spearman"]], 1],
)
def correlation(
    lags,
    method="pearson",
    input_ts="-",
    print_input=False,
    start_date=None,
    end_date=None,
    columns=None,
    clean=False,
    index_type="datetime",
    names=None,
    source_units=None,
    target_units=None,
    skiprows=None,
    round_index=None,
    dropna=None,
):
    """Develop a correlation between time-series and potentially lags."""
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
    lags = tsutils.make_list(lags)

    if len(lags) == 1 and int(lags[0]) == 0:
        ntsd = pd.DataFrame()
        for cname, cdata in tsd.iteritems():
            x, y = autocorrelation(cdata)
            ntsd = ntsd.join(pd.DataFrame(y, index=x, columns=[cname]), how="outer")
        return ntsd

    ntsd = lag.lag(lags, input_ts=tsd,)
    return ntsd.corr(method=method)


correlation.__doc__ = correlation_cli.__doc__
