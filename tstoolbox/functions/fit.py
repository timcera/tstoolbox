#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import warnings

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from mando.rst_text_formatter import RSTHelpFormatter
import mando

from .. import tsutils

warnings.filterwarnings("ignore")


@mando.command("fit", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def fit_cli(
    method,
    lowess_frac=0.01,
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    source_units=None,
    target_units=None,
    print_input=False,
    tablefmt="csv",
):
    """Fit model to data.

    Parameters
    ----------
    method : str
        Any of 'lowess', 'linear' or list of same. The LOWESS technique is for
        vector data, like time-series, whereas the LOESS is a generalized
        technique that can be applied to multi-dimensional data.  For working
        with time-series LOESS and LOWESS are identical.
    lowess_frac : float
        [optional, default=0.01, range between 0 and 1]

        Fraction of data used for `method`="lowess".
    {input_ts}
    {start_date}
    {end_date}
    {skiprows}
    {names}
    {columns}
    {dropna}
    {clean}
    {source_units}
    {target_units}
    {round_index}
    {index_type}
    {print_input}
    {tablefmt}

    """
    tsutils.printiso(
        fit(
            method,
            lowess_frac=lowess_frac,
            input_ts=input_ts,
            skiprows=skiprows,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            clean=clean,
            round_index=round_index,
            index_type=index_type,
            names=names,
            source_units=source_units,
            target_units=target_units,
            print_input=print_input,
        ),
        tablefmt=tablefmt,
    )


@tsutils.validator(
    method=[str, ["domain", ["lowess", "linear"]], None],
    lowess_frac=[float, ["range", [0, 1]], None],
)
def fit(
    method,
    lowess_frac=0.01,
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    source_units=None,
    target_units=None,
    print_input=False,
):
    """Calculate fit."""
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
    method = tsutils.make_list(method)

    ntsd = pd.DataFrame()
    tmptsd = pd.DataFrame()
    for meth in method:
        if meth == "lowess":
            for cname, cdata in tsd.iteritems():
                smooth = lowess(cdata, tsd.index, frac=lowess_frac)
                index, data = np.transpose(smooth)
                ftsd = pd.DataFrame(data, index=tsd.index, columns=[cname])
                tmptsd = tmptsd.join(ftsd, how="outer")
        elif meth == "linear":
            for cname, cdata in tsd.iteritems():
                index = tsd.index.astype("l")
                index = index - index[0]
                m, b = np.polyfit(index, cdata, 1)
                data = m * index + b
                ftsd = pd.DataFrame(data, index=tsd.index, columns=[cname])
                tmptsd = tmptsd.join(ftsd, how="outer")

        tmptsd.columns = [tsutils.renamer(i, meth) for i in tmptsd.columns]
        ntsd = ntsd.join(tmptsd, how="outer")
    return tsutils.return_input(print_input, tsd, ntsd)


fit.__doc__ = fit_cli.__doc__
