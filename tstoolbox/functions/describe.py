#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import warnings

import mando
from mando.rst_text_formatter import RSTHelpFormatter

from .. import tsutils

warnings.filterwarnings("ignore")


@mando.command("describe", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def describe_cli(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    clean=False,
    transpose=False,
    tablefmt="csv",
):
    """Print out statistics for the time-series.

    Parameters
    ----------
    transpose
        [optional, default is False, output format]

        If 'transpose' option is used, will transpose describe output.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {skiprows}
    {index_type}
    {names}
    {clean}
    {tablefmt}

    """
    tsutils.printiso(
        describe(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            clean=clean,
            transpose=transpose,
        ),
        showindex="always",
        tablefmt=tablefmt,
    )


@tsutils.validator(transpose=[bool, ["domain", [True, False]], 1])
def describe(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    clean=False,
    transpose=False,
):
    """Print out statistics for the time-series."""
    tsd = tsutils.common_kwds(
        tsutils.read_iso_ts(
            input_ts, skiprows=skiprows, names=names, index_type=index_type
        ),
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        dropna=dropna,
        clean=clean,
    )
    if transpose is True:
        ntsd = tsd.describe().transpose()
    else:
        ntsd = tsd.describe()

    ntsd.index.name = "Statistic"
    return ntsd


describe.__doc__ = describe_cli.__doc__
