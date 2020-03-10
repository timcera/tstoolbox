#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import warnings

import mando
from mando.rst_text_formatter import RSTHelpFormatter

import numpy as np

from .. import tsutils

warnings.filterwarnings("ignore")


@mando.command("add_trend", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def add_trend_cli(
    start_offset,
    end_offset,
    start_index=0,
    end_index=-1,
    input_ts="-",
    start_date=None,
    end_date=None,
    skiprows=None,
    columns=None,
    clean=False,
    dropna="no",
    names=None,
    source_units=None,
    target_units=None,
    round_index=None,
    index_type="datetime",
    print_input=False,
    tablefmt="csv",
):
    """Add a trend.

    Adds a linear interpolated trend to the input data.  The trend
    values start at [`start_index`, `start_offset`] and end at
    [`end_index`, `end_offset`].

    Parameters
    ----------
    start_offset : float
        The starting value for the applied trend.  This is the starting
        value for the linear interpolation that will be added to the
        input data.
    end_offset : float
        The ending value for the applied trend.  This is the ending
        value for the linear interpolation that will be added to the
        input data.
    start_index : int
        [optional, default is 0, transformation]

        The starting index where `start_offset` will be initiated.  Rows
        prior to `start_index` will not be affected.
    end_index : int
        [optional, default is -1, transformation]

        The ending index where `end_offset` will be set.  Rows after
        `end_index` will not be affected.
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

    """
    tsutils.printiso(
        add_trend(
            start_offset,
            end_offset,
            start_index=start_index,
            end_index=end_index,
            input_ts=input_ts,
            columns=columns,
            clean=clean,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
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
    start_offset=[float, ["pass", []], 1],
    end_offset=[float, ["pass", []], 1],
    start_index=[int, ["pass", []], 1],
    end_index=[int, ["pass", []], 1],
)
def add_trend(
    start_offset,
    end_offset,
    start_index=0,
    end_index=-1,
    input_ts="-",
    columns=None,
    clean=False,
    start_date=None,
    end_date=None,
    dropna="no",
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    source_units=None,
    target_units=None,
    print_input=False,
):
    """Add a trend."""
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
    # Need it to be float since will be using np.nan
    ntsd = tsd.copy().astype("float64")

    ntsd.iloc[:, :] = np.nan
    ntsd.iloc[start_index, :] = float(start_offset)
    ntsd.iloc[end_index, :] = float(end_offset)
    ntsd = ntsd.interpolate(method="values")

    ntsd = ntsd + tsd

    ntsd = tsutils.memory_optimize(ntsd)
    return tsutils.return_input(print_input, tsd, ntsd, "trend")


add_trend.__doc__ = add_trend_cli.__doc__
